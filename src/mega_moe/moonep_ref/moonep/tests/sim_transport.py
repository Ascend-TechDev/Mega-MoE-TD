# coding=utf-8
"""测试专用的单进程多 rank 模拟器（**非生产件**，不进 moonep 包的生产面）。

生产传输后端只有 ``moonep.buffer.ShmemStreamTransport``（aclshmem 对称堆）。
本文件为数值 oracle 测试（tests/test_sim_reference.py 等）提供无 NPU/无多进程
环境下可跑的 transport 模拟：单进程内用共享镜像 dict 模拟 R 个 rank 的
对称张量，push/pull 退化为镜像上的直接块拷贝，barrier 为真实线程屏障
（仅在多线程驱动段生效）。

接口（展平块号寻址的模拟实现，与 SymmetricArena 的 transport 鸭子约定对齐）：

- ``SimTransport(num_ranks)``               构造 R 个 rank 的模拟世界；
- ``arena_for(rank) -> SymmetricArena``     生成 rank 视角的 arena（Buffer 以 ``arena=`` 传入）；
- ``set_rank_input(rank, tag, tensor)``     预载控制面输入（"topk"/"tpe"）；
  ``allgather_ctrl(local, tag)`` 断言 ``torch.equal(local, 预载[rank])`` 后返回全组堆叠——
  模拟 allgather 的同时防止测试驱动的阶段轮转错误。

**展平块号语义**：register 时镜像存 ``chunk_blocks * block_elems`` 的 padded
张量（padding 区可寻址、无逻辑含义，与生产侧一致），返回逻辑形状视图；
push/pull 按 ``pe = flat // chunk_blocks`` / ``off = flat % chunk_blocks``
拆分后直接在 ``mirror[name][pe]`` 的 padded 张量上块拷贝，本 rank 与远端
同一条路径（天然统一）；pull 返回 ``[n, block_elems]``（应答序 == 请求序，
逐位对应）。

集合时序约束：测试驱动必须按 API 阶段轮转（例如全部 rank 的 dispatch 完成后
再逐 rank combine），与真实全组语义一致。
"""

import threading

import torch

from moonep.buffer import SymmetricArena


class _SimRankView:
    """SimTransport 的单 rank 视角：实现 transport 鸭子接口，委托回 SimTransport。"""

    def __init__(self, sim: "SimTransport", rank: int):
        self._sim = sim
        self.rank = rank

    @property
    def world_size(self):
        return self._sim.num_ranks

    def _register(self, name, tensor, block_elems, chunk_blocks):
        return self._sim._do_register(self.rank, name, tensor, block_elems,
                                      chunk_blocks)

    def _tensor(self, name):
        return self._sim._do_tensor(self.rank, name)

    def _push_all(self, name, flat_indices, payload):
        return self._sim._do_push_all(self.rank, name, flat_indices, payload)

    def _pull_all(self, name, flat_indices):
        return self._sim._do_pull_all(self.rank, name, flat_indices)

    def _pull_into(self, name, src_flats, dst_flats):
        return self._sim._do_pull_into(self.rank, name, src_flats, dst_flats)

    def _barrier(self):
        return self._sim._do_barrier()

    def _allgather_ctrl(self, local, tag):
        return self._sim._do_allgather_ctrl(self.rank, local, tag)


class SimTransport:
    """单进程模拟 R 个 rank 的 transport：共享镜像 dict{name: [R]个 padded tensor}
    + 展平块号拆分后的直接块拷贝。"""

    def __init__(self, num_ranks: int):
        if not isinstance(num_ranks, int) or num_ranks <= 0:
            raise ValueError(f"num_ranks 必须为正整数，得到 {num_ranks!r}")
        self.num_ranks = num_ranks
        # 共享镜像：name -> [R] 个各 rank 的 padded 物理张量（None 表示该 rank 尚未 register）
        self._mirror = {}
        # name -> (block_elems, chunk_blocks, dtype, logical_shape)
        self._block = {}
        # 预载的控制面输入：(rank, tag) -> tensor
        self._ctrl = {}
        # 跨 rank 屏障：差异化 rank 结构下 planning/dispatch/combine 的 kernel
        # 内屏障存在 rank 间即时数据依赖（如 rank1 代算 rank0 的 C1 并回写），
        # 测试驱动须以多线程组织（每 rank 一个线程），屏障为真实 rendezvous。
        # armed 计数器：仅在 run_threaded 驱动段内生效；非线程段（主线程单
        # rank 调用、destroy 等）退化为 no-op（顺序语义）。
        self._bar = threading.Barrier(num_ranks)
        self._armed = 0

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def arena_for(self, rank: int) -> SymmetricArena:
        """生成 rank 视角的 SymmetricArena。"""
        if not isinstance(rank, int) or not 0 <= rank < self.num_ranks:
            raise ValueError(f"rank 必须在 [0, {self.num_ranks}) 内，得到 {rank!r}")
        return SymmetricArena(_SimRankView(self, rank))

    def set_rank_input(self, rank: int, tag: str, tensor: torch.Tensor) -> None:
        """预载 rank 的控制面输入（allgather_ctrl 的对照真值）。"""
        if not isinstance(rank, int) or not 0 <= rank < self.num_ranks:
            raise ValueError(f"rank 越界：{rank!r}")
        if not isinstance(tag, str) or not tag:
            raise ValueError(f"tag 必须为非空字符串，得到 {tag!r}")
        self._ctrl[(rank, tag)] = tensor.detach().clone()

    def _arm(self):
        """进入多线程驱动段：屏障生效（测试驱动在启动 rank 线程前调用）。"""
        self._armed += 1

    def _disarm(self):
        """离开多线程驱动段：屏障退化为 no-op。"""
        self._armed -= 1

    def _abort(self):
        """某 rank 线程异常时调用：打破屏障释放其他线程（避免死锁）。"""
        try:
            self._bar.abort()
        except Exception:
            pass

    def _do_barrier(self):
        if self._armed > 0:
            self._bar.wait(timeout=600)   # 超时响亮失败，不挂死
        return None

    # ------------------------------------------------------------------
    # 以下供 _SimRankView 委托调用
    # ------------------------------------------------------------------
    def _do_register(self, rank, name, tensor, block_elems, chunk_blocks):
        """登记 rank 的物理张量到共享镜像；全组须同名同序同形同 chunk_blocks。

        镜像存 ``chunk_blocks * block_elems`` 的 padded 张量（零初始化，
        padding 区内容确定），逻辑内容拷入前缀；返回逻辑形状视图（基址即
        chunk 基址，与生产侧 from_blob 行为一致）。
        """
        meta = (int(block_elems), int(chunk_blocks), tensor.dtype,
                tuple(tensor.shape))
        if name not in self._mirror:
            self._mirror[name] = [None] * self.num_ranks
            self._block[name] = meta
        else:
            assert self._block[name] == meta, (
                f"对称张量 {name!r} 各 rank 的注册元信息不一致："
                f"{self._block[name]} vs {meta}"
            )
        padded = torch.zeros(chunk_blocks * block_elems, dtype=tensor.dtype)
        padded[: tensor.numel()] = tensor.reshape(-1)
        logical = padded[: tensor.numel()].view(tensor.shape)
        # 镜像持有 padded 张量（展平坐标可寻址 padding 区），逻辑视图给调用方
        self._mirror[name][rank] = padded
        return logical

    def _do_tensor(self, rank, name):
        padded = self._mirror[name][rank]
        if padded is None:
            raise KeyError(f"rank {rank} 尚未 register 对称张量 {name!r}")
        return padded

    def _do_push_all(self, rank, name, flat_indices, payload):
        """按展平块号写块：pe = flat // chunk_blocks，off = flat % chunk_blocks；
        本 rank 与远端同一条镜像拷贝路径。"""
        block, chunk_blocks = self._block[name][0], self._block[name][1]
        flat_list = flat_indices.tolist()
        for i, f in enumerate(flat_list):
            pe, off = f // chunk_blocks, f % chunk_blocks
            dst_phys = self._do_tensor(pe, name)
            view = dst_phys.view(-1, block)
            view[off] = payload[i].to(view.dtype)

    def _do_pull_all(self, rank, name, flat_indices):
        """按展平块号读块；返回 [n, block_elems]（应答序 == 请求序，深拷贝）。"""
        block, chunk_blocks = self._block[name][0], self._block[name][1]
        dtype = self._block[name][2]
        flat_list = flat_indices.tolist()
        rows = []
        for f in flat_list:
            pe, off = f // chunk_blocks, f % chunk_blocks
            src_phys = self._do_tensor(pe, name)
            view = src_phys.view(-1, block)
            rows.append(view[off].clone())
        # n=0 时保持 [0, block] 形状与 dtype（与生产侧 torch.empty 一致）
        if not rows:
            return torch.empty(0, block, dtype=dtype)
        return torch.stack(rows)

    def _do_pull_into(self, rank, name, src_flats, dst_flats):
        """按展平块号读块并直接写回本 rank 镜像的目标块（对称堆直达语义）。

        src 在远端镜像、dst 在本 rank 镜像上取块视图，逐对块拷贝（本 rank
        源与远端源同一条路径，与生产侧 _pull_into 的堆内/远端分叉等价）。
        """
        block, chunk_blocks = self._block[name][0], self._block[name][1]
        dst_view = self._do_tensor(rank, name).view(-1, block)
        for s, d in zip(src_flats.tolist(), dst_flats.tolist()):
            pe, off = s // chunk_blocks, s % chunk_blocks
            src_view = self._do_tensor(pe, name).view(-1, block)
            dst_view[d] = src_view[off]

    def _do_allgather_ctrl(self, rank, local, tag):
        """断言 local 与预载逐位一致，返回全组堆叠 [R, *shape]。"""
        preloaded = self._ctrl.get((rank, tag))
        assert preloaded is not None, (
            f"rank {rank} 的控制面输入 tag={tag!r} 未预载（set_rank_input）"
        )
        assert torch.equal(local, preloaded), (
            f"rank {rank} 的 allgather_ctrl 输入与预载不一致（tag={tag!r}）——"
            f"测试驱动的阶段轮转或输入构造有误"
        )
        stack = []
        for r in range(self.num_ranks):
            t = self._ctrl.get((r, tag))
            assert t is not None, (
                f"rank {r} 的控制面输入 tag={tag!r} 未预载（set_rank_input）"
            )
            assert t.shape == local.shape and t.dtype == local.dtype, (
                f"tag={tag!r} 各 rank 控制面输入形状/dtype 不一致"
            )
            stack.append(t)
        return torch.stack(stack, dim=0)
