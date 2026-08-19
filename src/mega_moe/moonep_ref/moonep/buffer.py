# coding=utf-8
"""moonep 对称内存封装（SymmetricArena）+ 全局 arena 注册表 + ShmemStreamTransport（生产传输后端）。

本模块对应设计契约 docs/design.md §3。kernel / launch 体内的**唯一跨 rank 通道**
是 SymmetricArena，所有跨 rank 数据移动收敛为如下集合词汇（块粒度，元素单位 =
register 时声明的 block_elems）：

- ``register(name, tensor, block_elems, *, chunk_blocks=None)``  登记对称张量
  （全组必须同名同序同 chunk_blocks 调用），返回本 rank 物理张量（逻辑形状
  视图）；内部建立 ``data_ptr -> 注册项`` 注册表；
- ``tensor(name)``                         取本 rank 物理张量；
- ``push_all(local_tensor, flat_indices, payload)``   按 **int64 展平块号**
  把若干块写入目的 rank 的对称张量（**按本地张量对象的 data_ptr 解析注册
  身份**，不是按 name），集合调用；
- ``pull_all(local_tensor, flat_indices) -> Tensor``  按展平块号读出若干块，
  返回 ``[n, block_elems]``（**与请求顺序一一对应**），集合调用；
- ``barrier()`` / ``allgather_ctrl(local, tag)``  全组屏障 / 控制面 allgather，
  ``tag`` 区分调用点（"topk"/"tpe"），返回 ``[R, *shape]``。

**展平块号寻址**（对齐 GPU MoonEP 的原始地址运算形态）：每个已注册对称
张量拥有一个全组一致的 **chunk 步长 chunk_blocks**（每 rank 占用的块数，
含可选 padding——对应 GPU 的 NvS_padded / meta_chunk_padded）。跨 rank
坐标一律写成展平块号

    flat = pe * chunk_blocks + block_off
    pe  = flat // chunk_blocks      （目的 / 源 rank）
    off = flat % chunk_blocks       （该 rank chunk 内块偏移）

pe == 本 rank 的项由 transport 走本地拷贝（GPU 语义：所有 chunk RW 映射、
本地 VA 直写）；其余项走 shmem RMA 原语。padding 区
``[logical_blocks, chunk_blocks)`` 可寻址但无逻辑含义（对应 GPU padding 行）。

模块级全局 arena 注册表（launch_prefetch / launch_grad_reduce 等**无 ctx** 的
launch_* 函数取 arena 的通道）：

- ``register_arena(arena)``      Buffer.__init__ 调用，登记一个 arena；
- ``resolve_arena_for(tensor)``  按 tensor 的 data_ptr 在所有已注册 arena 中
                                 查找其注册归属；
- ``unregister_arena(arena)``    Buffer.destroy 调用，反登记（契约 §5
                                 "含全局注册表反登记"的落地接口）。

生产 transport（唯一内置后端；数值验证用的单进程模拟器下沉为测试专用件
``tests/sim_transport.py``，不属于本包生产面）：

- ``ShmemStreamTransport(group, heap_size)``  生产路径：构造时惰性 import
  shmem / torch_npu（未安装抛 ImportError 并附 source_code/shmem 安装指引）；
  运行时为 aclshmemx_init_attr 初始化（uniqueid 优先、ipport 兜底），register 时
  aclshmemx_align(2MB) 对称堆分配 + from_blob 封装 torch 张量；push/pull 用
  aclshmemx_putmem_on_stream / aclshmemx_getmem_on_stream（逐块循环 +
  quiet + 流同步）；barrier 用 aclshmemx_barrier_all_on_stream（pybind 未导出，
  经 ctypes 直调 libshmem.so）；allgather_ctrl 走 HCCL group。
  API 签名以 source_code/shmem/include/host/ 头文件为准（见各方法注释）。

transport 对 SymmetricArena 暴露的鸭子接口（模块内私有约定）：

- ``rank`` / ``world_size``                        本 rank 序号 / 组内 rank 数；
- ``_register(name, tensor, block_elems, chunk_blocks)``   登记并返回本 rank
  物理张量（逻辑形状视图）；
- ``_tensor(name)``                                取本 rank 物理张量；
- ``_push_all(name, flat_indices, payload)`` / ``_pull_all(name, flat_indices)``
  展平块号块搬运；
- ``_barrier()`` / ``_allgather_ctrl(local, tag)``              同步 / 控制面汇聚。

dtype 契约（全包统一）：meta/int32、payload/bf16、route weight/fp32、梯度/fp32；
展平块号一律 int64。参考实现优先语义精确，不做性能技巧。
"""

import ctypes
import os
import weakref

import torch
import torch.distributed as dist

__all__ = [
    "SymmetricArena",
    "ShmemStreamTransport",
    "register_arena",
    "resolve_arena_for",
    "unregister_arena",
]


# shmem 未安装时的安装指引（source_code/shmem 提供 setup.py）
_SHMEM_INSTALL_HINT = (
    "ShmemStreamTransport 依赖 cann-shmem 运行包，但当前环境未安装。\n"
    "安装指引：进入 source_code/shmem 目录执行 `pip install .`"
    "（等价 `python3 setup.py install`），并确保 NPU 驱动与 CANN 环境就绪。"
)

_TORCH_NPU_INSTALL_HINT = (
    "ShmemStreamTransport 依赖 torch_npu，但当前环境未安装。\n"
    "请先安装与当前 torch 版本匹配的 torch_npu（Ascend PyTorch 适配包）。"
)


def _check_group_meta(group, world_size, meta):
    """用 all_gather_object 校验全组 register 元信息一致（同名同序同形状）。"""
    metas = [None] * world_size
    dist.all_gather_object(metas, meta, group=group)
    if any(m != meta for m in metas):
        raise RuntimeError(
            f"对称张量 register 全组不一致：本 rank 元信息 {meta}，全组 {metas}。"
            "契约要求全组同名同序同 chunk_blocks 调用 register。"
        )


def _all_gather_ctrl(group, world_size, local):
    """allgather_ctrl 的 HCCL/gloo 实现：all_gather_into_tensor，返回 [R,*shape]。"""
    local_c = local.contiguous()
    gathered = torch.empty(
        world_size * local_c.numel(), dtype=local_c.dtype, device=local_c.device
    )
    dist.all_gather_into_tensor(gathered, local_c.view(-1), group=group)
    return gathered.view(world_size, *local_c.shape)


class SymmetricArena:
    """一个通信域（EP group）的对称内存注册表 + 传输面（kernel 体内的唯一跨 rank 通道）。

    签名与语义以 docs/design.md §3 为准。本类只做注册表管理（name 表 +
    ``data_ptr -> 注册项`` 指针表）与入参校验，实际数据搬运委托给构造时
    传入的 transport。
    """

    def __init__(self, transport):
        self._transport = transport
        # name -> (本 rank 物理张量（逻辑视图）, block_elems, chunk_blocks)
        self._regs = {}
        # data_ptr -> (name, block_elems, chunk_blocks)（契约 §3 规定的注册表；
        # push_all/pull_all 按本地张量对象的 data_ptr 在此解析注册身份）
        self._ptr_index = {}

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def transport(self):
        """底层 transport（ShmemStreamTransport；测试侧可为 tests/sim_transport.py 的模拟器视角）。"""
        return self._transport

    @property
    def rank(self) -> int:
        return self._transport.rank

    @property
    def world(self) -> int:
        """组内 rank 数（契约 §3 属性名）。"""
        return self._transport.world_size

    @property
    def world_size(self) -> int:
        """``world`` 的兼容别名（torch.distributed 风格命名）。"""
        return self._transport.world_size

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register(self, name: str, tensor: torch.Tensor, block_elems: int,
                 *, chunk_blocks: "int | None" = None) -> torch.Tensor:
        """登记对称张量（全组同名同序同 chunk_blocks 调用）；返回本 rank 物理张量。

        block_elems：块粒度（元素数）。张量元素总数必须能被 block_elems 整除，
        第 i 块即 ``tensor.view(-1, block_elems)[i]``（**逻辑块**）。

        chunk_blocks：展平块号的 **chunk 步长**（每 rank 占用的块数，含
        padding），必须 >= 逻辑块数；None 时取逻辑块数（无 padding）。
        展平坐标 ``flat = pe * chunk_blocks + block_off`` 据此拆分
        ``pe = flat // chunk_blocks`` / ``off = flat % chunk_blocks``。
        ShmemStreamTransport 按 ``chunk_blocks * block_elems`` 分配对称堆，
        ``[逻辑块数, chunk_blocks)`` 为 padding 区（可寻址、无逻辑含义，
        对应 GPU 的 NvS_padded / meta_chunk_padded 余量）。

        返回值为入参逻辑形状的本 rank 物理张量（视图，基址即 chunk 基址）。
        内部建立 ``data_ptr -> 注册项`` 注册表，供 push_all/pull_all 按本地
        张量对象解析注册身份。注意：ShmemStreamTransport 下返回的是对称堆
        上的新张量（内容已从入参拷入），调用方此后应使用返回值而非入参。
        """
        if name in self._regs:
            raise ValueError(f"对称张量 {name!r} 重复 register")
        if not isinstance(block_elems, int) or block_elems <= 0:
            raise ValueError(f"block_elems 必须为正整数，得到 {block_elems!r}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"对称张量 {name!r} 必须为 torch.Tensor，得到 {type(tensor)!r}")
        if not tensor.is_contiguous():
            raise ValueError(f"对称张量 {name!r} 必须连续（contiguous）")
        if tensor.numel() % block_elems != 0:
            raise ValueError(
                f"对称张量 {name!r} 元素数 {tensor.numel()} 不能按块 {block_elems} 整除"
            )
        logical_blocks = tensor.numel() // block_elems
        if chunk_blocks is None:
            chunk_blocks = logical_blocks
        if not isinstance(chunk_blocks, int) or chunk_blocks < logical_blocks:
            raise ValueError(
                f"chunk_blocks 必须为 >= 逻辑块数 {logical_blocks} 的整数，"
                f"得到 {chunk_blocks!r}"
            )
        phys = self._transport._register(name, tensor, block_elems, chunk_blocks)
        ptr = phys.data_ptr()
        if ptr in self._ptr_index:
            # 同一物理地址注册两个名字会让 data_ptr 解析产生二义，直接拒绝
            raise ValueError(
                f"对称张量 {name!r} 的 data_ptr 与已注册张量 "
                f"{self._ptr_index[ptr][0]!r} 冲突"
            )
        self._regs[name] = (phys, block_elems, chunk_blocks)
        self._ptr_index[ptr] = (name, block_elems, chunk_blocks)
        return phys

    def tensor(self, name: str) -> torch.Tensor:
        """取本 rank 的物理张量（ShmemStreamTransport 下为对称堆张量）。"""
        if name not in self._regs:
            raise KeyError(f"对称张量 {name!r} 尚未 register")
        return self._regs[name][0]

    def _resolve(self, local_tensor: torch.Tensor):
        """按本地张量对象（data_ptr）解析注册身份，返回
        (name, 物理张量, block_elems, chunk_blocks)。"""
        if not isinstance(local_tensor, torch.Tensor):
            raise TypeError(
                f"push_all/pull_all 第一个参数必须为已注册的本地 torch.Tensor，"
                f"得到 {type(local_tensor)!r}"
            )
        entry = self._ptr_index.get(local_tensor.data_ptr())
        if entry is None:
            raise KeyError(
                "传入张量未在本 arena 注册（按 data_ptr 解析失败）；"
                "push_all/pull_all 只接受 register 返回的本 rank 物理张量"
            )
        name, block_elems, chunk_blocks = entry
        return name, self._regs[name][0], block_elems, chunk_blocks

    def _owns_ptr(self, data_ptr: int) -> bool:
        """全局注册表用：data_ptr 是否为本 arena 已注册张量的基址。"""
        return data_ptr in self._ptr_index

    # ------------------------------------------------------------------
    # 入参校验
    # ------------------------------------------------------------------
    def _check_flat(self, name, phys, block_elems, chunk_blocks, flat_indices,
                    payload=None):
        """展平块号校验：一维 int64、[0, world*chunk_blocks) 界内（含 padding 区）、
        push 侧 payload [n, block] 且 dtype 匹配。"""
        if not isinstance(flat_indices, torch.Tensor) or \
                flat_indices.dtype != torch.int64 or flat_indices.dim() != 1:
            raise ValueError(
                f"展平块号必须为一维 int64 张量，得到 "
                f"{type(flat_indices)!r}"
                f"{getattr(flat_indices, 'dtype', '')} "
                f"shape={tuple(getattr(flat_indices, 'shape', ()))}"
            )
        n = flat_indices.numel()
        if payload is not None:
            if payload.dim() != 2 or payload.shape[0] != n or \
                    payload.shape[1] != block_elems:
                raise ValueError(
                    f"push_all payload 形状应为 [{n}, {block_elems}]，"
                    f"得到 {tuple(payload.shape)}"
                )
            if payload.dtype != phys.dtype:
                raise ValueError(
                    f"push_all payload dtype {payload.dtype} 与注册 dtype "
                    f"{phys.dtype} 不一致"
                )
        if n > 0:
            lo = int(flat_indices.min().item())
            hi = int(flat_indices.max().item())
            limit = self.world * chunk_blocks
            if lo < 0 or hi >= limit:
                raise ValueError(
                    f"展平块号越界：[{lo}, {hi}]，合法范围 [0, {limit})"
                    f"（name={name!r}，world={self.world}，"
                    f"chunk_blocks={chunk_blocks}；换算 pe=flat//"
                    f"{chunk_blocks}、off=flat%{chunk_blocks}，"
                    f"请检查 kernel 的展平坐标表达式）"
                )

    # ------------------------------------------------------------------
    # 集合词汇（展平块号寻址：flat = pe * chunk_blocks + block_off）
    # ------------------------------------------------------------------
    def push_all(self, local_tensor: torch.Tensor,
                 flat_indices: torch.Tensor, payload: torch.Tensor) -> None:
        """把 payload[i] 写入展平块号 flat_indices[i] 指向的 (pe, off) 块：
        ``pe = flat // chunk_blocks``（目的 rank），``off = flat % chunk_blocks``
        （该 rank chunk 内块偏移）。pe == 本 rank 的项走本地拷贝。
        集合调用（无写也以 n=0 参与，全组须在同一 API 阶段内调用）。"""
        name, phys, block_elems, chunk_blocks = self._resolve(local_tensor)
        self._check_flat(name, phys, block_elems, chunk_blocks,
                         flat_indices, payload)
        self._transport._push_all(name, flat_indices, payload)

    def pull_all(self, local_tensor: torch.Tensor,
                 flat_indices: torch.Tensor) -> torch.Tensor:
        """按展平块号读出若干块，返回 ``[n, block_elems]`` 张量
        （**与请求顺序一一对应**，不按 rank 分组）。pe == 本 rank 的项走本地
        取数。集合调用（全组须在同一 API 阶段内调用）。"""
        name, phys, block_elems, chunk_blocks = self._resolve(local_tensor)
        self._check_flat(name, phys, block_elems, chunk_blocks, flat_indices)
        return self._transport._pull_all(name, flat_indices)

    def pull_into(self, local_tensor: torch.Tensor,
                  src_flats: torch.Tensor,
                  dst_flats: torch.Tensor) -> None:
        """按展平块号 ``src_flats[i]`` 读块，**直接写入本 rank chunk 内
        ``dst_flats[i]`` 块**（对称堆直达，不经过普通内存落盘张量）。

        与 ``pull_all`` 的区别：pull_all 把应答物化到新分配的普通张量返回，
        调用方再拷贝到目的地；pull_into 的目的地址即本 rank 对称堆内的块，
        远端数据一跳到位（对齐 GPU 版 TMA 直写 ``prefetch_buffers`` 的语义，
        省一次 device 内拷贝与落盘张量分配）。

        ``src_flats``：展平块号（``pe = flat // chunk_blocks`` 为源 rank，
        ``off = flat % chunk_blocks`` 为源 chunk 内块偏移），pe == 本 rank
        的项走堆内本地拷贝。``dst_flats``：本 rank 自身 chunk 内的目标块号，
        须落在 ``[0, chunk_blocks)``，长度与 ``src_flats`` 一致；目标块互不
        重叠由调用方保证（prefetch 的槽行天然互异）。集合调用（无请求也须
        以 n=0 参与，全组在同一 API 阶段内调用）。"""
        name, phys, block_elems, chunk_blocks = self._resolve(local_tensor)
        self._check_flat(name, phys, block_elems, chunk_blocks, src_flats)
        if not isinstance(dst_flats, torch.Tensor) or \
                dst_flats.dtype != torch.int64 or dst_flats.dim() != 1 or \
                dst_flats.numel() != src_flats.numel():
            raise ValueError(
                f"目标块号必须为与源块号等长的一维 int64 张量，得到 "
                f"{type(dst_flats)!r}"
                f"{getattr(dst_flats, 'dtype', '')} "
                f"shape={tuple(getattr(dst_flats, 'shape', ()))}"
            )
        if dst_flats.numel() > 0:
            dlo = int(dst_flats.min().item())
            dhi = int(dst_flats.max().item())
            if dlo < 0 or dhi >= chunk_blocks:
                raise ValueError(
                    f"目标块号越界：[{dlo}, {dhi}]，合法范围 "
                    f"[0, {chunk_blocks})（name={name!r}；pull_into 的目标必须"
                    f"是本 rank chunk 内的块）"
                )
        self._transport._pull_into(name, src_flats, dst_flats)

    def barrier(self) -> None:
        """全组屏障。"""
        self._transport._barrier()

    def allgather_ctrl(self, local: torch.Tensor, tag: str) -> torch.Tensor:
        """返回 [R,*shape] 控制面汇聚；tag 区分调用点（"topk"/"tpe"）。"""
        return self._transport._allgather_ctrl(local, tag)


# ----------------------------------------------------------------------
# 模块级全局 arena 注册表（契约 §3）
#
# 用途：launch_prefetch / launch_grad_reduce 等**无 ctx** 的 launch 函数，
# 只拿到对称张量本身，经 resolve_arena_for 按 data_ptr 反查其注册归属的
# arena。Buffer.__init__ 调 register_arena 登记，Buffer.destroy 调
# unregister_arena 反登记（契约 §5）。
#
# 用 WeakSet 持有 arena：即便调用方漏掉 unregister_arena，arena 被 GC 后
# 注册表自动失效，不会悬垂引用阻止回收。
# ----------------------------------------------------------------------
_GLOBAL_ARENAS = weakref.WeakSet()


def register_arena(arena: SymmetricArena) -> None:
    """把 arena 登记进全局注册表（Buffer.__init__ 调用，幂等）。"""
    if not isinstance(arena, SymmetricArena):
        raise TypeError(f"register_arena 只接受 SymmetricArena，得到 {type(arena)!r}")
    _GLOBAL_ARENAS.add(arena)


def unregister_arena(arena: SymmetricArena) -> None:
    """把 arena 从全局注册表反登记（Buffer.destroy 调用，幂等）。"""
    _GLOBAL_ARENAS.discard(arena)


def resolve_arena_for(tensor: torch.Tensor) -> SymmetricArena:
    """按 tensor 的 data_ptr 在所有已注册 arena 中查找其注册归属。

    命中唯一 arena 则返回；未命中抛 KeyError；命中多个（同一 data_ptr 被
    多个存活 arena 注册，属调用方使用错误）抛 RuntimeError。
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"resolve_arena_for 只接受 torch.Tensor，得到 {type(tensor)!r}")
    ptr = tensor.data_ptr()
    hits = [arena for arena in list(_GLOBAL_ARENAS) if arena._owns_ptr(ptr)]
    if not hits:
        raise KeyError(
            "按 data_ptr 未在任何已注册 arena 中找到该张量的注册归属；"
            "请确认对称张量已经 arena.register 且所在 Buffer 已 register_arena"
        )
    if len(hits) > 1:
        raise RuntimeError(
            "同一 data_ptr 命中多个已注册 arena，全局注册表存在二义；"
            "请检查是否有 Buffer 未 destroy 反登记或重复 register_arena"
        )
    return hits[0]


class ShmemStreamTransport:
    """生产路径 transport：aclshmem 对称堆 + stream 上的 host 侧 RMA。

    构造时惰性 import shmem 与 torch_npu（未安装抛 ImportError 并附安装指引）。
    初始化（对应 source_code/shmem/include/host/init/shmem_host_init.h）：
    uniqueid 优先——rank0 调 aclshmemx_get_uniqueid 生成 uniqueid 并经 HCCL group
    广播，全组走 aclshmemx_set_attr_uniqueid_args +
    aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID)（pybind 复合接口
    ``aclshmem_init_using_unique_id``）；若设置环境变量 MOONEP_SHMEM_IPPORT
    （如 "tcp://127.0.0.1:8666"）则走 ipport 兜底——全组各自填 InitAttr 后走
    aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT)（pybind 接口 ``aclshmem_init``）。

    register 时按 ``chunk_blocks * block_elems`` 以 shmem_host_heap.h 的
    ``aclshmemx_align(2MB, nbytes)`` 分配对称堆（pybind 当前仅暴露旧名
    ``aclshmem_align``，优先尝试新名），from_blob 封装成 padded 张量后切出
    逻辑形状视图返回（优先 shmem 包的 ``construct_tensor_from_ptr``，兜底
    torch_npu 私有构造接口）。push/pull 用 shmem_host_rma.h 的
    ``aclshmemx_putmem_on_stream`` / ``aclshmemx_getmem_on_stream``
    （逐块循环 + ``aclshmemx_quiet_on_stream`` + 流同步），barrier 用
    shmem_host_cc.h 的 ``aclshmemx_barrier_all_on_stream``（pybind 未导出，
    经 ctypes 直调 libshmem.so），allgather_ctrl 走 HCCL group。
    本路径仅在真实 NPU 环境可用。
    """

    # 对称堆分配对齐：2MB（2 的幂，满足 aclshmemx_align 对齐参数要求）
    _ALIGN_BYTES = 2 * 1024 * 1024

    def __init__(self, group=None, heap_size: int = 1024 * 1024 * 1024):
        # 惰性导入：未安装时抛带安装指引的 ImportError
        try:
            import shmem
        except ImportError as exc:
            raise ImportError(_SHMEM_INSTALL_HINT) from exc
        try:
            import torch_npu  # noqa: F401  （注册 npu 设备与私有构造接口）
        except ImportError as exc:
            raise ImportError(_TORCH_NPU_INSTALL_HINT) from exc
        if not isinstance(heap_size, int) or heap_size <= 0:
            raise ValueError(f"heap_size 必须为正整数（字节），得到 {heap_size!r}")

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "ShmemStreamTransport: torch.distributed 未初始化。"
                "请先 init_process_group（HCCL），并确认已安装正确的 shmem 包；"
                "单 rank 调试也须初始化单卡进程组。"
            )
        self._shmem = shmem
        self._torch_npu = torch_npu
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.heap_size = heap_size
        # name -> (padded 堆张量, 逻辑视图, block_elems, chunk_blocks, 堆指针, 字节数)
        self._regs = {}
        self._libshmem = None  # ctypes 句柄，按需解析（barrier 用）

        # 先初始化 aclshmem 运行时（首个 shmem 公共 API 调用会触发 shmem 包的
        # 惰性原生库加载与 _pyshmem 子模块挂载），之后才能安全取 shmem._pyshmem
        self._init_shmem_runtime()
        self._pyshmem = shmem._pyshmem

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_shmem_runtime(self):
        """初始化 aclshmem 运行环境（幂等：已初始化则跳过）。

        uniqueid 优先：rank0 生成 uniqueid 并经 HCCL group 广播，全组调
        aclshmem_init_using_unique_id（内部即 aclshmemx_set_attr_uniqueid_args +
        aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID)，见
        shmem_host_init.h:53/83）；若设置环境变量 MOONEP_SHMEM_IPPORT
        （如 "tcp://127.0.0.1:8666"）则走 ipport 路径（InitAttr +
        aclshmem_init，内部即 aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT)）。
        """
        shmem = self._shmem
        # aclshmemx_init_status: 0=未初始化 1=堆已建 2=已初始化
        if int(shmem.aclshmemx_init_status()) == int(shmem.InitStatus.INITIALIZED):
            return

        ipport = os.environ.get("MOONEP_SHMEM_IPPORT")
        if ipport:
            # ipport 路径：全组各自构造 InitAttr 后初始化
            attr = shmem.InitAttr()
            attr.my_rank = self.rank
            attr.n_ranks = self.world_size
            attr.local_mem_size = self.heap_size
            attr.ip_port = ipport
            attr.option_attr.data_op_engine_type = shmem.OpEngineType.MTE
            ret = shmem.aclshmem_init(attr)
            if ret != 0:
                raise RuntimeError(f"aclshmem_init(ipport) 失败，ret={ret}")
            return

        # uniqueid 路径：rank0 生成，经 HCCL group 广播字节串
        uid_bytes = None
        if self.rank == 0:
            uid_bytes = bytes(shmem.aclshmem_get_unique_id())
        uid_bytes = self._broadcast_bytes(uid_bytes)
        ret = shmem.aclshmem_init_using_unique_id(
            self.rank, self.world_size, self.heap_size, uid_bytes
        )
        if ret != 0:
            raise RuntimeError(f"aclshmem_init_using_unique_id 失败，ret={ret}")

    def _group_device(self):
        """HCCL 集合通信用设备：hccl 后端用当前 npu，其余（gloo 等）用 CPU。"""
        backend = dist.get_backend(self.group)
        if backend == "gloo":
            return torch.device("cpu")
        return torch.device(f"npu:{torch.npu.current_device()}")

    def _broadcast_bytes(self, payload):
        """经 HCCL group 从 rank0 广播字节串（长度前缀 + uint8 负载）。"""
        device = self._group_device()
        len_buf = torch.empty(1, dtype=torch.int64, device=device)
        if self.rank == 0:
            len_buf.fill_(len(payload))
        dist.broadcast(len_buf, src=0, group=self.group)
        n = int(len_buf.item())
        byte_buf = torch.empty(n, dtype=torch.uint8, device=device)
        if self.rank == 0:
            byte_buf.copy_(torch.tensor(list(payload), dtype=torch.uint8))
        dist.broadcast(byte_buf, src=0, group=self.group)
        return bytes(byte_buf.cpu().tolist())

    def _current_stream(self):
        """当前 npu stream 的裸指针（aclrtStream，int 形态）。"""
        return torch.npu.current_stream().npu_stream

    def _load_libshmem(self):
        """按 ctypes 解析 libshmem.so（aclshmemx_barrier_all_on_stream 的通道，
        pybind 当前未导出该符号，见 shmem_host_cc.h:67）。"""
        if self._libshmem is not None:
            return self._libshmem
        try:
            # 优先从 shmem 包内的 backend 目录加载，保证与 pybind 用的是同一份
            from shmem import _get_backend_so_dir

            lib = ctypes.CDLL(os.path.join(str(_get_backend_so_dir()), "libshmem.so"))
        except Exception:
            lib = ctypes.CDLL("libshmem.so")
        lib.aclshmemx_barrier_all_on_stream.restype = None
        lib.aclshmemx_barrier_all_on_stream.argtypes = [ctypes.c_void_p]
        self._libshmem = lib
        return lib

    def _tensor_from_ptr(self, data_ptr, shape, dtype):
        """from_blob 封装：把对称堆指针包装成 torch 张量（不占有内存，禁止 resize）。

        优先用 shmem 包自带的 construct_tensor_from_ptr（运行时初始化后经
        shmem 包 ``_ensure_native`` 挂载到顶层命名空间）；兜底走 torch_npu
        私有构造接口（同一语义的本地实现）。
        """
        device = torch.device(f"npu:{torch.npu.current_device()}")
        shape = tuple(int(s) for s in shape)
        from_ptr = getattr(self._shmem, "construct_tensor_from_ptr", None)
        if from_ptr is not None:
            return from_ptr(data_ptr, shape, dtype, device)

        # ---- 兜底：torch_npu 私有接口手写 from_blob ----
        torch_npu = self._torch_npu
        numel = 1
        for s in shape:
            numel *= s
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        # 连续 stride
        stride = []
        acc = 1
        for s in reversed(shape):
            stride.append(acc)
            acc *= s
        stride = tuple(reversed(stride))
        storage = torch_npu._C._construct_storage_from_data_pointer(
            data_ptr, device, nbytes
        )
        metadata = {
            "data_ptr": data_ptr,
            "device": device,
            "nbytes": nbytes,
            "dtype": dtype,
            "size": shape,
            "stride": stride,
            "storage_offset": 0,
        }
        return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(
            metadata, storage
        )

    # ------------------------------------------------------------------
    # transport 鸭子接口
    # ------------------------------------------------------------------
    def _register(self, name, tensor, block_elems, chunk_blocks):
        # 全组同名同序同 chunk_blocks 校验：对称堆分配顺序与步长全组一致，
        # 才能保证各 PE 地址对称、展平块号含义一致
        meta = (name, tuple(tensor.shape), str(tensor.dtype), int(block_elems),
                int(chunk_blocks))
        _check_group_meta(self.group, self.world_size, meta)

        nbytes = chunk_blocks * block_elems * tensor.element_size()
        # aclshmemx_align(2MB) 分配对称堆内存（shmem_host_heap.h:93）；
        # pybind 当前仅暴露旧名 aclshmem_align（同一条堆分配路径），优先尝试新名
        align_fn = getattr(self._shmem, "aclshmemx_align", None) or self._shmem.aclshmem_align
        try:
            ptr = align_fn(self._ALIGN_BYTES, nbytes)
        except RuntimeError as exc:
            raise RuntimeError(
                f"aclshmemx_align 分配失败：name={name!r} nbytes={nbytes}，"
                f"请检查 heap_size={self.heap_size} 是否足够（{exc}）"
            ) from exc
        if not ptr:
            raise RuntimeError(
                f"aclshmemx_align 分配失败：name={name!r} nbytes={nbytes}，"
                f"请检查 heap_size={self.heap_size} 是否足够"
            )
        padded = self._tensor_from_ptr(
            ptr, (chunk_blocks * block_elems,), tensor.dtype)
        # padding 区内容确定（显式清零，对齐 GPU 零初始化语义），随后把
        # 初始内容拷入逻辑区；调用方此后应使用返回的逻辑视图
        padded.zero_()
        logical = padded[: tensor.numel()].view(tensor.shape)
        logical.copy_(tensor.to(padded.device))
        self._regs[name] = (padded, logical, block_elems, chunk_blocks, ptr, nbytes)
        return logical

    def _tensor(self, name):
        return self._regs[name][1]

    def _push_all(self, name, flat_indices, payload):
        padded, _logical, block, chunk_blocks, ptr, _nbytes = self._regs[name]
        view = padded.view(-1, block)
        stream = self._current_stream()
        block_bytes = block * padded.element_size()

        n = flat_indices.numel()
        if n > 0:
            # 展平块号拆分：pe = flat // chunk_blocks，off = flat % chunk_blocks
            flat_list = flat_indices.cpu().tolist()
            pes = [f // chunk_blocks for f in flat_list]
            offs = [f % chunk_blocks for f in flat_list]

            # 本 rank 目标走本地拷贝（GPU 语义：本地 VA 直写）
            local_sel = [i for i, p in enumerate(pes) if p == self.rank]
            if local_sel:
                off_t = torch.tensor([offs[i] for i in local_sel],
                                     dtype=torch.int64, device=padded.device)
                sel_t = torch.tensor(local_sel, dtype=torch.int64,
                                     device=payload.device)
                view[off_t] = payload[sel_t].to(dtype=padded.dtype)

            # 远端目标：逐块 putmem_on_stream（shmem_host_rma.h:634：
            # dst 传本地对称地址，库内翻译到目的 PE 的同偏移地址；
            # src 须为本机 NPU 上的连续张量；elem_size 形参按字节计）
            remote_sel = [i for i, p in enumerate(pes) if p != self.rank]
            if remote_sel:
                payload = payload.contiguous()
                src_base = payload.data_ptr()
                putmem = self._pyshmem.aclshmemx_putmem_on_stream
                for i in remote_sel:
                    dst_addr = ptr + offs[i] * block_bytes
                    src_addr = src_base + i * block_bytes
                    putmem(dst_addr, src_addr, block_bytes, pes[i], stream)

        # quiet + 流同步（shmem_host_rma.h:697）：保证 push_all 返回时数据已送达目的 PE
        self._pyshmem.aclshmemx_quiet_on_stream(stream)
        torch.npu.current_stream().synchronize()

    def _pull_all(self, name, flat_indices):
        padded, _logical, block, chunk_blocks, ptr, _nbytes = self._regs[name]
        view = padded.view(-1, block)
        stream = self._current_stream()
        block_bytes = block * padded.element_size()

        n = flat_indices.numel()
        # 落盘缓冲按请求顺序填充（应答序 == 请求序，不得按 pe 重排）
        landing = torch.empty(n, block, dtype=padded.dtype, device=padded.device)
        if n > 0:
            flat_list = flat_indices.cpu().tolist()
            pes = [f // chunk_blocks for f in flat_list]
            offs = [f % chunk_blocks for f in flat_list]

            # 本 rank 源走本地取数
            local_sel = [i for i, p in enumerate(pes) if p == self.rank]
            if local_sel:
                off_t = torch.tensor([offs[i] for i in local_sel],
                                     dtype=torch.int64, device=padded.device)
                sel_t = torch.tensor(local_sel, dtype=torch.int64,
                                     device=padded.device)
                landing[sel_t] = view[off_t]

            # 远端源：逐块 getmem_on_stream（shmem_host_rma.h:613）：src 传
            # 本地对称地址，库内翻译到源 PE 的同偏移地址读回
            getmem = self._pyshmem.aclshmemx_getmem_on_stream
            for i, p in enumerate(pes):
                if p == self.rank:
                    continue
                dst_addr = landing.data_ptr() + i * block_bytes
                src_addr = ptr + offs[i] * block_bytes
                getmem(dst_addr, src_addr, block_bytes, p, stream)

        # quiet + 流同步：保证 pull_all 返回时落盘缓冲可读
        self._pyshmem.aclshmemx_quiet_on_stream(stream)
        torch.npu.current_stream().synchronize()
        return landing

    def _pull_into(self, name, src_flats, dst_flats):
        """按展平块号把远端块直接 getmem 进本 rank 对称堆目标块（无落盘中转）。

        dst 为本 rank 堆内地址（getmem 的目的地址是任意本机 NPU 地址，
        对称堆行同样适用）；src 传本地对称地址，库内翻译到源 PE 的同偏移
        地址（shmem_host_rma.h:613，与 _pull_all 同一寻址约定）。
        pe == 本 rank 的源走堆内本地块拷贝（同块自拷贝为 no-op 语义安全）。
        """
        padded, _logical, block, chunk_blocks, ptr, _nbytes = self._regs[name]
        view = padded.view(-1, block)
        stream = self._current_stream()
        block_bytes = block * padded.element_size()

        n = src_flats.numel()
        if n > 0:
            getmem = self._pyshmem.aclshmemx_getmem_on_stream
            for s, d in zip(src_flats.cpu().tolist(),
                            dst_flats.cpu().tolist()):
                pe, off = divmod(s, chunk_blocks)
                if pe == self.rank:
                    view[d] = view[off]
                else:
                    getmem(ptr + d * block_bytes,   # dst：本 rank 堆内目标块
                           ptr + off * block_bytes,  # src：本地对称地址→源 PE
                           block_bytes, pe, stream)

        # quiet + 流同步：保证 pull_into 返回时目标块可读（n=0 也参与，
        # 保持集合调用的时序对齐）
        self._pyshmem.aclshmemx_quiet_on_stream(stream)
        torch.npu.current_stream().synchronize()

    def _barrier(self):
        stream = self._current_stream()
        barrier_fn = getattr(self._pyshmem, "aclshmemx_barrier_all_on_stream", None)
        if barrier_fn is not None:
            barrier_fn(stream)
        else:
            # pybind 未导出（shmem_host_cc.h:67）时走 ctypes（同一 libshmem.so）
            self._load_libshmem().aclshmemx_barrier_all_on_stream(ctypes.c_void_p(stream))
        # 屏障排在 stream 上，host 侧同步等待全组到齐
        torch.npu.current_stream().synchronize()

    def _allgather_ctrl(self, local, tag):
        # 控制面汇聚走 HCCL group
        return _all_gather_ctrl(self.group, self.world_size, local)

    # ------------------------------------------------------------------
    # 资源回收（Buffer.destroy 的可选挂点）
    # ------------------------------------------------------------------
    def finalize(self):
        """释放对称堆分配并 finalize aclshmem。调用后本 transport 不可再用。"""
        # 先屏障，确保无在途 RMA
        self._barrier()
        free_fn = getattr(self._shmem, "aclshmemx_free", None) or self._shmem.aclshmem_free
        for name in list(self._regs):
            _padded, _logical, _block, _cb, ptr, _nbytes = self._regs.pop(name)
            free_fn(ptr)
        self._shmem.aclshmem_finalize()
