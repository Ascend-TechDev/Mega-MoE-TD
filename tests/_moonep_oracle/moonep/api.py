# coding=utf-8
"""moonep 顶层 API（torch 语义级参考实现，昇腾路径）。

语义基准：GPU 版 MoonEP（source_code/MoonEP/moonep/api.py）。
绑定级设计契约见 docs/design.md；相对 GPU MoonEP 的偏差全集以契约 §0 为准，
除此之外不擅自偏离。

维度符号（与 GPU 源码一致）：S = 每 rank 输入 token 数，K = routed top-k，
N = S*K，E = EP 组 routed 专家总数，R = EP 组大小（num_ep_ranks），
epn = E/R，B = 每 rank 权重预取槽数（默认 epn），H = hidden size，
H' = 专家 FFN 中间维度，NvS = S*K + (token_padding-1)*2*epn
（每 rank 槽位上界：N 个真实槽 + VM-group 段 padding 余量，构造时冻结）。

与 GPU api.py 的 API 层差异（契约 §0/§5）：

- 各集合 API 只**编排 launch_* 宿主启动函数**（不在本模块直接实例化
  kernel 类）；launch_* 内部按 ctx 常量构造本包 torch kernel 并调用
  （契约 §4：kernel 类名/__init__ 常量配置/__call__ 入出参与源码逐字一致）；
- async_finish 仅保留参数位：所有集合操作同步执行，事件位恒为 None
  （参考实现无 CUDA 事件机制）；
- enable_pdl / num_sms / comm_stream_priority 仅留参：参考实现不消费，
  仅作为 launch_* 的 pdl_trigger/pdl_launch/num_sms 实参透传（kernel
  侧同样留参不消费）；
- 通信缓冲区为 arena 自建注册的本 rank 对称分片：hidden_buf [NvS, H]
  bf16（块 = H 元素）与 meta [meta_chunk_padded] int32（块 = 1 元素）；
  route weight 复用 meta 的 WEIGHTS 区（fp32 位壳视图），不单独建张量；
- GPU 的 VMM/多播粒度对齐、CUDA stream/event、record_stream 等机制无参考
  实现对应物；meta chunk 仅保留 16B（4 个 int32 元素）对齐。

用法（单进程多 rank 数值验证路径，模拟器为测试专用件
tests/sim_transport.py，不在生产包面内）：

    from sim_transport import SimTransport
    sim = SimTransport(R)
    buffers = [Buffer(S, H, K, E, R, token_padding=tp, arena=sim.arena_for(r))
               for r in range(R)]
    # dispatch fwd
    hidden_nvsh, route_w_nvs, cu_seqlens, plan = buffers[r].dispatch(
        hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert)
    buffers[r].prefetch_weight(plan=plan, full_gate_weight=w_gate,
                               full_up_weight=w_up, full_down_weight=w_down)
    # combine fwd
    out_sh, gathered_w_sk, _ = buffers[r].combine(
        plan=plan, hidden_nvsh=expert_out_nvsh, route_weights_nvs=route_w_nvs)
    # combine bwd：复用 plan 再散布梯度
    grad_nvsh, _, _, _ = buffers[r].dispatch(grad_out_sh, plan=plan)
    # dispatch bwd：梯度按 K 份求和回 token-major，并把副本专家权重梯度归并回属主
    grad_hidden_sh, _, _ = buffers[r].combine(plan=plan, hidden_nvsh=grad_nvsh)
    buffers[r].reduce_grad(plan=plan, full_gate_grad=g_gate, full_up_grad=g_up,
                           full_down_grad=g_down, gate_reduce_buffer=rb_gate,
                           up_reduce_buffer=rb_up, down_reduce_buffer=rb_down)
    buffers[r].destroy()
"""

import os

import torch
import torch.distributed as dist

from .buffer import (
    ShmemStreamTransport,
    SymmetricArena,
    register_arena,
    unregister_arena,
)
from .combine import launch_combine
from .combine_prologue import launch_combine_prologue
from .dispatch import launch_dispatch
from .dispatch_epilogue import launch_dispatch_epilogue
from .grad_reduce import launch_grad_reduce
# 契约目录结构里的 inter_rank_sync.py（跨 rank 同步原语 + launch 形态）
from .inter_rank_sync import launch_inter_rank_sync
from .planning import (
    MoonEPCommPlan,
    allocate_planning_outputs,
    launch_planning,
    physical_tokens_per_expert,
)
from .prefetch import launch_prefetch

__all__ = ["Buffer"]

# GPU 版 planning 的 BLOCK_SIZE_P2（planning.py），仅用于 num_vblocks 常量推导
_BLOCK_SIZE_P2 = 2048

# meta 的 BARRIER 区槽数（源码 api.py：cross_rank_barrier 的 2 相位信号 +
# 1 相位/符号计数，自复位双缓冲）；参考实现的屏障走 transport，槽位仅作布局兼容
_BARRIER_SLOTS = 3


def _align_up(x: int, alignment: int) -> int:
    """把 x 向上对齐到 alignment 的倍数。"""
    return ((x + alignment - 1) // alignment) * alignment


def _num_sms_dedup_from_env(max_sms: int) -> int:
    """解析本地 epilogue/prologue 的 SM 数（语义对齐 GPU api.py 同名函数）。

    ``MOONEP_NUM_SMS_DEDUP`` 是有意的环境变量覆盖位（而非 Buffer 形参），
    便于 benchmark 任务不改公开 API 即可扫描取值。参考实现的 torch kernel 体
    不消费该数值，仅作为 launch_* 的 num_sms 实参透传（保持构造契约）。
    参考实现无 CUDA device 可查硬件 SM 数，上界取 Buffer 的 num_sms。
    """
    raw = os.environ.get("MOONEP_NUM_SMS_DEDUP")
    if raw is None or raw == "":
        return max_sms
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"MOONEP_NUM_SMS_DEDUP must be an integer in [1, {max_sms}], got {raw!r}"
        ) from exc
    if not (1 <= value <= max_sms):
        raise ValueError(
            f"MOONEP_NUM_SMS_DEDUP must be in [1, {max_sms}], got {value}"
        )
    return value


def _probe_int_attr(obj, names):
    """在 obj 上按名字列表探测一个非负 int（属性或零参方法均可），失败返回 None。"""
    if obj is None:
        return None
    for attr in names:
        if not hasattr(obj, attr):
            continue
        value = getattr(obj, attr)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return int(value)
    return None


def _probe_rank(obj):
    """尽力从 arena/transport（或其持有的 transport）推断本 rank；找不到返回 None。"""
    seen = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        rank = _probe_int_attr(obj, ("rank",))
        if rank is not None:
            return rank
        # 兜底：对象若持有 process group，则经 torch.distributed 求 rank
        group = getattr(obj, "group", None)
        if group is not None and dist.is_available() and dist.is_initialized():
            try:
                return int(dist.get_rank(group=group))
            except Exception:
                return None
        obj = getattr(obj, "transport", None) or getattr(obj, "_transport", None)
    return None


def _probe_world(obj):
    """尽力从 arena/transport（或其持有的 transport）推断组大小；找不到返回 None。"""
    seen = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        # 契约 §3 的属性名为 world；兼容 world_size / num_ranks 形态
        world = _probe_int_attr(obj, ("world", "world_size", "num_ranks"))
        if world is not None and world > 0:
            return world
        obj = getattr(obj, "transport", None) or getattr(obj, "_transport", None)
    return None


def _launch_full_weight_prefetches(
    ctx,
    full_gate_weight: torch.Tensor,
    full_up_weight: torch.Tensor,
    full_down_weight: torch.Tensor,
    experts_to_copy: torch.Tensor,
) -> None:
    """逐投影调 launch_prefetch（属主局部寻址版）。

    full_w [epn+B, H, H'] bf16（epn = E//R）：行 [0, epn) 为本 rank 属主专家
    权重（remote_expert，prefetch 远端读源），行 [epn, epn+B) 为本调用填充
    的预取槽（prefetch_buffers）。每 rank 物理仅 epn+B 行——对比全局表
    [E+B] 布局省 (R-1)*epn 行镜像死空间。full_w 整体即已注册对称张量，
    launch_prefetch 内部经 resolve_arena_for 反查 arena。
    experts_to_copy 为本 rank 的槽位专家表 [B]（plan.experts_to_copy[rank]）。
    """
    E = int(ctx["E"])
    R = int(ctx["R"])
    epn = E // R
    num_sms = int(ctx["num_sms"])
    for full_weight in (full_gate_weight, full_up_weight, full_down_weight):
        launch_prefetch(
            full_weight,
            full_weight[epn:],
            experts_to_copy,
            num_sms=num_sms,
        )


def _launch_full_grad_reduces(
    ctx,
    experts_to_copy: torch.Tensor,
    full_gate_grad: torch.Tensor,
    full_up_grad: torch.Tensor,
    full_down_grad: torch.Tensor,
    gate_reduce_buffer: torch.Tensor,
    up_reduce_buffer: torch.Tensor,
    down_reduce_buffer: torch.Tensor,
) -> None:
    """逐投影调 launch_grad_reduce（属主局部寻址版）。

    full_grad [epn, H, H'] fp32（epn = E//R）：本 rank 属主专家的梯度行
    （局部行号索引），reduce_grad 把全组副本槽梯度累加进来；reduce buffer
    [B, H, H'] fp32 须已在本 rank arena 注册（launch_grad_reduce 内部经
    resolve_arena_for 反查 arena）。experts_to_copy 为全组 [R, B] 表。
    """
    E = int(ctx["E"])
    R = int(ctx["R"])
    B = int(ctx["B"])
    rank = int(ctx["rank"])
    num_sms = int(ctx["num_sms"])
    for name, full_grad, reduce_buffer in (
        ("gate", full_gate_grad, gate_reduce_buffer),
        ("up", full_up_grad, up_reduce_buffer),
        ("down", full_down_grad, down_reduce_buffer),
    ):
        assert full_grad is not None, f"full_{name}_grad is required"
        assert full_grad.dtype == torch.float32 and full_grad.is_contiguous(), \
            f"full_{name}_grad must be contiguous fp32 [epn, H, H']"
        assert full_grad.ndim == 3 and int(full_grad.shape[0]) == E // R, \
            f"full_{name}_grad first dim must be epn=E//R"
        launch_grad_reduce(
            full_grad,
            reduce_buffer,
            experts_to_copy,
            rank=rank,
            num_sms=num_sms,
            meta_buf=ctx["meta_buf"],
            meta_stride=int(ctx["meta_chunk_padded"]),
            barrier_off=int(ctx["BARRIER_OFF"]),
            grid_sync_bar=ctx["grid_sync_bar"],
        )


class Buffer:
    """MoonEP 通信缓冲区持有者（torch 语义级参考实现，契约 §5）。

    持有一个通信域（EP group）的 SymmetricArena、自建注册的对称张量
    hidden_buf [NvS, H] bf16（块 = H 元素）与 meta [meta_chunk_padded]
    int32（块 = 1 元素，WEIGHTS 区物化在其中），并提供 dispatch /
    prefetch_weight / combine / reduce_grad 四个集合操作。四个操作均为
    集合调用（全组须同参进入同一 API 阶段）；参考实现同步执行
    （async_finish 事件位恒为 None）。进程组拆除前调用 destroy()。
    """

    # 自建对称张量的注册名（全组同名同序 register）
    HIDDEN_BUF_NAME = "hidden_buf"
    META_BUF_NAME = "meta_buf"

    def __init__(
        self,
        S: int,
        H: int,
        K: int,
        E: int,
        num_ep_ranks: int,
        num_sms: "int | None" = None,
        token_padding: int = 128,
        B: "int | None" = None,
        group: "dist.ProcessGroup | None" = None,
        comm_stream_priority: int = -1,
        enable_pdl: bool = True,
        transport=None,
        arena=None,
    ):
        """分配并持有全部通信缓冲区，构建 ctx 并把 arena 登记进全局注册表。

        NvS = S*K + (token_padding-1)*2*E//R（公式逐字对齐 GPU api.py:268-279）；
        B 默认 E//R；meta 区域偏移按 GPU api.py:298-331 逐字计算。
        arena 优先；否则 transport 包装为 SymmetricArena；transport=None 时
        用 ShmemStreamTransport（需已安装 shmem 并初始化进程组）。

        Args:
            S: 每 rank 输入 token 数。
            H: hidden size。
            K: 每 token 的 routed top-k。
            E: EP 组 routed 专家总数，必须能被 num_ep_ranks 整除。
            num_ep_ranks: EP 组大小 R；传入 group 时必须等于 group 的 world size。
            num_sms: 仅留参（GPU 版为通信 kernel 的 SM 数；None 默认 32；
                参考实现不消费，仅作 launch_* 的 num_sms 实参透传）。
            token_padding: 每个非空 VM-group 段向上对齐到的 token 数倍数。
            B: 每 rank 权重预取槽数；None 默认 E // num_ep_ranks。
            group: torch.distributed 进程组；用于 ShmemStreamTransport 及 rank
                推断。
            comm_stream_priority: 仅留参（GPU 版为异步通信流优先级；
                参考实现无 stream 概念）。
            enable_pdl: 仅留参（GPU 版的 PDL 启动开关；参考实现无语义作用，
                仅作 launch_* 的 pdl_trigger/pdl_launch 实参透传）。
            transport: 传输层实现；与 arena 二选一（arena 优先）。
            arena: SymmetricArena（如测试模拟器 arena_for(r) 的产物）。

        注：GPU 版的 ``explicitly_destroy`` 形参在参考实现中去除（契约 §5
        签名以此为准）——参考实现不持有显式硬件资源，destroy() 的语义为
        幂等清理引用 + 全局 arena 注册表反登记。
        """
        # 尽早落位可销毁态，保证半构造对象上调 destroy() 仍安全
        self._destroyed = False
        self._ctx = None
        self.arena = None
        self.transport = None
        self.group = None

        # ---- 参数校验（对齐 GPU api.py:477-486 + 参考实现扩展）----
        assert isinstance(comm_stream_priority, int), (
            f"comm_stream_priority must be an int, got "
            f"{type(comm_stream_priority).__name__}"
        )
        assert isinstance(enable_pdl, bool), (
            f"enable_pdl must be a bool, got {type(enable_pdl).__name__}"
        )
        R = int(num_ep_ranks)
        assert R > 0, f"num_ep_ranks must be a positive int, got {num_ep_ranks}"
        assert E % R == 0, f"E ({E}) must be divisible by R ({R})"
        assert isinstance(token_padding, int) and token_padding > 0, \
            f"token_padding must be a positive int, got {token_padding}"
        epn = E // R
        if B is None:
            B = epn
        assert isinstance(B, int) and B > 0, f"B must be a positive int, got {B}"
        for name, value in (("S", S), ("H", H), ("K", K)):
            assert isinstance(value, int) and value > 0, \
                f"{name} must be a positive int, got {value}"
        N = S * K
        int32_max = 2**31 - 1
        assert 0 < N < int32_max, (
            "planning requires 0 < S*K < int32_max: "
            f"S={S}, K={K}, S*K={N}, int32_max={int32_max}"
        )

        # ================================================================
        # 槽位上界（公式逐字对齐 GPU api.py:268-279）
        # 每个目的 rank 至多 epn 个本地专家段 + epn 个远程专家段（每个目的
        # rank 的远程 token 只来自单一 home 组），每个非空段最多浪费
        # token_padding-1 个槽；NvS_capacity 即单源填充的 CAP（S*K）。
        # ================================================================
        NvS_capacity = N
        token_padding_extra = (token_padding - 1) * 2 * epn
        NvS = NvS_capacity + token_padding_extra
        # src_info 线性编码 src_rank*NvS+offv 必须落在 int32（契约 §2 扩展字段）
        assert R * NvS <= int32_max, (
            "src_info linear encoding requires R*NvS <= int32_max: "
            f"R={R}, NvS={NvS}, R*NvS={R * NvS}, int32_max={int32_max}"
        )
        # 参考实现无 VMM 粒度对齐：hidden_buf 即 [NvS, H]，NvS_padded 退化为 NvS
        NvS_padded = NvS
        num_vblocks = (N + _BLOCK_SIZE_P2 - 1) // _BLOCK_SIZE_P2

        if num_sms is None:
            num_sms = 32
        assert isinstance(num_sms, int) and num_sms > 0, \
            f"num_sms must be a positive int, got {num_sms}"
        num_sms_dedup = _num_sms_dedup_from_env(num_sms)

        # ================================================================
        # meta 区域偏移（元素单位 int32，逐字对齐 GPU api.py:298-331）
        # 布局：WEIGHTS[NvS) | TPE[R*E) | PLAN 广播/输出暂存 | TOPK0[N4) |
        #       ORDER[N4) | ORDER0[N4) | BARRIER[3) | SRC_INFO[NvS)
        # ================================================================
        WEIGHTS_OFF = 0
        TPE_OFF = _align_up(NvS, 4)
        PLAN_OFF = _align_up(TPE_OFF + R * E, 4)
        broadcast_elems = 3 * E * R
        assert broadcast_elems % 4 == 0, (
            f"broadcast_elems ({broadcast_elems}) must be divisible by 4"
        )
        planning_out_elems = (
            broadcast_elems
            + R * (E + B)
            + 2 * R * (E + B)
            + B * R
            + 2 * R
        )
        # C1 scratch 段按 4 元素 padding，保证向量化拷贝 16B 对齐
        N4 = _align_up(N, 4)
        TOPK0_OFF = _align_up(PLAN_OFF + planning_out_elems, 4)
        ORDER_OFF = TOPK0_OFF + N4
        ORDER0_OFF = ORDER_OFF + N4
        BARRIER_OFF = ORDER0_OFF + N4
        # 跨 rank 屏障每 rank 3 槽（2 相位信号 + 1 相位/符号计数，自复位双
        # 缓冲）；参考实现的屏障走 transport，槽位仅作布局兼容保留
        SRC_INFO_OFF = BARRIER_OFF + _BARRIER_SLOTS
        meta_chunk_logical = SRC_INFO_OFF + NvS
        # GPU 版 chunk 还须满足 VMM 映射与多播绑定粒度；参考实现每 rank 一个
        # 独立 int32 张量，仅保留 16B（4 元素）对齐
        meta_chunk_padded = _align_up(meta_chunk_logical, 4)

        # ================================================================
        # rank / arena 解析（契约 §5：arena 优先；否则 transport 包装；
        # transport=None 时用 ShmemStreamTransport）
        # ================================================================
        if group is not None:
            assert dist.is_available() and dist.is_initialized(), (
                "Buffer: 传入 group 前必须先调用 init_process_group"
            )
            assert R == dist.get_world_size(group=group), (
                f"num_ep_ranks ({R}) must equal group world size "
                f"({dist.get_world_size(group=group)})"
            )
        rank = None
        if arena is None:
            if transport is None:
                transport = ShmemStreamTransport(group=group)
            arena = SymmetricArena(transport)
        if rank is None:
            rank = _probe_rank(arena)
        if rank is None and group is not None:
            rank = int(dist.get_rank(group=group))
        if rank is None:
            rank = _probe_rank(transport)
        if rank is None:
            if R == 1:
                rank = 0
            else:
                raise RuntimeError(
                    "Buffer: 无法推断本 rank（R>1 且未传 group）；请传入 group，"
                    "或传入带 rank 信息的 arena"
                )
        world = _probe_world(arena)
        if world is not None:
            assert world == R, (
                f"arena 的 world size ({world}) 与 num_ep_ranks ({R}) 不一致"
            )

        # ---- 实例属性（常量配置与 rank 身份）----
        self.S = int(S)
        self.H = int(H)
        self.K = int(K)
        self.E = int(E)
        self.R = R
        self.epn = epn
        self.B = int(B)
        self.N = N
        self.NvS = NvS
        self.NvS_capacity = NvS_capacity
        self.NvS_padded = NvS_padded
        self.num_vblocks = num_vblocks
        self.token_padding = int(token_padding)
        self.rank = rank
        self.group = group
        self.transport = transport
        self.arena = arena
        # 仅留参：参考实现不消费，仅作 launch_* 实参透传
        self.num_sms = num_sms
        self.num_sms_dedup = num_sms_dedup
        self.comm_stream_priority = comm_stream_priority
        self.enable_pdl = enable_pdl

        # meta 布局常量（实例属性，供检视与测试）
        self.WEIGHTS_OFF = WEIGHTS_OFF
        self.TPE_OFF = TPE_OFF
        self.PLAN_OFF = PLAN_OFF
        self.TOPK0_OFF = TOPK0_OFF
        self.ORDER_OFF = ORDER_OFF
        self.ORDER0_OFF = ORDER0_OFF
        self.BARRIER_OFF = BARRIER_OFF
        self.SRC_INFO_OFF = SRC_INFO_OFF
        self.meta_chunk_logical = meta_chunk_logical
        self.meta_chunk_padded = meta_chunk_padded

        # ================================================================
        # 自建对称张量注册（集合调用：全组同名同序同 chunk_blocks register）
        # dtype 契约：payload bf16 / meta int32；零初始化保证 padding 行与
        # 空权重槽内容确定。chunk_blocks 为展平块号的 chunk 步长（对应 GPU
        # 的 NvS_padded / meta_chunk_padded；NvS_padded 当前退化为 NvS，
        # 契约 §0）。注意应使用 register 的返回值（ShmemStreamTransport
        # 下为对称堆上的新物理张量），而非入参。
        # ================================================================
        self._hidden_buf_local = arena.register(
            self.HIDDEN_BUF_NAME,
            torch.zeros(NvS, H, dtype=torch.bfloat16),
            H,
            chunk_blocks=NvS_padded,
        )
        self._meta_local = arena.register(
            self.META_BUF_NAME,
            torch.zeros(meta_chunk_padded, dtype=torch.int32),
            1,
            chunk_blocks=meta_chunk_padded,
        )
        # weights_buf 复用 meta 的 WEIGHTS 区（契约 §5：不再单独建张量）；
        # WEIGHTS_OFF=0，fp32 位壳视图与 meta chunk 共享存储（data_ptr 相同）
        self._weights_buf_local = (
            self._meta_local[WEIGHTS_OFF:WEIGHTS_OFF + NvS].view(torch.float32)
        )

        # ================================================================
        # ctx（契约 §2 键集；launch_* 层经 ctx 取常量、对称分片与 arena）
        # 键名与 GPU 源码 _create_context 一致；GPU 专属的 grid_sync_bar 等
        # 键以 None 占位（契约 §2 允许），GPU scratch 张量键不建（launch_*
        # 内部以 plan.dst 等占位实参满足 kernel 签名）。
        # ================================================================
        self._ctx = {
            "rank": rank,
            "group": group,
            "R": R, "E": E, "S": S, "K": K, "H": H,
            "B": self.B,
            "N": N, "NvS": NvS,
            "NvS_capacity": NvS_capacity,
            "NvS_padded": NvS_padded,
            "num_sms": num_sms,
            "num_sms_dedup": num_sms_dedup,
            "token_padding": self.token_padding,
            "num_vblocks": num_vblocks,
            # meta 布局
            "meta_chunk_logical": meta_chunk_logical,
            "meta_chunk_padded": meta_chunk_padded,
            "WEIGHTS_OFF": WEIGHTS_OFF,
            "TPE_OFF": TPE_OFF,
            "PLAN_OFF": PLAN_OFF,
            "TOPK0_OFF": TOPK0_OFF,
            "ORDER_OFF": ORDER_OFF,
            "ORDER0_OFF": ORDER0_OFF,
            "BARRIER_OFF": BARRIER_OFF,
            "SRC_INFO_OFF": SRC_INFO_OFF,
            # 本 rank 对称分片（arena.register 返回的物理张量）
            "hidden_buf_local": self._hidden_buf_local,
            "meta_buf": self._meta_local,
            "weights_buf_local": self._weights_buf_local,
            # GPU 专属键位（契约 §2：可以 None 占位）
            "grid_sync_bar": None,
            "comm_stream": None,
            # —— 参考实现扩展键（契约 §2）——
            "arena": arena,
            "device": self._hidden_buf_local.device,
        }

        # ---- 全局 arena 注册表登记（契约 §5；供 launch_prefetch /
        # launch_grad_reduce 等无 ctx 的 launch_* 经 resolve_arena_for
        # 按 data_ptr 反查本 arena）----
        register_arena(arena)

    # ------------------------------------------------------------------
    # 基础访问器
    # ------------------------------------------------------------------
    @property
    def destroyed(self) -> bool:
        return self._destroyed

    @property
    def hidden_buf_local(self) -> torch.Tensor:
        """本 rank hidden 对称分片 [NvS, H] bf16（register 返回的物理张量）。"""
        return self._hidden_buf_local

    @property
    def weights_buf_local(self) -> torch.Tensor:
        """本 rank weights 对称分片 [NvS] fp32（meta WEIGHTS 区的位壳视图）。"""
        return self._weights_buf_local

    @property
    def meta_buf_local(self) -> torch.Tensor:
        """本 rank meta 对称分片 [meta_chunk_padded] int32（register 返回的物理张量）。"""
        return self._meta_local

    def _require_ctx(self) -> dict:
        assert not self._destroyed, "MoonEP Buffer has been destroyed"
        assert self._ctx is not None, "MoonEP Buffer is not initialized"
        return self._ctx

    def destroy(self) -> None:
        """清理本 Buffer 持有的引用并反登记全局 arena 注册表，幂等。

        参考实现无显式硬件资源可释放：先做全组屏障（对齐 GPU destroy 的
        dist.barrier 语义：全组到齐后才允许反登记，模拟器下为
        no-op），再按"先视图后属主"的顺序释放引用（镜像 GPU destroy），
        最后从全局注册表反登记 arena。进程组拆除前调用。
        """
        if self._destroyed:
            return
        arena = self.arena
        if arena is not None:
            # 全组到齐屏障（GPU：dist.barrier(group)；参考实现走 transport）
            arena.barrier()
            # 全局注册表反登记（契约 §5：destroy 含全局注册表反登记）
            unregister_arena(arena)
        # 先视图后属主（镜像 GPU destroy 的顺序语义）
        self._weights_buf_local = None
        self._hidden_buf_local = None
        self._meta_local = None
        if self._ctx is not None:
            self._ctx.clear()
        self._ctx = None
        self.arena = None
        self.transport = None
        self.group = None
        self._destroyed = True

    # ------------------------------------------------------------------
    # 对称张量解析（允许直接传注册名字符串）
    # ------------------------------------------------------------------
    def _resolve_tensor(self, value, what: str) -> torch.Tensor:
        """把框架传入的对称张量解析为 arena 注册的本 rank 物理张量。

        无 ctx 的 launch_*（launch_prefetch/launch_grad_reduce）经全局注册表
        按 data_ptr 反查 arena，因此传入张量必须是 register() 返回的本 rank
        物理张量（或其同 data_ptr 的前缀切片）；也允许直接传注册名字符串
        （等价 arena.tensor(name)）。
        """
        if isinstance(value, str):
            return self.arena.tensor(value)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{what}: 期望 register() 返回的本 rank 物理张量（torch.Tensor）"
                f"或注册名字符串，得到 {type(value).__name__}"
            )
        return value

    # ------------------------------------------------------------------
    # 物理组计数辅助
    # ------------------------------------------------------------------
    def local_tokens_per_expert(self, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """本 rank 各物理组的 padded token 计数 [epn+B]。

        物理组序 = epn 个本地专家段（本地专家序号升序）+ B 个副本槽段（槽序）。
        语义即 planning.physical_tokens_per_expert：diff(cu_seqlens) 的
        [本地专家段 | 槽段] 切片（physical_tokens_per_expert 的封装，契约 §5）。
        """
        self._require_ctx()
        return physical_tokens_per_expert(cu_seqlens, self.rank, self.epn, self.E)

    # ------------------------------------------------------------------
    # dispatch / combine 的执行体（步骤顺序逐字对齐 GPU
    # _run_dispatch_on_current_stream api.py:588-630 /
    # _run_combine_on_current_stream api.py:632-663；参考实现无 stream
    # 概念，同步执行即"当前流"语义）
    # ------------------------------------------------------------------
    def _run_dispatch_on_current_stream(
        self,
        ctx: dict,
        hidden_sh: torch.Tensor,
        route_weights_sk: "torch.Tensor | None",
        planning_args,
        plan: MoonEPCommPlan,
        hidden_nvsh: torch.Tensor,
        route_weights_nvs: "torch.Tensor | None",
        *,
        inter_rank_sync: bool,
        zero_copy: bool,
    ) -> None:
        # 1) inter_rank_sync：规划前的跨 rank 对齐点（可选）
        if inter_rank_sync:
            launch_inter_rank_sync(ctx)

        # 2) planning：新鲜规划（allgather topk/tpe + 全组冗余重算 + plan
        #    扩展字段归位由 launch_planning 内部完成）；plan 复用路径跳过。
        if planning_args is not None:
            topk_flat, tokens_per_expert, cu_seqlens = planning_args
            launch_planning(ctx, topk_flat, tokens_per_expert, cu_seqlens, plan)

        # 3) dispatch：payload/权重按 dst 散布（负 dst 只写权重），并按
        #    plan.zero_fill_ranges 清零本 shard padding 行。新鲜规划紧前
        #    发布了 dst/src_info，可安全物化 plan 持有的 dedup 结构
        #    （build_dedup_map=True）；复用路径保持已保存结构不动
        #    （build_dedup_map=False），zero 路径两条路径都执行。
        launch_dispatch(
            ctx,
            hidden_sh,
            route_weights_sk,
            plan,
            build_dedup_map=planning_args is not None,
            pdl_trigger=self.enable_pdl,
        )
        # 4) epilogue：本 shard 上 dup 行原地扇出（纯本地）；此后 shard 即
        #    完整的用户可见 [NvS, H] 布局。
        launch_dispatch_epilogue(ctx, plan, pdl_launch=self.enable_pdl)

        # 5) 边界拷贝（zero_copy=False 时拷出新张量；True 时输出位即缓冲
        #    区视图，无需拷贝）。ctx['weights_buf_local'] 已是 fp32 位壳
        #    视图（GPU 侧为 int32 视图、此处再 .view(torch.float32)，等价）。
        if not zero_copy:
            hidden_nvsh.copy_(ctx["hidden_buf_local"])
            if route_weights_nvs is not None:
                route_weights_nvs.copy_(ctx["weights_buf_local"])

    def _run_combine_on_current_stream(
        self,
        ctx: dict,
        hidden_sh: torch.Tensor,
        plan: MoonEPCommPlan,
        hidden_nvsh: torch.Tensor,
        route_weights_nvs: "torch.Tensor | None",
        route_weights_sk: "torch.Tensor | None",
        *,
        inter_rank_sync: bool,
        zero_copy: bool,
    ) -> None:
        # 1) inter_rank_sync：暂存前的对齐点（可选，保持 GPU master 时代
        #    的位置）；CombineKernel 入口的跨 rank 屏障负责发布本 shard 的
        #    暂存 + 归并写入。
        if inter_rank_sync:
            launch_inter_rank_sync(ctx)

        # 2) zero_copy=False 时先把外部输入暂存进本 rank 对称分片
        #    （zero_copy=True 已由 data_ptr 断言保证入参即分片视图）。
        if not zero_copy:
            ctx["hidden_buf_local"].copy_(hidden_nvsh)
            if route_weights_nvs is not None:
                ctx["weights_buf_local"].copy_(route_weights_nvs)

        # 3) prologue：dup 组 fp32 无权求和写回 primary 行（纯本地）
        launch_combine_prologue(ctx, plan, pdl_trigger=self.enable_pdl)

        # 4) combine：逐 token 的 K 个非负 dst 经 arena.pull_all 拉属主行、
        #    fp32 累加写 output；权重 gather 从本 rank meta WEIGHTS 区读回
        #    （负 dst raw 解码后照常 gather）。
        launch_combine(
            ctx,
            hidden_sh,
            plan.dst,
            output_sk=route_weights_sk,
            pdl_launch=self.enable_pdl,
        )

    # ------------------------------------------------------------------
    # dispatch（dispatch fwd / combine bwd 的再散布路径）
    # ------------------------------------------------------------------
    def dispatch(
        self,
        hidden_sh: torch.Tensor,
        route_weights_sk: "torch.Tensor | None" = None,
        topk_experts_sk: "torch.Tensor | None" = None,
        tokens_per_expert: "torch.Tensor | None" = None,
        plan: "MoonEPCommPlan | None" = None,
        async_finish: bool = False,
        *,
        inter_rank_sync: bool = True,
        zero_copy: bool = False,
    ):
        """dispatch fwd：做规划（除非复用 plan）并把 token 按专家分组散布到各 rank。

        Args:
            hidden_sh: [S, H] bf16 输入 token。
            route_weights_sk: [S, K] fp32 路由权重；None 时完全跳过 weights 缓冲。
            topk_experts_sk: [S, K] int32 专家 id；plan 为 None 时必填。
            tokens_per_expert: [E] int32 本 rank 每专家 token 数；plan 为 None 时必填。
            plan: 复用的 MoonEPCommPlan —— 跳过规划（allgather + PlanningKernel
                冗余重算），topk_experts_sk / tokens_per_expert 被忽略。这是
                combine bwd 路径：用 fwd 的 plan 再散布 grad_output_sh，即把
                输出梯度散布回 VM 组序。
            async_finish: 仅保留参数位；参考实现同步执行（无 CUDA 事件机制），
                True 时返回五元组、事件位恒为 None。
            inter_rank_sync: 规划前做一次跨 rank 对齐（默认 True）。
            zero_copy: True 时返回通信缓冲区视图（hidden_buf_local 与 meta
                WEIGHTS 区的 fp32 位壳视图），不做边界拷贝；视图会被本 Buffer
                的下一次 dispatch/combine 覆写，调用方不得跨通信调用持有（尤其
                autograd 不得 save 它们做 backward——那正是需要 zero_copy=False
                的场景）。行内容仅在 cu_seqlens 覆盖的 padded 段内有定义。

        Returns:
            (hidden_nvsh, route_weights_nvs, cu_seqlens, plan)，
            async_finish=True 时追加事件位（恒为 None）：

            - hidden_nvsh: [NvS, H] bf16，物理 VM 组序的散布结果。
            - route_weights_nvs: [NvS] fp32；route_weights_sk 为 None 时为 None。
            - cu_seqlens: [E+B] int32，各 VM 组行的 padded 结束偏移；
              plan 复用路径为 None。
            - plan: MoonEPCommPlan；保存供 prefetch/combine 及两次 backward 使用。
        """
        ctx = self._require_ctx()

        # ---- 输入校验（dtype 契约：payload bf16 / 路由权重 fp32 / meta int32）----
        assert hidden_sh is not None
        assert hidden_sh.dtype == torch.bfloat16 and hidden_sh.is_contiguous(), \
            "dispatch: hidden_sh must be contiguous bf16 [S, H]"
        assert tuple(hidden_sh.shape) == (self.S, self.H), \
            f"dispatch: hidden_sh shape must be ({self.S}, {self.H}), " \
            f"got {tuple(hidden_sh.shape)}"
        if route_weights_sk is not None:
            assert route_weights_sk.dtype == torch.float32 and route_weights_sk.is_contiguous(), \
                "dispatch: route_weights_sk must be contiguous fp32 [S, K]"
            assert tuple(route_weights_sk.shape) == (self.S, self.K), \
                f"dispatch: route_weights_sk shape must be ({self.S}, {self.K})"

        if plan is None:
            assert topk_experts_sk is not None and tokens_per_expert is not None
            topk_flat = topk_experts_sk.reshape(-1)
            assert topk_flat.dtype == torch.int32 and topk_flat.numel() == int(ctx["N"])
            assert tokens_per_expert.dtype == torch.int32
            assert tokens_per_expert.numel() == int(ctx["E"]) and tokens_per_expert.is_contiguous()
            plan, cu_seqlens = allocate_planning_outputs(ctx)
            planning_args = (topk_flat, tokens_per_expert, cu_seqlens)
        else:
            cu_seqlens = None
            planning_args = None
            assert isinstance(plan, MoonEPCommPlan)

        # ---- 输出位准备：zero_copy 返回缓冲区视图，否则分配新张量 ----
        if zero_copy:
            hidden_nvsh = ctx["hidden_buf_local"]
            route_weights_nvs = (
                ctx["weights_buf_local"] if route_weights_sk is not None else None
            )
        else:
            hidden_nvsh = torch.empty_like(ctx["hidden_buf_local"])
            route_weights_nvs = (
                torch.empty(
                    int(ctx["NvS"]),
                    dtype=torch.float32,
                    device=ctx["meta_buf"].device,
                )
                if route_weights_sk is not None else None
            )

        # ---- 步骤顺序对齐 GPU _run_dispatch_on_current_stream ----
        self._run_dispatch_on_current_stream(
            ctx,
            hidden_sh,
            route_weights_sk,
            planning_args,
            plan,
            hidden_nvsh,
            route_weights_nvs,
            inter_rank_sync=inter_rank_sync,
            zero_copy=zero_copy,
        )

        if async_finish:
            # 参考实现同步执行、无事件机制；事件位恒为 None
            return hidden_nvsh, route_weights_nvs, cu_seqlens, plan, None
        return hidden_nvsh, route_weights_nvs, cu_seqlens, plan

    # ------------------------------------------------------------------
    # prefetch_weight（dispatch fwd 的权重侧）
    # ------------------------------------------------------------------
    def prefetch_weight(
        self,
        plan: "MoonEPCommPlan | None" = None,
        async_finish: bool = False,
        *,
        full_gate_weight: "torch.Tensor | None" = None,
        full_up_weight: "torch.Tensor | None" = None,
        full_down_weight: "torch.Tensor | None" = None,
    ):
        """按 plan 选中的远程专家，把其权重预取进本地预取槽（[epn, epn+B) 行）。

        Args:
            plan: dispatch 返回的 MoonEPCommPlan。
            async_finish: 仅保留参数位；参考实现同步执行，恒返回 None
                （GPU 版 async_finish=True 时返回通信流事件，参考实现无事件
                机制，事件位为 None）。
            full_gate_weight / full_up_weight / full_down_weight:
                [epn+B, H, H'] bf16 连续权重张量（epn = E//R，本 rank 局部
                布局），[0, epn) 行为本 rank 属主专家权重（prefetch 远端读源）、
                [epn, epn+B) 行为本调用填充的预取槽。三者必须已作为对称张量在
                arena 注册（传入 register() 返回的本 rank 物理张量；也允许
                直接传注册名字符串）——逐投影以 full_w 整表作 remote_expert、
                full_w[epn:] 作 prefetch_buffers 调 launch_prefetch（契约 §5；
                每 rank 物理仅 epn+B 行，对比 GPU 全局表 [E+B] 省镜像死空间）。

        Returns:
            恒为 None（async_finish 的事件位亦为 None）。

        与 dispatch 分离，使 plan 复用路径（combine bwd）可跳过重复预取。
        """
        ctx = self._require_ctx()

        assert isinstance(plan, MoonEPCommPlan), "Buffer.prefetch_weight: plan is required"
        weights = (full_gate_weight, full_up_weight, full_down_weight)
        assert all(w is not None for w in weights), \
            "prefetch_weight tensors must be provided together"
        resolved = []
        epn = int(ctx["E"]) // int(ctx["R"])
        for name, w in zip(("gate", "up", "down"), weights):
            w = self._resolve_tensor(w, f"full_{name}_weight")
            assert w.dtype == torch.bfloat16 and w.is_contiguous(), \
                f"full_{name}_weight must be contiguous bf16 [epn+B, H, H']"
            assert w.ndim == 3 and int(w.shape[0]) == epn + int(ctx["B"]), \
                f"full_{name}_weight first dim must be epn+B"
            # 注：不断言 shape[1]==H（对齐 GPU 契约，GPU api.py 仅校验首维，
            # 参考实现首维为局部表 epn+B）——down 投影自然布局为
            # [epn+B, H', H]，kernel 从张量自身
            # shape 读取行列数，逐投影行列各异不影响正确性。
            resolved.append(w)

        _launch_full_weight_prefetches(
            ctx,
            *resolved,
            plan.experts_to_copy[int(ctx["rank"])],
        )
        return None

    # ------------------------------------------------------------------
    # combine（combine fwd / dispatch bwd 的 K 份求和路径）
    # ------------------------------------------------------------------
    def combine(
        self,
        plan: "MoonEPCommPlan | None" = None,
        hidden_nvsh: "torch.Tensor | None" = None,
        route_weights_nvs: "torch.Tensor | None" = None,
        async_finish: bool = False,
        inter_rank_sync: bool = True,
        *,
        zero_copy: bool = False,
    ):
        """combine fwd：从对称缓冲区取回专家输出并按 K 求和回 token-major [S, H]。

        同时充当 dispatch bwd：对 grad_hidden_nvsh 做 combine，即把每 token 的
        K 份散布梯度求和回 token-major 梯度。

        Args:
            plan: dispatch 返回的 MoonEPCommPlan。
            hidden_nvsh: [NvS, H] bf16，物理 VM 组序的专家输出。
            route_weights_nvs: [NvS] fp32，可选；传入 dispatch 返回的权重时
                一并取回 token-major。
            async_finish: 仅保留参数位；参考实现同步执行，事件位恒为 None。
            inter_rank_sync: 暂存前做一次跨 rank 对齐（默认 True）。
            zero_copy: True 时 hidden_nvsh（及 route_weights_nvs，若给）必须
                恰为某次 zero_copy=True dispatch 返回的视图——调用方的 FFN
                原地写分片、不做边界拷贝（经 data_ptr() 断言校验）；False 时
                输入为普通张量，先拷贝进分片。

        Returns:
            (hidden_sh, route_weights_sk, event)：

            - hidden_sh: [S, H] bf16，MoonEP 新分配的 combine 输出。
            - route_weights_sk: [S, K] fp32 取回的路由权重；
              route_weights_nvs 为 None 时为 None。
            - event: 恒为 None（参考实现同步执行，无事件机制）。
        """
        ctx = self._require_ctx()

        assert isinstance(plan, MoonEPCommPlan), "Buffer.combine: plan is required"

        assert hidden_nvsh is not None
        assert hidden_nvsh.dtype == torch.bfloat16
        assert hidden_nvsh.is_contiguous()
        assert tuple(hidden_nvsh.shape) == (int(ctx["NvS"]), int(ctx["H"]))
        if route_weights_nvs is not None:
            assert route_weights_nvs.dtype == torch.float32
            assert route_weights_nvs.is_contiguous()
            assert tuple(route_weights_nvs.shape) == (int(ctx["NvS"]),)
        if zero_copy:
            # data_ptr 断言文案与 GPU api.py:936-946 逐字一致
            assert hidden_nvsh.data_ptr() == ctx["hidden_buf_local"].data_ptr(), (
                "combine(zero_copy=True): hidden_nvsh must alias the NVL shard "
                "view returned by dispatch(zero_copy=True)"
            )
            if route_weights_nvs is not None:
                assert route_weights_nvs.data_ptr() == \
                    ctx["weights_buf_local"].data_ptr(), (
                    "combine(zero_copy=True): route_weights_nvs must alias "
                    "the NVL weights view returned by dispatch(zero_copy=True)"
                )

        hidden_sh = torch.empty(
            int(ctx["S"]),
            int(ctx["H"]),
            dtype=hidden_nvsh.dtype,
            device=hidden_nvsh.device,
        )
        route_weights_sk = (
            torch.empty(
                int(ctx["S"]),
                int(ctx["K"]),
                dtype=torch.float32,
                device=hidden_nvsh.device,
            )
            if route_weights_nvs is not None else None
        )

        # ---- 步骤顺序对齐 GPU _run_combine_on_current_stream ----
        self._run_combine_on_current_stream(
            ctx,
            hidden_sh,
            plan,
            hidden_nvsh,
            route_weights_nvs,
            route_weights_sk,
            inter_rank_sync=inter_rank_sync,
            zero_copy=zero_copy,
        )
        # 参考实现同步执行：同步/异步两个模式第三位（事件位）均为 None
        return hidden_sh, route_weights_sk, None

    # ------------------------------------------------------------------
    # reduce_grad（dispatch bwd 的权重侧）
    # ------------------------------------------------------------------
    def reduce_grad(
        self,
        plan: "MoonEPCommPlan | None" = None,
        async_finish: bool = False,
        *,
        full_gate_grad: "torch.Tensor | None" = None,
        full_up_grad: "torch.Tensor | None" = None,
        full_down_grad: "torch.Tensor | None" = None,
        gate_reduce_buffer: "torch.Tensor | None" = None,
        up_reduce_buffer: "torch.Tensor | None" = None,
        down_reduce_buffer: "torch.Tensor | None" = None,
    ):
        """把副本专家的权重梯度归并回其属主 rank（dispatch bwd 的权重侧）。

        每个 rank 远程读取各 rank reduce 缓冲区中属于自己专家的槽位梯度，
        按 (r, b) 字典序累加进本地梯度行，随后清零本 rank 已被消费的槽位，
        供下一 microbatch 使用。

        Args:
            plan: dispatch 返回的 MoonEPCommPlan。
            async_finish: 仅保留参数位；参考实现同步执行，恒返回 None
                （GPU 版 async_finish=True 时返回通信流事件，参考实现无事件
                机制，事件位为 None）。
            full_gate_grad / full_up_grad / full_down_grad:
                [epn, H, H'] fp32 连续梯度张量（epn = E//R）：本 rank 属主
                专家的梯度行（按局部行号索引；普通张量即可，无需注册对称）。
            gate_reduce_buffer / up_reduce_buffer / down_reduce_buffer:
                已注册的 [B, H, H'] fp32 对称张量（传入 register() 返回的
                本 rank 物理张量；也允许直接传注册名字符串）——副本槽梯度
                的归并中转。逐投影以 full_grad 整表（属主局部表）作
                remote_expert_grads、reduce buffer 作 remote_reduce_buffers
                调 launch_grad_reduce（契约 §5）。

        Returns:
            恒为 None（async_finish 的事件位亦为 None）。

        与 combine 分离；真实训练里由共享通信流保证 Buffer 的 barrier/meta
        资源串行化，参考实现同步执行天然满足。
        """
        ctx = self._require_ctx()

        grads = (full_gate_grad, full_up_grad, full_down_grad)
        buffers = (gate_reduce_buffer, up_reduce_buffer, down_reduce_buffer)
        assert all(t is not None for t in (*grads, *buffers)), \
            "reduce_grad tensors must be provided together"
        assert isinstance(plan, MoonEPCommPlan), "Buffer.reduce_grad: plan is required"
        epn = int(ctx["E"]) // int(ctx["R"])
        for name, full_grad in zip(("gate", "up", "down"), grads):
            assert full_grad.dtype == torch.float32 and full_grad.is_contiguous(), \
                f"full_{name}_grad must be contiguous fp32 [epn, H, H']"
            assert full_grad.ndim == 3 and int(full_grad.shape[0]) == epn, \
                f"full_{name}_grad first dim must be epn=E//R"
            # 同 prefetch_weight：不断言 shape[1]==H，允许 down 投影 [epn, H', H]。
        resolved_buffers = []
        for name, buf in zip(("gate", "up", "down"), buffers):
            buf = self._resolve_tensor(buf, f"{name}_reduce_buffer")
            assert buf.dtype == torch.float32 and buf.is_contiguous(), \
                f"{name}_reduce_buffer must be contiguous fp32 [B, H, H']"
            assert buf.ndim == 3 and int(buf.shape[0]) == int(ctx["B"]), \
                f"{name}_reduce_buffer first dim must be B"
            resolved_buffers.append(buf)

        _launch_full_grad_reduces(
            ctx,
            plan.experts_to_copy,
            *grads,
            *resolved_buffers,
        )
        return None
