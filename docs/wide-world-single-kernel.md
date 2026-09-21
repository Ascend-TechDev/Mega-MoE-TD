# Wide-world single-kernel forward (W > AICore count)

The single-kernel forward (`enable_single_kernel_forward`) supports EP worlds
wider than the per-card AICore program count — e.g. one EP group of W=128
across 16 x 8-card servers running Kimi-K3 (E=896, 7 home experts per rank,
MoonEP replica budget 7).  The historical constructor gate that rejected
`world_size > num_aicore_programs` is gone; the sections below record what
changed, the memory budget at wide worlds, and the bring-up sequence.

## What the kernel now does differently

| Concern | Mechanism |
|---|---|
| Destination coverage | Dispatch programs stride units: `for unit in range(pid, max(cores, W), cores)` with `destination = unit % W, lane = unit // W`. For `W <= cores` this is bit-identical to the old `pid % W` mapping, so the W=2/8 dist suites validate the same assignment they always did. |
| Return coverage | Return workers (2 per core) stride `max(2*cores, W)` units; `_wait_dynamic_wave_returns` expects `cdiv(max(2*cores, W) - local_rank, W)` ADDs per slot. |
| Per-rank metadata/planning | All `pid < WORLD_SIZE` gates became strided loops; `_moonep_b2_home` / `_moonep_b3_destination` / `_balanced_count_cube_destination` take their rank index as an explicit argument (grid wrappers keep the multi-kernel launches unchanged). |
| UDMA replica push | One Vector subcore owns every peer QP assigned to its program — 4 peers per subcore at W=128 on 32 cores. Still exactly one owner per peer. |
| Scatter destination lookup | `tl.static_range` scans (O(W) compile-time unroll, per-rank specialization) replaced by a `source_prefix[E]` table plus a fixed-step binary search over the non-decreasing `alloc_cumsum` row. |
| Planning B.0/B.1 | B.0 folds the per-source loop into one `(R, BLOCK_E)` gather; B.1 keeps balance and the transfer matrix in global memory (this backend cannot lower vector selects through a dynamic `scf.for`, see `_kernel_moonep_b2`'s note) instead of a register-resident `(R, R)` matrix. |
| Dispatch readiness | `_wait_dispatch_row_range` binary-searches a per-expert segment-start table `[EPR, W+1]` (two interleaved `W.bit_length()`-step searches bracket the overlapping sources) instead of scanning all W sources per FC1 tile. |
| Per-rank binaries | `LOCAL_RANK` is a non-specialized runtime argument; every rank of a world compiles and shares one binary. |

The fixed-step binary searches (scatter destination, dispatch readiness) run
exactly `W.bit_length()` steps.  Fewer steps silently return an unconverged
lane — the formula and its guard are host-modeled exhaustively in
`tests/function/test_single_kernel_wide_world.py`.

## Memory budget notes at W=128 (Kimi-K3, tokens=4096)

- **Dispatch signal slots are world-invariant**: `W x physical_experts x
  max_source_tiles` with `physical_experts = 2E/W`, so the product is
  `2E x max_source_tiles` at every world size.
- **Count cube** `[W, NUM_BINS_PAD]`: 128 x 2048 int32 = 1 MB per rank.
- **MoonEP planning tables** (`alloc_cumsum [E, W]`, `inverse [W, E]`):
  ~450 KB each.
- **Replica symmetric buffers shrink with world size**: budget B = E/W
  slots, so W=128 needs 7 slots (~440 MB) versus 112 slots (~7 GB) at W=8.
  `MOE_FUSED_ASH_SIZE_GB` needs go **down**, not up.
- **`receive_capacity_factor` must be explicit** (e.g. 1.25).  The `None`
  default resolves to `float(world_size)` and would budget `peer_mem` for
  the entire world's routes — infeasible at W=128.

## Validation without 128 cards

1. Host model tests: `python -m pytest tests/function/test_single_kernel_wide_world.py`
   (pure Python; no torch/triton/NPU needed).
2. Compile validation (no NPU allocation):

   ```bash
   python benchmark/layer/compile_single_kernel_forward.py \
     --case performance-fwd-kimi-k3-wide-w128-t64 --moonep --cores 32 \
     --output-dir /tmp/w128-t64
   ```

   `--rank` no longer changes the binary hash (shared-binary check).
   Compile the t4k variant and compare wall time against the w8 trimmed
   case to confirm no world-size compile explosion.
3. The W=2/8 dist suites are the behavioral equivalence proof for every
   code path that also runs at W <= 32 (all of them — wide worlds only
   change the unit striding, which degenerates to the old mapping).

## Bring-up sequence for 16 x 8 servers

- **w64 first**: only the dispatch side strided mapping is new there
  (return workers still cover 64 = 2*32 units exactly as before); it is the
  natural intermediate step.
- Then w128 with the MoonEP skewed-route benchmark and
  `profile_single_kernel_forward.py` stage timing.
- Watch two known wide-world effects that are correct but can cost
  performance: `global_waves` advances at the slowest rank's pace (tail
  idle waves grow with world size), and each UDMA-owning Vector subcore
  serializes 4 peers' pushes (submission is asynchronous, but the window
  is 4x longer than at W=8).

## Still open (out of scope for this change)

- The multi-kernel MoonEP path (`return_saved` training forward) still has
  `tl.static_range(0, LOCAL_RANK)` in `_kernel_map_balanced_routes` and
  `tl.static_range(0, R)` in `_kernel_finalize_balanced_metadata` — same
  compile explosion at W=128, separate change.
- Backward path has not been audited for wide worlds at all.
