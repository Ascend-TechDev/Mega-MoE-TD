# Current-human baseline harness design philosophy

<!-- CANONICAL_ARCHITECTURE_SOURCE: current-human-baseline-v1 -->

This document is the single architecture authority for the Mega-MoE-TD
current-human baseline. README tables, historical branches, private bundles,
chat transcripts, and old result directories are not executable evidence.

## 1. Thesis

The harness exists to make one narrow claim replayable: on the exact repository
identity and the same Qwen3-30B-A3B EP=2 fixtures, measure the real current-main
unfused and fused forward paths and the two current-main backward wgrad modes.
The harness is an evidence producer, not an optimizer and not a compatibility
layer for historical metrics.

## 2. Component boundaries

### 2.1 Fixed experiment contract

`benchmark/contracts/current_human_baseline_v1.schema.json` and
`benchmark/baseline_contract.py` jointly freeze:

- repository URL, commit, tree, and the pinned `3rdparty/bigop` gitlink;
- Qwen3-30B-A3B, BF16, expert-parallel world size 2;
- 4096, 8192, and 16384 tokens per rank with one fixture seed;
- four and only four arms: grouped-GEMM+HCCL forward, fused current-main
  forward, backward with default Torch wgrad, and backward with opt-in Triton
  wgrad;
- identical warmup count, sample count, synchronized full-call timing boundary,
  and per-sample rank-MAX reduction;
- forward full-output correctness and backward five-gradient correctness before
  any timing;
- canonical JSON payload and file hashes with full sample arrays.
- exact `MOE_FULL_BENCH_ROUTE_MODE=dense_random` and
  `MOE_FULL_BENCH_ACTIVE_EXPERTS=8` bindings plus explicit
  `MOE_BWD_TRACE=""` (the exact trace-disabled value used by current-main's
  `bool(os.environ.get("MOE_BWD_TRACE"))`) in the plan, environment, and
  execution receipt.

Any missing or conflicting field is invalid evidence. The Python validator is
the executable authority; the JSON schema is the portable shape declaration.

### 2.2 One session runner

`benchmark/current_human_baseline.py` is the only authoritative entry point.
It creates one identity-bound plan and directly invokes provider adapters. It
does not execute or parse the older forward/backward benchmark runners. Those
runners remain diagnostics because their fixtures and timing protocols differ.

Host `--dry-run` loads only the contract and provider descriptions. Device
execution additionally requires an external environment receipt before any
Torch/NPU import. The receipt must bind Python, Torch, torch-npu, Triton, CANN,
ACLSHMEM, and bigop. Absence or mismatch is terminal `INVALID`.

The runner first validates the expected receipt and explicit routing variables.
It then imports only the runtime components needed to recompute their real
versions, source files or gitlink, and hashes. The recomputed environment must
equal the receipt before device selection, process-group initialization, or
fixture construction. CANN identity must come from a version file beneath the
active toolkit root; bigop identity is the fixed product-base gitlink.
The routing environment is checked again immediately before each backward
fixture so trace drift cannot enter backward correctness or timing.

### 2.3 Providers

`benchmark/providers/grouped_hccl.py` calls the exact current-main unfused
Torch-NPU grouped-GEMM + HCCL function. It is the real small-op forward
baseline. `benchmark/providers/current_main.py` calls the exact production
fused forward and backward APIs and controls only the documented
`MOE_WGRAD_TRITON` switch, restoring the caller environment after each call.

Providers describe their repository source and paths without importing device
modules. There is no Megatron arm, mock provider, README fallback, or alternate
provider registry in this contract.

### 2.4 Receipt validator

The runner emits no result when an arm raises, a correctness gate fails, an
identity is absent, or a sample array is incomplete. A complete receipt is a
canonical envelope containing the payload hash, all four arms, every shape,
every correctness gate, and all raw rank-MAX samples. A second sidecar hashes
the exact serialized envelope bytes. The COMPLETE payload binds a freshly
recomputed clean checkout identity (origin, HEAD/tree/parents, product base,
and exact ten-path delta) plus the required sidecar algorithm/suffix. Consumers
must use the sidecar-verifying read API; a missing or tampered sidecar is not a
receipt.

DRY_RUN and COMPLETE validation accept an explicit physical Git checkout path,
never a caller-supplied identity mapping. The locator must be absolute,
lexically identical to its resolved path, non-symlink, and the exact checkout
top-level with a physical in-tree `.git` directory. Sanitized local Git plumbing
binds HEAD/tree/sole-parent, designated branch, product ancestry, clean state,
and the cumulative ten-path delta. Local origin and remote-tracking refs are not
authority: each validation performs a sanitized live `git ls-remote --heads`
against the fixed GitCode URL and exact reviewed feature ref, requiring exactly
one live ref equal to local HEAD. Both statuses persist that checkout locator,
repository, branch, live ref/commit, and local object identity, and consumers
rerun the same verifier. Raw receipt bytes must also equal the canonical
serialization exactly; reformatting and rehashing the sidecar does not produce
valid evidence. The portable schema declares every required plan field while
the Python validator checks the full fixed plan values.

## 3. Validation rigor

Host tests are causal: mutations to the canonical sections or four-arm schema
must turn strict lint red; missing commit/tree/provider/environment identity
must fail; missing gradients or a truncated sample array must fail; and dry-run
must prove that no Torch, torch-npu, Triton, or ACLSHMEM module was imported.
Duplicate canonical documents or runners, an eleventh provider, a deleted
harness path, a dirty checkout, route-variable drift, placeholder environment
identity, live-ref drift, a self-signed checkout mapping, an incomplete plan,
noncanonical raw JSON, missing live authority, path aliases, detached or locally
forged repositories, dry-run identity omission, and sidecar tampering must
independently turn their gates red.

Device evidence is not accepted merely because the process exits zero. Every
requested shape must appear, every precision result must be `PASS`, every arm
must contain the exact sample count, and the canonical hashes must verify.

## 4. Evidence principles

1. **Exact identity precedes execution.** Repository commit/tree, submodule,
   providers, model, parallelism, shapes, switches, and environment are inputs.
2. **No silent skip.** Exceptions, missing arms/shapes, OOM, or initialization
   failures terminate without a complete receipt.
3. **Correctness precedes timing.** Forward output and all five backward
   gradients must pass on the exact fixture before collecting a sample.
4. **Same-session pairing.** Legal comparisons share checkout, environment,
   device set, fixture contract, warmup, samples, and timing boundary.
5. **Only like operations compare.** Grouped+HCCL forward may compare with fused
   forward. Backward default may compare with backward Triton-wgrad. A
   forward/backward latency ratio is invalid.
6. **Raw evidence is load-bearing.** Full sample arrays and canonical hashes are
   retained; medians, summaries, and tables are derived views only.
7. **History is never a provider.** README metrics, old bundles, old result
   files, and chat receipts cannot satisfy a current run.
8. **Dry-run is not performance.** Host validation proves schema, routing, and
   failure behavior only and must never be reported as precision or speed.
9. **No test provider in production.** This repository defines no fake/mock
   provider path; provider identity is the exact current-main source.
10. **Fail closed.** Ambiguity is `INVALID`, never a best-effort substitute.

## 5. Mechanized invariants

`python scripts/architecture_lint.py --strict` enforces the sole canonical
marker, required sections, exact paths, arm/shape sets, provider identities,
single authoritative runner, host-safe provider imports, and prohibition on
README fallback. Mutation-sensitive host tests prevent the lint itself from
becoming a prose-only gate.
