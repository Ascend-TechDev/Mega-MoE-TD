# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Build and link the tiny SHMEM PIPE_S UDMA device shim.

Ascend Triton's extern ABI expects ``_mlir_ciface_*`` symbols in an LLVM
bitcode module.  SHMEM 1.6 exposes the low-level UDMA implementation only as
an always-inline C++ device function, so it has to be instantiated once for
BF16.  The resulting module is content-addressed under the system temporary
directory and shared by all local ranks.
"""

from functools import lru_cache
import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile


_PUT_SYMBOL = "_mlir_ciface_aclshmemi_udma_put_nbi"
_PUT_SIGNAL_SYMBOL = "_mlir_ciface_aclshmemx_udma_put_signal_nbi"
_QUIET_SYMBOL = "_mlir_ciface_aclshmemx_udma_quiet"
_BITCODE_ABI_VERSION = b"owner-push-symbols-v4"


def _require_directory(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise RuntimeError(f"{description} directory does not exist: {path}")
    return path


def _resolve_cann_root() -> Path:
    configured = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME"
    )
    if not configured:
        raise RuntimeError(
            "CANN environment is not initialized; source set_env.sh before "
            "compiling MoonEP UDMA kernels"
        )
    return _require_directory(Path(configured).resolve(), "CANN")


def _resolve_shmem_root() -> Path:
    spec = importlib.util.find_spec("shmem")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("SHMEM 1.6 Python package is not installed")
    return _require_directory(
        Path(next(iter(spec.submodule_search_locations))).resolve(),
        "SHMEM package",
    )


def _resolve_cxx_include_dirs() -> list[Path]:
    cxx_root = Path("/usr/include/c++")
    versions = sorted(
        (entry for entry in cxx_root.iterdir() if entry.is_dir()),
        key=lambda entry: tuple(
            int(part) if part.isdigit() else -1
            for part in entry.name.replace(".", " ").split()
        ),
        reverse=True,
    ) if cxx_root.is_dir() else []
    if not versions:
        raise RuntimeError("BiSheng device shim build cannot locate libstdc++ headers")
    version_dir = versions[0]
    target_dirs = sorted(
        entry
        for entry in version_dir.iterdir()
        if entry.is_dir() and (entry / "bits" / "c++config.h").is_file()
    )
    include_dirs = [version_dir]
    if target_dirs:
        include_dirs.append(target_dirs[0])
    backward = version_dir / "backward"
    if backward.is_dir():
        include_dirs.append(backward)
    return include_dirs


def _run_build_command(command: list[str], description: str) -> None:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        output_lines = result.stdout.splitlines()
        tail = "\n".join(output_lines[-80:])
        raise RuntimeError(f"{description} failed:\n{tail}")


def _compile_command(
    bisheng: Path,
    source: Path,
    output: Path,
    cann_root: Path,
    shmem_root: Path,
    compatibility_device_include: Path,
) -> list[str]:
    clang_roots = sorted(
        (cann_root / "tools" / "bisheng_compiler" / "lib" / "clang").glob("*")
    )
    if not clang_roots:
        raise RuntimeError("CANN BiSheng Clang resource directory is missing")
    resource_dir = clang_roots[-1]
    include_dirs = [
        cann_root / "include",
        cann_root / "include" / "ascendc" / "highlevel_api",
        cann_root / "asc",
        cann_root / "asc" / "include",
        cann_root / "asc" / "include" / "basic_api",
        cann_root / "asc" / "include" / "utils" / "debug",
        shmem_root / "include",
        shmem_root / "src",
        shmem_root / "src" / "device",
        compatibility_device_include,
    ]
    command = [
        str(bisheng),
        "-cc1",
        "-triple",
        "hiipu64-hisilicon-cce",
        "-std=c++17",
        "-O3",
        "-D__NPU_DEVICE__",
        "-D__CCE_AICORE_ENABLE_MIX__",
        "-DTILING_KEY_VAR=0",
        "-fcce-is-aicore",
        "-cce-enable-mix",
        "-mllvm",
        "-enable-mix=true",
        "-fcce-enable-asc-lang",
        "-mllvm",
        "-enable-ascendc-lang=true",
        "-target-cpu",
        "dav-c310-vec",
        "-emit-llvm-bc",
        "-resource-dir",
        str(resource_dir),
        "-include",
        "__clang_cce_runtime_wrapper.h",
    ]
    for include_dir in include_dirs:
        command.extend(("-I", str(include_dir)))
    for include_dir in _resolve_cxx_include_dirs():
        command.extend(("-internal-isystem", str(include_dir)))
    command.extend(
        (
            "-internal-isystem",
            str(resource_dir / "include"),
            "-internal-isystem",
            "/usr/local/include",
            "-internal-externc-isystem",
            "/include",
            "-internal-externc-isystem",
            "/usr/include",
            "-fgnuc-version=4.2.1",
            "-fcxx-exceptions",
            "-fexceptions",
            "-x",
            "asc",
            str(source),
            "-o",
            str(output),
        )
    )
    return command


def _cache_digest(source: Path, shmem_root: Path, bisheng: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_BITCODE_ABI_VERSION)
    inputs = (
        source,
        source.parent / "shmem_compat" / "host_device"
        / "shmemi_host_device_constant.h",
        shmem_root / "src" / "device" / "gm2gm" / "engine"
        / "shmem_device_udma.hpp",
        shmem_root / "src" / "device" / "gm2gm" / "engine"
        / "shmemi_device_udma.h",
    )
    for input_path in inputs:
        if not input_path.is_file():
            raise RuntimeError(f"required UDMA source is missing: {input_path}")
        digest.update(input_path.read_bytes())
    compiler_stat = bisheng.stat()
    digest.update(str(bisheng).encode())
    digest.update(str(compiler_stat.st_size).encode())
    digest.update(str(compiler_stat.st_mtime_ns).encode())
    return digest.hexdigest()[:20]


@lru_cache(maxsize=1)
def udma_pipe_s_bitcode_path() -> str:
    """Return a cached BF16 PIPE_S shim, building it once when necessary."""
    cann_root = _resolve_cann_root()
    shmem_root = _resolve_shmem_root()
    csrc_root = Path(__file__).resolve().parents[1] / "kernels" / "csrc"
    source = csrc_root / "shmem_udma_pipe_s.cpp"
    compatibility_device_include = csrc_root / "shmem_compat" / "device"
    bisheng = cann_root / "tools" / "bisheng_compiler" / "bin" / "bisheng"
    if not bisheng.is_file():
        raise RuntimeError(f"CANN BiSheng compiler is missing: {bisheng}")

    cache_root = Path(tempfile.gettempdir()) / "mega_moe_udma_pipe_s"
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = _cache_digest(source, shmem_root, bisheng)
    output = cache_root / f"shmem_udma_pipe_s_{digest}.bc"
    lock_path = cache_root / f"shmem_udma_pipe_s_{digest}.lock"

    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if output.is_file() and output.stat().st_size:
            return str(output)
        with tempfile.TemporaryDirectory(dir=cache_root) as build_directory:
            build_root = Path(build_directory)
            raw_bitcode = build_root / "raw.bc"
            raw_llvm = build_root / "raw.ll"
            link_llvm = build_root / "link.ll"
            staged_output = build_root / "linked.bc"
            _run_build_command(
                _compile_command(
                    bisheng,
                    source,
                    raw_bitcode,
                    cann_root,
                    shmem_root,
                    compatibility_device_include,
                ),
                "compiling the SHMEM PIPE_S shim",
            )
            _run_build_command(
                [
                    str(bisheng),
                    "-cc1",
                    "-triple",
                    "hiipu64-hisilicon-cce",
                    "-emit-llvm",
                    "-x",
                    "ir",
                    str(raw_bitcode),
                    "-o",
                    str(raw_llvm),
                ],
                "disassembling the SHMEM PIPE_S shim",
            )
            llvm = raw_llvm.read_text(encoding="utf-8")
            suffixed_symbols = (
                f"@{_PUT_SYMBOL}.vector",
                f"@{_PUT_SIGNAL_SYMBOL}.vector",
                f"@{_QUIET_SYMBOL}.vector",
            )
            for suffixed in suffixed_symbols:
                if llvm.count(suffixed) != 1:
                    raise RuntimeError(
                        f"unexpected BiSheng symbol layout for {suffixed}"
                    )
            # A standalone AIV kernel calls the plain C-interface symbol,
            # whereas an AIC/AIV mixed kernel appends ``.vector``.  Preserve
            # BiSheng's vector definitions and export plain aliases so the
            # same bitcode works for both compilation modes.
            attribute_marker = "\nattributes #0"
            if llvm.count(attribute_marker) != 1:
                raise RuntimeError("unexpected BiSheng LLVM attribute layout")
            aliases = (
                f"\n@{_PUT_SYMBOL} = dso_local alias "
                f"void (ptr, ptr, i32, i32), ptr @{_PUT_SYMBOL}.vector\n"
                f"@{_PUT_SIGNAL_SYMBOL} = dso_local alias "
                "void (ptr, ptr, i32, ptr, i32, i32), ptr "
                f"@{_PUT_SIGNAL_SYMBOL}.vector\n"
                f"@{_QUIET_SYMBOL} = dso_local alias "
                f"void (i32), ptr @{_QUIET_SYMBOL}.vector\n"
            )
            llvm = llvm.replace(
                attribute_marker,
                aliases + attribute_marker,
                1,
            )
            link_llvm.write_text(llvm, encoding="utf-8")
            _run_build_command(
                [
                    str(bisheng),
                    "-cc1",
                    "-triple",
                    "hiipu64-hisilicon-cce",
                    "-emit-llvm-bc",
                    "-x",
                    "ir",
                    str(link_llvm),
                    "-o",
                    str(staged_output),
                ],
                "assembling the SHMEM PIPE_S shim",
            )
            os.replace(staged_output, output)
    return str(output)


def udma_pipe_s_bisheng_options() -> str:
    """Return complete BiSheng link flags for a PIPE_S Triton kernel."""
    from triton.backends.ascend.compiler import get_libdevice

    return (
        f"-cce-link-aicore-ll-module {get_libdevice()} "
        f"-cce-link-aicore-ll-module {udma_pipe_s_bitcode_path()}"
    )


__all__ = ["udma_pipe_s_bisheng_options", "udma_pipe_s_bitcode_path"]
