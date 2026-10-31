"""Probe fragment replication without modifying the compiler implementation."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIDTHS = (64, 96, 100, 120, 127, 128, 192, 229, 256, 257)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_metadata(source):
    import tilelang

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    actual = Path(tilelang.__file__).resolve().parent.parent
    if actual != source:
        raise RuntimeError("TileLang was imported from a different checkout")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise RuntimeError("The baseline has tracked source modifications")
    libraries = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        if any(name in line for name in ("libtilelang.so", "libtvm_compiler.so")):
            library = Path(line.split()[-1]).resolve()
            if not library.is_relative_to(source):
                raise RuntimeError("A compiler library belongs to a different checkout")
            libraries[library.name] = hashlib.sha256(library.read_bytes()).hexdigest()
    if "libtilelang.so" not in libraries:
        raise RuntimeError("The loaded compiler library was not identified")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != tomllib.loads((ROOT / "versions.toml").read_text())["baseline"]["commit"]:
        raise RuntimeError("The compiler checkout does not match the baseline manifest")
    return {
        "commit": commit,
        "tracked_source_clean": True,
        "native_libraries_sha256": libraries,
    }


def build_function(width, kind, layout):
    import tilelang
    import tilelang.language as T

    padded = (width + 31) // 32 * 32 if layout in ("padded_warp", "padded_guard") else width
    if layout in ("warp", "padded_warp", "padded_guard"):
        fragment_layout = tilelang.Fragment(
            (1, padded),
            forward_thread_fn=lambda i, j, rep: j % 32 + rep * 32,
            forward_index_fn=lambda i, j: j // 32,
            replicate=4,
        )

    @T.prim_func
    def function(X: T.Tensor((1, width), "float32"), Y: T.Tensor((1, width), "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((1, padded), "float32")
            if layout != "implicit":
                T.annotate_layout({fragment: fragment_layout})
            if layout == "padded_guard":
                for i, j in T.Parallel(1, padded):
                    if j < width:
                        fragment[i, j] = X[i, j]
                    else:
                        fragment[i, j] = 0.0
            elif layout == "padded_warp":
                T.clear(fragment)
                T.copy(X, fragment[0:1, 0:width])
            else:
                T.copy(X, fragment)
            if kind == "elementwise":
                for i, j in T.Parallel(1, padded):
                    fragment[i, j] = fragment[i, j] * 2.0
            if layout == "padded_guard":
                for i, j in T.Parallel(1, padded):
                    if j < width:
                        Y[i, j] = fragment[i, j]
            elif layout == "padded_warp":
                T.copy(fragment[0:1, 0:width], Y)
            else:
                T.copy(fragment, Y)

    @T.prim_func
    def reduction(X: T.Tensor((1, width), "float32"), Y: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((1, padded), "float32")
            result = T.alloc_fragment((1,), "float32")
            if layout != "implicit":
                T.annotate_layout({fragment: fragment_layout})
            if layout == "padded_guard":
                for i, j in T.Parallel(1, padded):
                    if j < width:
                        fragment[i, j] = X[i, j]
                    else:
                        fragment[i, j] = 0.0
            elif layout == "padded_warp":
                T.clear(fragment)
                T.copy(X, fragment[0:1, 0:width])
            else:
                T.copy(X, fragment)
            T.reduce_sum(fragment, result, dim=1)
            T.copy(result, Y)

    return reduction if kind == "reduce" else function


def run_case(args):
    import tilelang
    import torch

    row = {"width": args.width, "operation": args.kind, "layout": args.layout}
    row["metadata"] = source_metadata(args.source)
    row["gpu"] = "A100" if "A100" in torch.cuda.get_device_name() else "other"
    stage = "frontend"
    try:
        function = build_function(args.width, args.kind, args.layout)
        stage = "compile"
        kernel = tilelang.compile(
            function,
            target="cuda",
            execution_backend="tvm_ffi",
            pass_configs={
                "tl.enable_dump_ir": True,
                "tl.dump_ir_path": str(args.raw / "ir"),
            },
        )
        (args.raw / "kernel.cu").write_text(kernel.get_kernel_source())
        stage = "execute"
        generator = torch.Generator(device="cuda").manual_seed(2399)
        patterns = {
            "arange": torch.arange(args.width, device="cuda", dtype=torch.float32).reshape(1, args.width),
            "ones": torch.ones((1, args.width), device="cuda"),
            "random": torch.randn((1, args.width), generator=generator, device="cuda"),
        }
        errors = {}
        for name, inputs in patterns.items():
            expected = inputs.double().sum(1).float() if args.kind == "reduce" else inputs * (2 if args.kind == "elementwise" else 1)
            output = torch.empty_like(expected)
            for _ in range(3):
                output.fill_(float("nan"))
                kernel(inputs, output)
                torch.cuda.synchronize()
                torch.testing.assert_close(output, expected, rtol=0, atol=1e-4)
            errors[name] = float((output - expected).abs().max())
        row.update(status="passed", max_absolute_error=errors)
    except Exception as error:  # noqa: BLE001 - preserve compiler and runtime failures as experiment outcomes
        import traceback

        traceback.print_exc()
        lines = str(error).splitlines()
        signal = [
            line.strip()
            for line in lines
            if any(
                word in line
                for word in (
                    "Inconsistent layouts",
                    "replicate",
                    "Fragment(",
                    "no available layout",
                    "surjective",
                    "bijective",
                    "AssertionError",
                    "Check failed",
                )
            )
        ]
        row.update(
            status="failed",
            stage=stage,
            error_type=type(error).__name__,
            diagnostic=signal[-12:] or lines[-3:],
        )
    write_json(args.output, row)
    print(f"{args.kind} N={args.width} {args.layout}: {row['status']}", flush=True)
    return 0 if row["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=["implicit"],
        choices=["implicit", "warp", "padded_warp", "padded_guard"],
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        default=["copy", "elementwise", "reduce"],
        choices=["copy", "elementwise", "reduce"],
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return a nonzero status when any case fails",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--kind", help=argparse.SUPPRESS)
    parser.add_argument("--layout", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--raw", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source = args.source.resolve()
    if args.worker:
        return run_case(args)
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    pin = tomllib.loads((ROOT / "versions.toml").read_text())["baseline"]["commit"]
    commit = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if commit != pin:
        raise ValueError("Source checkout does not match the baseline manifest")
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(args.source),
        TILELANG_DISABLE_CACHE="1",
        TVM_FFI_DISABLE_TORCH_C_DLPACK="1",
        TVM_LOG_DEBUG="DEFAULT=1",
    )
    rows = []
    for layout in args.layouts:
        for width in args.widths:
            for kind in args.operations:
                name = f"{layout}_{kind}_{width}"
                raw = args.raw_dir.resolve() / name
                raw.mkdir(parents=True, exist_ok=True)
                output = raw / "case.json"
                output.unlink(missing_ok=True)
                command = [
                    str(args.python),
                    str(Path(__file__).resolve()),
                    "--source",
                    str(args.source),
                    "--raw-dir",
                    str(args.raw_dir),
                    "--worker",
                    "--width",
                    str(width),
                    "--kind",
                    kind,
                    "--layout",
                    layout,
                    "--raw",
                    str(raw),
                    "--output",
                    str(output),
                ]
                with (raw / "compile.log").open("w") as log:
                    completed = subprocess.run(
                        command,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=180,
                        check=False,
                    )
                if not output.exists():
                    raise RuntimeError(f"Worker failed before recording a verdict: {name}, exit={completed.returncode}")
                row = json.loads(output.read_text())
                if completed.returncode != (0 if row["status"] == "passed" else 1):
                    raise RuntimeError("Worker verdict and exit code disagree")
                rows.append(row)
                print(f"{name}: {row['status']}", flush=True)
                write_json(args.output_dir / "execution.json", {"cases": rows})
    return int(args.require_pass and any(row["status"] != "passed" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
