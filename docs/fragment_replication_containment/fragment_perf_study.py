"""Measure compilation and CUDA Graph kernel time in separate processes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


WIDTHS = (96, 100, 120, 127, 128, 256)


def make_kernel(width, mode):
    import tilelang
    import tilelang.language as T

    padded = (width + 31) // 32 * 32 if mode == "guarded" else width
    if mode != "implicit":
        layout = tilelang.Fragment(
            (1, padded),
            forward_thread_fn=lambda i, j, rep: j % 32 + rep * 32,
            forward_index_fn=lambda i, j: j // 32,
            replicate=4,
        )

    @T.prim_func
    def main(X: T.Tensor((1, width), "float32"), Y: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((1, padded), "float32")
            result = T.alloc_fragment((1,), "float32")
            if mode != "implicit":
                T.annotate_layout({fragment: layout})
            if mode == "guarded":
                for i, j in T.Parallel(1, padded):
                    if j < width:
                        fragment[i, j] = X[i, j]
                    else:
                        fragment[i, j] = 0.0
            else:
                T.copy(X, fragment)
            T.reduce_sum(fragment, result, dim=1)
            T.copy(result, Y)

    return main


def worker(args):
    import tilelang
    import torch

    source = Path(args.source).resolve()
    actual = Path(tilelang.__file__).resolve().parent.parent
    if actual != source:
        raise RuntimeError("Compiler source does not match the requested checkout")
    library = Path(tilelang._LIB_PATH).resolve()
    if not library.is_relative_to(source):
        raise RuntimeError("Compiler library does not match the requested checkout")
    row = {
        "width": args.width,
        "layout": args.layout,
        "compiler_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    try:
        function = make_kernel(args.width, args.layout)
        start = time.perf_counter()
        try:
            kernel = tilelang.compile(function, target="cuda", execution_backend="tvm_ffi")
        finally:
            row["compile_ms"] = (time.perf_counter() - start) * 1000
        row["stage"] = "execute"
        generator = torch.Generator(device="cuda").manual_seed(2399)
        inputs = {
            "arange": torch.arange(args.width, dtype=torch.float32, device="cuda").reshape(1, args.width),
            "ones": torch.ones((1, args.width), dtype=torch.float32, device="cuda"),
            "random": torch.randn((1, args.width), dtype=torch.float32, device="cuda", generator=generator),
        }
        output = torch.empty((1,), dtype=torch.float32, device="cuda")
        errors = {}
        for name, x in inputs.items():
            output.fill_(float("nan"))
            kernel(x, output)
            torch.cuda.synchronize()
            expected = x.double().sum(dim=1).float()
            torch.testing.assert_close(output, expected, rtol=0, atol=1e-4)
            errors[name] = float((output - expected).abs().max())
        x = inputs["random"]
        for _ in range(5):
            kernel(x, output)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.graph_iterations):
                kernel(x, output)
        timings = []
        for _ in range(args.timing_repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            timings.append(begin.elapsed_time(end) * 1000 / args.graph_iterations)
        row.update(status="pass", max_absolute_error=errors, graph_latency_us=timings)
    except Exception as error:  # Record compilation and runtime failures as distinct outcomes.
        row.update(status="failed", stage=row.get("stage", "compile"), error_type=type(error).__name__)
    args.output.write_text(json.dumps(row, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--patched", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graph-iterations", type=int, default=100)
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--compiler", choices=["baseline", "patched", "both"], default="both")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--layout", choices=["implicit", "warp", "guarded"])
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if not args.baseline or not args.patched:
        parser.error("--baseline and --patched are required")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for width in WIDTHS:
            explicit = "guarded" if width % 32 else "warp"
            cases = [
                ("baseline", args.baseline, "implicit"),
                ("baseline", args.baseline, explicit),
                ("patched", args.patched, "implicit"),
                ("patched", args.patched, explicit),
            ]
            for label, source, layout in cases:
                if args.compiler != "both" and args.compiler != label:
                    continue
                for sample in range(args.repeats):
                    output = Path(tmp) / "sample.json"
                    output.unlink(missing_ok=True)
                    cmd = [
                        args.python,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--source",
                        str(source.resolve()),
                        "--width",
                        str(width),
                        "--layout",
                        layout,
                        "--output",
                        str(output),
                        "--graph-iterations",
                        str(args.graph_iterations),
                        "--timing-repeats",
                        str(args.timing_repeats),
                    ]
                    env = os.environ.copy()
                    env.update(PYTHONPATH=str(source.resolve()), TILELANG_DISABLE_CACHE="1", TVM_FFI_DISABLE_TORCH_C_DLPACK="1")
                    process = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180, check=False)
                    if process.returncode or not output.exists():
                        raise RuntimeError(f"Worker infrastructure failed: {label} {layout} {width}: {process.stderr[-1200:]}")
                    row = json.loads(output.read_text())
                    row.update(compiler=label, sample=sample)
                    rows.append(row)
                    print(f"{label} {layout} N={width} sample={sample}: {row['status']}", flush=True)
    summary = []
    for width in WIDTHS:
        for compiler, layout in [
            ("baseline", "implicit"),
            ("baseline", "guarded" if width % 32 else "warp"),
            ("patched", "implicit"),
            ("patched", "guarded" if width % 32 else "warp"),
        ]:
            if args.compiler != "both" and args.compiler != compiler:
                continue
            selected = [r for r in rows if r["width"] == width and r["compiler"] == compiler and r["layout"] == layout]
            passed = [r for r in selected if r["status"] == "pass"]
            item = {"width": width, "compiler": compiler, "layout": layout, "passed": len(passed), "total": len(selected)}
            if passed:
                item["compile_median_ms"] = statistics.median(r["compile_ms"] for r in passed)
                item["latency_median_us"] = statistics.median(statistics.median(r["graph_latency_us"]) for r in passed)
            summary.append(item)
    args.output.write_text(
        json.dumps(
            {
                "device": "A100",
                "graph_iterations": args.graph_iterations,
                "timing_repeats": args.timing_repeats,
                "samples": rows,
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
