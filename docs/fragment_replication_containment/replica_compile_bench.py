"""Measure replica condensation in isolation, including owner preservation."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import tilelang
from tilelang import tvm


def make_fragment(width):
    return tilelang.Fragment(
        (1,),
        forward_thread_fn=lambda i, rep: (rep % width) % 32 + (rep // width) * 32,
        forward_index_fn=lambda i: 0,
        replicate=4 * width,
    )


def owner_set(fragment):
    variables = list(fragment.get_forward_vars())
    analyzer = tvm.arith.Analyzer()
    result = set()
    for rep in range(int(fragment.replicate_size)):
        values = [rep, 0]
        mapping = {v: tvm.tirx.IntImm("int32", n) for v, n in zip(variables, values, strict=True)}
        result.add(int(analyzer.simplify(tvm.tirx.stmt_functor.substitute(fragment.thread, mapping))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if Path(tilelang.__file__).resolve().parent.parent != args.source.resolve():
        raise RuntimeError("Compiler source does not match the requested checkout")
    library = Path(tilelang._LIB_PATH).resolve()
    if not library.is_relative_to(args.source.resolve()):
        raise RuntimeError("Compiler library does not match the requested checkout")
    rows = []
    for width in (96, 100, 120, 127, 128, 256):
        fragment = make_fragment(width)
        for _ in range(2):
            fragment.condense_rep_var()
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter_ns()
            reduced = fragment.condense_rep_var()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        before = owner_set(fragment)
        after = owner_set(reduced)
        rows.append(
            {
                "width": width,
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "owners_equal": before == after,
                "before_count": len(before),
                "after_count": len(after),
            }
        )
    args.output.write_text(
        json.dumps({"compiler_sha256": hashlib.sha256(library.read_bytes()).hexdigest(), "cases": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
