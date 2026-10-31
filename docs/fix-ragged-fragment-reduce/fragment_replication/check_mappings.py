"""Enumerate logical/replica/thread/slot mappings using the real Fragment API."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import tilelang
from run_experiments import source_metadata
from tilelang import tvm


def evaluate(expressions, variables, values):
    substitutions = {var: tvm.tirx.IntImm("int32", value) for var, value in zip(variables, values, strict=True)}
    analyzer = tvm.arith.Analyzer()
    return [int(analyzer.simplify(tvm.tirx.stmt_functor.substitute(expression, substitutions))) for expression in expressions]


def describe(fragment, logical_width, candidate, csv_writer):
    variables = list(fragment.get_forward_vars())
    logical_variables = variables[-2:]
    replica_variables = variables[:-2]
    replication = int(fragment.replicate_size)
    thread_extent = int(fragment.get_thread_size())
    storage_width = int(fragment.get_input_shape()[-1])
    slots = int(fragment.get_output_shape()[0])
    placements = []
    for j in range(storage_width):
        for replica in range(replication):
            values = ([replica] if replica_variables else []) + [0, j]
            thread = evaluate([fragment.thread], variables, values)[0]
            slot = evaluate(list(fragment.index), logical_variables, [0, j])[0]
            placements.append((j, replica, thread, slot))
            csv_writer.writerow(
                [
                    logical_width,
                    candidate,
                    j,
                    replica,
                    thread,
                    slot,
                    int(j < logical_width),
                ]
            )
    valid = [placement for placement in placements if placement[0] < logical_width]
    physical = Counter((thread, slot) for _, _, thread, slot in placements)
    per_thread = Counter(thread for _, _, thread, _ in valid)
    result = {
        "width": logical_width,
        "candidate": candidate,
        "fragment_width": storage_width,
        "replicate_extent": replication,
        "thread_extent": thread_extent,
        "slots_per_thread": slots,
        "forward_thread": str(fragment.thread),
        "forward_index": [str(index) for index in fragment.index],
        "fragment": repr(fragment),
        "coverage": len({j for j, _, _, _ in valid}) == logical_width,
        "threads_in_cta": all(0 <= thread < 128 for _, _, thread, _ in placements),
        "no_physical_collisions": all(count == 1 for count in physical.values()),
        "valid_elements_per_thread_histogram": dict(sorted(Counter(per_thread.get(thread, 0) for thread in range(128)).items())),
        "storage_density": len(physical) / (slots * thread_extent),
        "padding_slots_in_cta": slots * 128 - len(valid),
    }
    try:
        inverse = fragment.inverse()
        result["inverse"] = repr(inverse)
        inverse_vars = list(inverse.get_forward_vars())
        inverse_expressions = list(inverse.index)
        mismatches = []
        for j, replica, thread, slot in placements:
            values = evaluate(inverse_expressions, inverse_vars, [slot, thread])
            if values != [0, j, replica]:
                mismatches.append({"input": [j, replica, thread, slot], "inverse": values})
        result["inverse_roundtrip"] = not mismatches
        result["inverse_mismatches"] = mismatches[:4]
    except Exception as error:  # noqa: BLE001 - report unsupported inverse mappings
        result["inverse_roundtrip"] = False
        signal = [
            line.strip()
            for line in str(error).splitlines()
            if any(word in line.lower() for word in ["error", "bijective", "surjective", "iter", "check failed"])
        ]
        result["inverse_error"] = signal[-5:]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = source_metadata(args.source.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with (args.output_dir / "mappings.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "width",
                "candidate",
                "logical_index",
                "replica",
                "thread",
                "slot",
                "valid_logical_index",
            ]
        )
        for width in (96, 100, 120, 127):
            base = tilelang.Fragment(
                (1, width),
                forward_thread_fn=lambda i, j: j,
                forward_index_fn=lambda i, j: 0,
            )
            warp = tilelang.Fragment(
                (1, width),
                forward_thread_fn=lambda i, j, rep: j % 32 + 32 * rep,
                forward_index_fn=lambda i, j: j // 32,
                replicate=4,
            )
            padded = (width + 31) // 32 * 32
            padded_warp = tilelang.Fragment(
                (1, padded),
                forward_thread_fn=lambda i, j, rep: j % 32 + 32 * rep,
                forward_index_fn=lambda i, j: j // 32,
                replicate=4,
            )
            candidates = {
                "linear_floor": base.replicate(128 // width),
                "linear_ceil": base.replicate((128 + width - 1) // width),
                "warp": warp,
                "padded_warp": padded_warp,
            }
            for candidate, fragment in candidates.items():
                result = describe(fragment, width, candidate, writer)
                results.append(result)
                print(
                    f"N={width} {candidate}: threads={result['thread_extent']}, replication={result['replicate_extent']}, inverse={result['inverse_roundtrip']}",
                    flush=True,
                )
    (args.output_dir / "mappings.json").write_text(json.dumps({"metadata": metadata, "candidates": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
