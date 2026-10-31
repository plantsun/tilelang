"""Compare the compiler's containment result with enumerated thread owners."""

import argparse
import ctypes
import json
from pathlib import Path

import tilelang
from check_mappings import evaluate
from run_experiments import source_metadata
from tilelang import tvm


def replicated(count, slot=0):
    return tilelang.Fragment(
        (1,),
        forward_thread_fn=lambda index, replica: replica,
        forward_index_fn=lambda index: slot,
        replicate=count,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = source_metadata(args.source.resolve())
    ctypes.CDLL(str(args.library.resolve()), mode=ctypes.RTLD_GLOBAL)
    prove_contains = tvm.get_global_func("tl.experiment.ProbeFragmentContains")
    guarded_inverse = tvm.get_global_func("tl.experiment.InverseWithPaddingGuard")
    rows = []
    for small_count, large_count in ((128, 96), (96, 128), (96, 96)):
        small, large = replicated(small_count), replicated(large_count)
        actual = bool(prove_contains(small, large, False))
        expected = set(range(small_count)).issubset(range(large_count))
        rows.append(
            {
                "small_extent": small_count,
                "large_extent": large_count,
                "enumerated_subset": expected,
                "compiler_result": actual,
                "mismatch": actual != expected,
            }
        )
        print(f"{small_count} in {large_count}: compiler={actual}, owners={expected}")
    index_result = bool(prove_contains(replicated(128, 0), replicated(128, 1), True))
    index_case = {
        "small_slot": 0,
        "large_slot": 1,
        "check_forward_index": True,
        "expected": False,
        "compiler_result": index_result,
    }
    print(f"different slots, index check enabled: compiler={index_result}, expected=False")
    inverse_rows = []
    for width in (96, 100, 120, 127):
        fragment = tilelang.Fragment(
            (1, width),
            forward_thread_fn=lambda i, j, rep: j % 32 + 32 * rep,
            forward_index_fn=lambda i, j: j // 32,
            replicate=4,
        )
        inverse = guarded_inverse(fragment)
        variables = list(inverse.get_forward_vars())
        outputs = list(inverse.index)
        valid = invalid = mismatches = 0
        for slot in range((width + 31) // 32):
            for thread in range(128):
                i, j, replica = evaluate(outputs, variables, [slot, thread])
                if i != 0 or not 0 <= replica < 4:
                    mismatches += 1
                elif j >= width:
                    invalid += 1
                else:
                    valid += 1
                    if j // 32 != slot or j % 32 + replica * 32 != thread:
                        mismatches += 1
        inverse_rows.append(
            {
                "width": width,
                "inverse": repr(inverse),
                "valid_physical_slots": valid,
                "guarded_padding_slots": invalid,
                "roundtrip_mismatches": mismatches,
            }
        )
        print(f"guarded inverse N={width}: valid={valid}, padding={invalid}, mismatches={mismatches}")
        if valid != width * 4 or mismatches:
            raise AssertionError("Guarded inverse failed exhaustive enumeration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "cases": rows,
                "index_case": index_case,
                "guarded_inverse": inverse_rows,
            },
            indent=2,
        )
        + "\n"
    )
    if rows[0]["enumerated_subset"] or not rows[1]["enumerated_subset"]:
        raise AssertionError("The independent owner-set control failed")


if __name__ == "__main__":
    main()
