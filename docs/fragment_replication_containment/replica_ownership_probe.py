"""Compare actual fragment owner sets before and after replica condensation."""

import argparse
import json

import tilelang
from tilelang import tvm


def owners(fragment):
    variables = list(fragment.get_forward_vars())
    analyzer = tvm.arith.Analyzer()
    result = set()
    for replica in range(int(fragment.replicate_size)):
        values = [replica, 0] if len(variables) == 2 else [0]
        mapping = {v: tvm.tirx.IntImm("int32", x) for v, x in zip(variables, values, strict=True)}
        thread = analyzer.simplify(tvm.tirx.stmt_functor.substitute(fragment.thread, mapping))
        result.add(int(thread))
    return sorted(result)


def make_original(width):
    return tilelang.Fragment(
        (1,),
        forward_thread_fn=lambda i, replica: (replica % width) % 32 + (replica // width) * 32,
        forward_index_fn=lambda i: 0,
        replicate=4 * width,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=[96, 100, 120, 127, 128])
    args = parser.parse_args()
    records = []
    for width in args.widths:
        original = make_original(width)
        condensed = original.condense_rep_var()
        before, after = owners(original), owners(condensed)
        records.append(
            {
                "width": width,
                "original_replication": int(original.replicate_size),
                "condensed_replication": int(condensed.replicate_size),
                "original_thread_mapping": str(original.thread),
                "condensed_thread_mapping": str(condensed.thread),
                "original_owners": before,
                "condensed_owners": after,
                "owners_equal": before == after,
            }
        )
    print(json.dumps({"cases": records}, indent=2))


if __name__ == "__main__":
    main()
