"""Run the implicit fragment-layout matrix and emit structured results."""

import argparse
import json
from pathlib import Path

import tilelang
import torch

from test_fragment_replication_inference import make_copy, make_reduction


def run(width, operation):
    row = {"width": width, "operation": operation}
    try:
        func = make_reduction(width) if operation == "reduce" else make_copy(width, operation == "elementwise")
        kernel = tilelang.compile(func, target="cuda", execution_backend="tvm_ffi")
    except Exception as error:  # Preserve unsupported compilation as a measured outcome.
        row.update(status="compile_failure", error_type=type(error).__name__)
        return row
    try:
        generator = torch.Generator(device="cuda").manual_seed(2399)
        patterns = {
            "arange": torch.arange(width, dtype=torch.float32, device="cuda").reshape(1, width),
            "ones": torch.ones((1, width), dtype=torch.float32, device="cuda"),
            "random": torch.randn((1, width), dtype=torch.float32, device="cuda", generator=generator),
        }
        errors = {}
        for name, x in patterns.items():
            expected = x.double().sum(dim=1).float() if operation == "reduce" else x * (2 if operation == "elementwise" else 1)
            y = torch.full_like(expected, float("nan"))
            for _ in range(3):
                y.fill_(float("nan"))
                kernel(x, y)
                torch.cuda.synchronize()
                torch.testing.assert_close(y, expected, rtol=0, atol=1e-4)
            errors[name] = float((y - expected).abs().max())
        row.update(status="pass", max_absolute_error=errors)
    except Exception as error:  # Preserve runtime and correctness failures separately.
        row.update(status="execution_failure", error_type=type(error).__name__)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 96, 100, 120, 127, 128, 256])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [run(width, op) for width in args.widths for op in ("copy", "elementwise", "reduce")]
    device = torch.cuda.get_device_name()
    result = json.dumps({"device": "A100" if "A100" in device else device, "cases": rows}, indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
