"""Run scoped Compute Sanitizer checks on the guarded padding prototype."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=100)
    args = parser.parse_args()
    source = args.source.resolve()
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(source),
        TILELANG_DISABLE_CACHE="1",
        TVM_FFI_DISABLE_TORCH_C_DLPACK="1",
    )
    script = Path(__file__).with_name("run_experiments.py")
    result = {"width": args.width, "layout": "padded_guard", "cases": []}
    for operation in ("copy", "reduce"):
        for tool in ("memcheck", "racecheck"):
            raw = args.raw_dir.resolve() / f"{operation}_{tool}"
            raw.mkdir(parents=True, exist_ok=True)
            case_file, log_file = raw / "case.json", raw / "sanitizer.log"
            case_file.unlink(missing_ok=True)
            log_file.unlink(missing_ok=True)
            command = [
                "compute-sanitizer",
                "--tool",
                tool,
                "--error-exitcode",
                "91",
                "--log-file",
                str(log_file),
                sys.executable,
                str(script),
                "--source",
                str(source),
                "--raw-dir",
                str(raw),
                "--worker",
                "--width",
                str(args.width),
                "--kind",
                operation,
                "--layout",
                "padded_guard",
                "--raw",
                str(raw),
                "--output",
                str(case_file),
            ]
            with (raw / "run.log").open("w") as log:
                process = subprocess.run(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                    check=False,
                )
            if process.returncode or not case_file.exists() or not log_file.exists():
                raise RuntimeError(f"{operation}/{tool} did not complete successfully")
            case = json.loads(case_file.read_text())
            if case["status"] != "passed":
                raise RuntimeError(f"{operation}/{tool} failed its numerical checks")
            text = log_file.read_text()
            if tool == "memcheck":
                match = re.search(r"ERROR SUMMARY: (\d+) errors", text)
                metrics = {"errors": int(match.group(1))} if match else None
            else:
                match = re.search(
                    r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)",
                    text,
                )
                metrics = (
                    {
                        "hazards": int(match.group(1)),
                        "errors": int(match.group(2)),
                        "warnings": int(match.group(3)),
                    }
                    if match
                    else None
                )
            if metrics is None or any(metrics.values()):
                raise RuntimeError(f"{operation}/{tool} reported an error or has no readable summary")
            if "metadata" in result and result["metadata"] != case["metadata"]:
                raise RuntimeError("Sanitizer cases used different compiler builds")
            result["metadata"] = case["metadata"]
            result["gpu"] = case["gpu"]
            result["cases"].append({"operation": operation, "tool": tool, **metrics})
            print(f"{operation}/{tool}: passed", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
