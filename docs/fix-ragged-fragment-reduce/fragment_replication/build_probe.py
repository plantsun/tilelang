"""Build the standalone probe against an already-built TileLang checkout."""

import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    includes = (
        "src",
        "3rdparty/tvm/include",
        "3rdparty/tvm/src",
        "3rdparty/tvm/3rdparty/tvm-ffi/include",
        "3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include",
    )
    library_dir = source / "build/lib"
    command = [os.environ.get("CXX", "c++"), "-std=c++17", "-shared", "-fPIC", "-O2"]
    command += [f"-I{source / include}" for include in includes]
    command += [
        str(Path(__file__).with_name("contains_probe.cc")),
        f"-L{library_dir}",
        f"-Wl,-rpath,{library_dir}",
        "-ltilelang",
        "-ltvm_compiler",
        "-ltvm_runtime",
        "-ltvm_ffi",
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
