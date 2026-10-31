# Fragment Replication 实验

实验研究 Fragment 的线程覆盖、复制关系、逆映射和不规则归约。结论、边界与上游问题见 [Replication 分析](replication_analysis.md)；结果使用 A100 验证。

`run_experiments.py` 比较隐式布局、显式 Warp 复制布局、填充切片和受保护的完整填充循环。`check_mappings.py` 枚举正逆映射，`contains_probe.cc` 与 `check_contains.py` 独立调用现有编译器检查函数。固定基线由 `versions.toml` 管理，结果见 `results/`。

## 环境与源码准备

需要 Linux、A100、CUDA Toolkit、Compute Sanitizer、支持 C++20 的主机编译器和 Python 3.10 或更新版本。依赖由 [requirements-fragment-replication.txt](requirements-fragment-replication.txt) 管理。使用独立 Python 环境，使 `PYTHONPATH` 指向待测源码；若环境已有其他源码的 editable 安装，应先消除导入冲突。实验会校验实际导入的源码与编译器动态库。

从仓库根目录设置实验目录：

```bash
EXPERIMENT_DIR="$PWD/docs/fix-ragged-fragment-reduce/fragment_replication"
SOURCE_DIR="${SOURCE_DIR:-$PWD/.fragment-replication/source}"
RAW_DIR="${RAW_DIR:-$PWD/.fragment-replication/output}"
mkdir -p "$(dirname "$SOURCE_DIR")" "$RAW_DIR"
python -m pip install -r "$EXPERIMENT_DIR/requirements-fragment-replication.txt"
```

若尚无固定基线检出，创建独立工作树并初始化依赖：

```bash
BASELINE_SHA="$(python -c 'import pathlib, sys; t = __import__("tomllib" if sys.version_info >= (3, 11) else "tomli"); print(t.loads((pathlib.Path(sys.argv[1]) / "versions.toml").read_text())["baseline"]["commit"])' "$EXPERIMENT_DIR")"
git fetch https://github.com/tile-ai/tilelang.git "$BASELINE_SHA"
git worktree add --detach "$SOURCE_DIR" "$BASELINE_SHA"
git -C "$SOURCE_DIR" submodule update --init --recursive
```

使用已准备好的基线检出时可跳过以上步骤。编译并设置运行环境：

```bash
export CXX="${CXX:-g++}"
cmake -S "$SOURCE_DIR" -B "$SOURCE_DIR/build" \
  -DUSE_CUDA=ON -DUSE_LLVM=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-DTVM_LOG_DEBUG
cmake --build "$SOURCE_DIR/build" --parallel 8
export PYTHONPATH="$SOURCE_DIR"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
```

`CXX` 应指向 CUDA Toolkit 支持且能够编译 C++20 的主机编译器。固定基线检查要求源码没有已跟踪文件的改动；修改编译器后，应在独立工作树验证并明确记录对应版本与改动。

## 运行

```bash
python "$EXPERIMENT_DIR/run_experiments.py" \
  --source "$SOURCE_DIR" --raw-dir "$RAW_DIR/implicit-raw" \
  --output-dir "$RAW_DIR/implicit"
python "$EXPERIMENT_DIR/run_experiments.py" \
  --source "$SOURCE_DIR" --raw-dir "$RAW_DIR/explicit-raw" \
  --output-dir "$RAW_DIR/explicit" --widths 96 100 120 127 \
  --layouts warp padded_warp padded_guard
python "$EXPERIMENT_DIR/check_mappings.py" \
  --source "$SOURCE_DIR" --output-dir "$RAW_DIR/mapping"
python "$EXPERIMENT_DIR/build_probe.py" \
  --source "$SOURCE_DIR" --output "$RAW_DIR/contains_probe.so"
python "$EXPERIMENT_DIR/check_contains.py" \
  --source "$SOURCE_DIR" --library "$RAW_DIR/contains_probe.so" \
  --output "$RAW_DIR/contains.json"
python "$EXPERIMENT_DIR/run_sanitizer.py" \
  --source "$SOURCE_DIR" --raw-dir "$RAW_DIR/sanitizer-raw" \
  --output "$RAW_DIR/sanitizer.json"
```

缩小的归约检查可作为修复前后的失败信号：

```bash
python "$EXPERIMENT_DIR/run_experiments.py" \
  --source "$SOURCE_DIR" --raw-dir "$RAW_DIR/reduce-raw" \
  --output-dir "$RAW_DIR/reduce" --widths 96 --operations reduce --require-pass
```

在固定基线上，该命令应返回非零状态，并在结果中记录布局不一致。运行全部实验时，失败案例作为结果记录，不使批量命令提前终止。
