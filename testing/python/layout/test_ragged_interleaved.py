"""End-to-end coverage for ragged interleaved fragment ownership."""

import re

import pytest
import tilelang
import tilelang.language as T
import torch


@pytest.mark.parametrize("width", [100, 120, 127])
@pytest.mark.parametrize("operation", ["sum", "max", "min"])
def test_interleaved_ragged_reduction(width, operation):
    layout = tilelang.Fragment(
        (1, width),
        forward_thread_fn=lambda i, j, rep: (j % 32) * 4 + rep,
        forward_index_fn=lambda i, j: j // 32,
        replicate=4,
    )

    @T.prim_func
    def main(X: T.Tensor((1, width), "float32"), Y: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((1, width), "float32")
            result = T.alloc_fragment((1,), "float32")
            T.annotate_layout({fragment: layout})
            T.copy(X, fragment)
            if operation == "sum":
                T.reduce_sum(fragment, result, dim=1)
            elif operation == "max":
                T.reduce_max(fragment, result, dim=1)
            else:
                T.reduce_min(fragment, result, dim=1)
            T.copy(result, Y)

    kernel = tilelang.compile(main, target="cuda", execution_backend="tvm_ffi")
    for value in (-2, 2):
        x = torch.full((1, width), value, dtype=torch.float32, device="cuda")
        y = torch.empty((1,), dtype=torch.float32, device="cuda")
        kernel(x, y)
        torch.cuda.synchronize()
        expected = x.sum(dim=1) if operation == "sum" else x[:, 0]
        torch.testing.assert_close(y, expected, rtol=0, atol=0)

    # Successful numerical output alone cannot reveal a register whose value
    # happened to be zero. Require the emitted local update to be predicated.
    source = kernel.get_kernel_source()
    loop = re.search(r"for \(int rv = 0; rv < 4; \+\+rv\) \{(.*?)\n  \}", source, re.DOTALL)
    assert loop is not None and "if (" in loop.group(1) and "fragment[rv]" in loop.group(1)


@pytest.mark.parametrize("width", [100, 127])
@pytest.mark.parametrize("interleaved", [False, True])
def test_explicit_rank_three_ragged_reduction(width, interleaved):
    def thread(i, k, j, rep):
        if interleaved:
            return (j % 32) * 4 + rep
        return rep * 32 + j % 32

    layout = tilelang.Fragment(
        (1, 1, width),
        forward_thread_fn=thread,
        forward_index_fn=lambda i, k, j: j // 32,
        replicate=4,
    )

    @T.prim_func
    def main(X: T.Tensor((1, 1, width), "float32"), Y: T.Tensor((1, 1), "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((1, 1, width), "float32")
            result = T.alloc_fragment((1, 1), "float32")
            T.annotate_layout({fragment: layout})
            T.copy(X, fragment)
            T.reduce_sum(fragment, result, dim=2)
            T.copy(result, Y)

    kernel = tilelang.compile(main, target="cuda", execution_backend="tvm_ffi")
    x = torch.arange(width, dtype=torch.float32, device="cuda").reshape(1, 1, width)
    y = torch.empty((1, 1), dtype=torch.float32, device="cuda")
    kernel(x, y)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x.sum(dim=2), rtol=0, atol=0)
    source = kernel.get_kernel_source()
    loop = re.search(r"for \(int rv = 0; rv < 4; \+\+rv\) \{(.*?)\n  \}", source, re.DOTALL)
    assert loop is not None and "if (" in loop.group(1) and "fragment[rv]" in loop.group(1)
