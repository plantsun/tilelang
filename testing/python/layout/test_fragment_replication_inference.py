"""End-to-end checks for automatically inferred replicated fragments."""

import pytest
import torch
import tilelang
import tilelang.language as T
from tilelang import tvm


def make_reduction(width, explicit=False, incompatible=False):
    @T.prim_func
    def main(X: T.Tensor((1, width), "float32"), Y: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((1, width), "float32")
            result = T.alloc_fragment((1,), "float32")
            if explicit:
                T.annotate_layout(
                    {
                        fragment: tilelang.Fragment(
                            (1, width),
                            forward_thread_fn=lambda i, j, rep: j % 32 + rep * 32,
                            forward_index_fn=lambda i, j: j // 32,
                            replicate=4,
                        )
                    }
                )
            if incompatible:
                T.annotate_layout(
                    {
                        fragment: tilelang.Fragment(
                            (1, width),
                            forward_thread_fn=lambda i, j: j,
                            forward_index_fn=lambda i, j: 0,
                        )
                    }
                )
            T.copy(X, fragment)
            T.reduce_sum(fragment, result, dim=1)
            T.copy(result, Y)

    return main


@pytest.mark.parametrize("width", [64, 96, 128, 256])
@pytest.mark.parametrize("explicit", [False, True])
def test_replicated_reduction(width, explicit):
    kernel = tilelang.compile(make_reduction(width, explicit), target="cuda", execution_backend="tvm_ffi")
    generator = torch.Generator(device="cuda").manual_seed(2399)
    inputs = [
        torch.arange(width, dtype=torch.float32, device="cuda").reshape(1, width),
        torch.ones((1, width), dtype=torch.float32, device="cuda"),
        torch.randn((1, width), dtype=torch.float32, device="cuda", generator=generator),
    ]
    for x in inputs:
        y = torch.full((1,), float("nan"), device="cuda")
        kernel(x, y)
        torch.cuda.synchronize()
        torch.testing.assert_close(y, x.double().sum(dim=1).float(), rtol=0, atol=1e-4)


def test_incompatible_explicit_layout_is_not_replaced():
    with pytest.raises(tvm.error.InternalError, match="no available layout|Layout may conflict|Inconsistent layouts"):
        tilelang.compile(make_reduction(96, incompatible=True), target="cuda", execution_backend="tvm_ffi")


def make_copy(width, elementwise):
    @T.prim_func
    def main(X: T.Tensor((1, width), "float32"), Y: T.Tensor((1, width), "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((1, width), "float32")
            T.copy(X, fragment)
            if elementwise:
                for i, j in T.Parallel(1, width):
                    fragment[i, j] = fragment[i, j] * 2.0
            T.copy(fragment, Y)

    return main


@pytest.mark.parametrize("width", [64, 96, 128, 256])
@pytest.mark.parametrize("elementwise", [False, True])
def test_implicit_copy_and_elementwise(width, elementwise):
    kernel = tilelang.compile(make_copy(width, elementwise), target="cuda", execution_backend="tvm_ffi")
    x = torch.arange(width, dtype=torch.float32, device="cuda").reshape(1, width)
    y = torch.full_like(x, float("nan"))
    kernel(x, y)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x * (2 if elementwise else 1), rtol=0, atol=0)


@pytest.mark.parametrize("rows,width,axis", [(3, 32, 1), (2, 96, 0), (2, 96, 1)])
def test_implicit_multidimensional_reduction(rows, width, axis):
    output_shape = (width,) if axis == 0 else (rows,)

    @T.prim_func
    def main(X: T.Tensor((rows, width), "float32"), Y: T.Tensor(output_shape, "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((rows, width), "float32")
            result = T.alloc_fragment(output_shape, "float32")
            T.copy(X, fragment)
            T.reduce_sum(fragment, result, dim=axis)
            T.copy(result, Y)

    kernel = tilelang.compile(main, target="cuda", execution_backend="tvm_ffi")
    generator = torch.Generator(device="cuda").manual_seed(2399)
    for x in (
        torch.arange(rows * width, dtype=torch.float32, device="cuda").reshape(rows, width),
        torch.ones((rows, width), dtype=torch.float32, device="cuda"),
        torch.randn((rows, width), dtype=torch.float32, device="cuda", generator=generator),
    ):
        y = torch.empty(output_shape, device="cuda")
        kernel(x, y)
        torch.cuda.synchronize()
        torch.testing.assert_close(y, x.double().sum(dim=axis).float(), rtol=0, atol=1e-4)
