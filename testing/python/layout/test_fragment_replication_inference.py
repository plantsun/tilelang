"""End-to-end checks for automatically inferred replicated fragments."""

import pytest
import torch
import tilelang
import tilelang.language as T
from tilelang import tvm


def make_reduction(width, explicit=False, incompatible=False, threads=128):
    @T.prim_func
    def main(X: T.Tensor((1, width), "float32"), Y: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=threads):
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


@pytest.mark.parametrize("width", [100, 120, 127])
def test_implicit_ragged_reduction(width):
    kernel = tilelang.compile(make_reduction(width), target="cuda", execution_backend="tvm_ffi")
    generator = torch.Generator(device="cuda").manual_seed(2399)
    for x in (
        torch.arange(width, dtype=torch.float32, device="cuda").reshape(1, width),
        torch.ones((1, width), dtype=torch.float32, device="cuda"),
        torch.randn((1, width), dtype=torch.float32, device="cuda", generator=generator),
    ):
        y = torch.full((1,), float("nan"), device="cuda")
        kernel(x, y)
        torch.cuda.synchronize()
        torch.testing.assert_close(y, x.double().sum(dim=1).float(), rtol=0, atol=1e-4)


@pytest.mark.parametrize("width", [100, 120, 127])
@pytest.mark.parametrize(
    "threads",
    [
        pytest.param(
            64,
            marks=pytest.mark.xfail(strict=True, raises=tvm.error.InternalError, reason="64-thread ragged partition is not yet injective"),
        ),
        256,
    ],
)
def test_implicit_ragged_reduction_block_sizes(width, threads):
    kernel = tilelang.compile(make_reduction(width, threads=threads), target="cuda", execution_backend="tvm_ffi")
    x = torch.arange(width, dtype=torch.float32, device="cuda").reshape(1, width)
    y = torch.full((1,), float("nan"), device="cuda")
    kernel(x, y)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x.sum(dim=1), rtol=0, atol=0)


@pytest.mark.parametrize("width", [100, 120, 127])
@pytest.mark.parametrize("operation", ["sum", "max", "min"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "int32"])
def test_implicit_ragged_reduction_identities(width, operation, dtype):
    @T.prim_func
    def main(X: T.Tensor((1, width), dtype), Y: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((1, width), dtype)
            result = T.alloc_fragment((1,), dtype)
            T.copy(X, fragment)
            if operation == "sum":
                T.reduce_sum(fragment, result, dim=1)
            elif operation == "max":
                T.reduce_max(fragment, result, dim=1)
            else:
                T.reduce_min(fragment, result, dim=1)
            T.copy(result, Y)

    kernel = tilelang.compile(main, target="cuda", execution_backend="tvm_ffi")
    torch_dtype = getattr(torch, dtype)
    for value in (-2, 2):
        x = torch.full((1, width), value, dtype=torch_dtype, device="cuda")
        y = torch.empty((1,), dtype=torch_dtype, device="cuda")
        kernel(x, y)
        torch.cuda.synchronize()
        expected = x.sum(dim=1).to(torch_dtype) if operation == "sum" else x[:, 0]
        torch.testing.assert_close(y, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [100, 120, 127])
@pytest.mark.parametrize("clear", [False, True])
def test_implicit_ragged_column_reduction(width, clear):
    @T.prim_func
    def main(X: T.Tensor((width, 1), "float32"), Y: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((width, 1), "float32")
            result = T.alloc_fragment((1,), "float32")
            T.fill(result, 5.0)
            T.copy(X, fragment)
            T.reduce_sum(fragment, result, dim=0, clear=clear)
            T.copy(result, Y)

    kernel = tilelang.compile(main, target="cuda", execution_backend="tvm_ffi")
    x = torch.arange(width, dtype=torch.float32, device="cuda").reshape(width, 1)
    y = torch.empty((1,), dtype=torch.float32, device="cuda")
    kernel(x, y)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x.sum(dim=0) + (0 if clear else 5), rtol=0, atol=0)


@pytest.mark.parametrize("width", [100, 120, 127])
@pytest.mark.parametrize("padded", [False, True])
def test_guarded_ragged_fragment_copy(width, padded):
    storage = (width + 31) // 32 * 32 if padded else width
    layout = tilelang.Fragment(
        (1, storage),
        forward_thread_fn=lambda i, j, rep: j % 32 + rep * 32,
        forward_index_fn=lambda i, j: j // 32,
        replicate=4,
    )

    @T.prim_func
    def main(X: T.Tensor((1, width), "float32"), Y: T.Tensor((1, width), "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((1, storage), "float32")
            T.annotate_layout({fragment: layout})
            if padded:
                T.clear(fragment)
                T.copy(X, fragment[0:1, 0:width])
                T.copy(fragment[0:1, 0:width], Y)
            else:
                T.copy(X, fragment)
                T.copy(fragment, Y)

    kernel = tilelang.compile(main, target="cuda", execution_backend="tvm_ffi")
    generator = torch.Generator(device="cuda").manual_seed(2399)
    for x in (
        torch.arange(width, dtype=torch.float32, device="cuda").reshape(1, width),
        torch.ones((1, width), dtype=torch.float32, device="cuda"),
        torch.randn((1, width), dtype=torch.float32, device="cuda", generator=generator),
    ):
        y = torch.full_like(x, float("nan"))
        kernel(x, y)
        torch.cuda.synchronize()
        torch.testing.assert_close(y, x, rtol=0, atol=0)


@pytest.mark.parametrize(
    "width",
    [
        96,
        128,
        100,
        120,
        127,
    ],
)
def test_reduction_replica_condensation_preserves_owners(width):
    fragment = tilelang.Fragment(
        (1,),
        forward_thread_fn=lambda i, rep: (rep % width) % 32 + (rep // width) * 32,
        forward_index_fn=lambda i: 0,
        replicate=4 * width,
    )

    def owner_set(layout):
        variables = list(layout.get_forward_vars())
        analyzer = tvm.arith.Analyzer()
        result = set()
        for rep in range(int(layout.replicate_size)):
            values = [rep, 0]
            mapping = {v: tvm.tirx.IntImm("int32", n) for v, n in zip(variables, values, strict=True)}
            result.add(int(analyzer.simplify(tvm.tirx.stmt_functor.substitute(layout.thread, mapping))))
        return result

    assert owner_set(fragment) == owner_set(fragment.condense_rep_var())


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


@pytest.mark.parametrize("elementwise", [False, True])
@pytest.mark.xfail(strict=True, raises=tvm.error.InternalError, reason="Two-dimensional ragged layout inference is not yet supported")
def test_implicit_ragged_matrix_copy(elementwise):
    @T.prim_func
    def main(X: T.Tensor((2, 100), "float32"), Y: T.Tensor((2, 100), "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((2, 100), "float32")
            T.copy(X, fragment)
            if elementwise:
                for i, j in T.Parallel(2, 100):
                    fragment[i, j] = fragment[i, j] * 2.0
            T.copy(fragment, Y)

    kernel = tilelang.compile(main, target="cuda", execution_backend="tvm_ffi")
    x = torch.arange(200, dtype=torch.float32, device="cuda").reshape(2, 100)
    y = torch.full_like(x, float("nan"))
    kernel(x, y)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x * (2 if elementwise else 1), rtol=0, atol=0)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.xfail(strict=True, raises=tvm.error.InternalError, reason="Two-dimensional ragged reduction is not yet supported")
def test_implicit_ragged_matrix_reduction(axis):
    output_shape = (100,) if axis == 0 else (2,)

    @T.prim_func
    def main(X: T.Tensor((2, 100), "float32"), Y: T.Tensor(output_shape, "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((2, 100), "float32")
            result = T.alloc_fragment(output_shape, "float32")
            T.copy(X, fragment)
            T.reduce_sum(fragment, result, dim=axis)
            T.copy(result, Y)

    kernel = tilelang.compile(main, target="cuda", execution_backend="tvm_ffi")
    x = torch.arange(200, dtype=torch.float32, device="cuda").reshape(2, 100)
    y = torch.full(output_shape, float("nan"), device="cuda")
    kernel(x, y)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x.sum(dim=axis), rtol=0, atol=0)


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
