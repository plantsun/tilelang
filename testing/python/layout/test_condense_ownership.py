"""Replica condensation must preserve independently enumerated owners."""

import pytest
import tilelang
from tilelang import tvm


def owners(fragment, logical):
    variables = list(fragment.get_forward_vars())
    analyzer = tvm.arith.Analyzer()
    result = set()
    for rep in range(int(fragment.replicate_size)):
        mapping = {v: tvm.tirx.IntImm("int32", n) for v, n in zip(variables, (rep, logical), strict=True)}
        result.add(int(analyzer.simplify(tvm.tirx.stmt_functor.substitute(fragment.thread, mapping))))
    return result


@pytest.mark.parametrize("mapping", ["separated", "strided", "irregular"])
@pytest.mark.parametrize("logical", [0, 1])
def test_condense_preserves_static_thread_sets(mapping, logical):
    def forward(i, rep):
        lane = (rep % 100) % 32
        group = rep // 100
        if mapping == "separated":
            return i * 128 + group * 32 + lane
        if mapping == "strided":
            return i * 256 + 2 * (group * 32 + lane)
        return i * 256 + group * 64 + lane

    original = tilelang.Fragment((2,), forward_thread_fn=forward, forward_index_fn=lambda i: i, replicate=400)
    condensed = original.condense_rep_var()
    assert owners(original, logical) == owners(condensed, logical)
    if mapping != "irregular":
        assert int(condensed.replicate_size) == 128


def test_condense_preserves_unverified_large_domain():
    original = tilelang.Fragment(
        (2,),
        forward_thread_fn=lambda i, rep: i * 4096 + (rep % 100) % 32 + (rep // 100) * 32,
        forward_index_fn=lambda i: i,
        replicate=8193,
    )
    condensed = original.condense_rep_var()
    assert int(condensed.replicate_size) == int(original.replicate_size)
    for logical in (0, 1):
        assert owners(original, logical) == owners(condensed, logical)


@pytest.mark.parametrize("pattern", ["cyclic", "blocked"])
def test_condense_proves_large_periodic_domains(pattern):
    if pattern == "cyclic":
        original = tilelang.Fragment(
            (2,), forward_thread_fn=lambda i, rep: i * 128 + rep % 128, forward_index_fn=lambda i: i, replicate=8193
        )
        expected = 128
        coordinates = (0, 1)
    else:
        original = tilelang.Fragment((16385,), forward_thread_fn=lambda i, rep: rep // 8, forward_index_fn=lambda i: i, replicate=512)
        expected = 64
        coordinates = (0, 16384)
    condensed = original.condense_rep_var()
    assert int(condensed.replicate_size) == expected
    for logical in coordinates:
        assert owners(original, logical) == owners(condensed, logical)
