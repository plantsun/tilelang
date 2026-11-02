# Fragment Ownership and Guarded Reduction: Review Notes

## Problem and patch boundary

Fully replicated Fragment containment previously accepted some incompatible thread sets and could skip the requested physical-slot check. The standalone containment patch checks physical slots when requested and proves that each small-fragment owner lies within the large fragment's replicated thread range. Nonzero thread offsets, noncontiguous source mappings, bounded thread ranges and symbolic coordinates are covered by regression tests. An unprovable containment is conservatively rejected.

The standalone patch does not make ragged reductions compile. Its test-only bridge calls the actual compiler implementation without adding a production API. The bridge currently depends on a Linux C++ toolchain; an upstream-native C++ test is an alternative worth discussing.

## Follow-on design

Warp replication candidates are explored only after existing free-inference attempts fail. Static nonintegral candidates are restricted to CUDA, at most two dimensions, and a single non-unit logical dimension. Explicit layout constraints are not replaced. A padded loop candidate is accepted only after injectivity and valid-domain inverse round-trip have been proved.

Replica condensation first tries symbolic cyclic or blocked equivalence after normalizing the replica coordinate. Otherwise, it compares original and compressed thread-owner sets on bounded static domains. A changed set is replaced only when a compact arithmetic progression is proved for every logical coordinate. If equivalence or a compact representation cannot be proved, the original Fragment is retained. This preserves ownership but may cause a later inference attempt to reject a layout that previously compiled.

Reduction ownership records both local physical indices and an optional valid-domain predicate. For static scalar-output layouts with one physical slot dimension and a nondivisible thread-owned factor, the predicate is derived from a single thread-owned digit's extent and stride and the logical reduction extent. Within that entry condition, an unprovable thread or local-slot mapping is rejected. This covers both consecutive and interleaved replicas without matching a particular Sum operation. Invalid local slots are skipped, preserving the operator's identity and preventing repeated accumulation. Vector packing is disabled when the predicate is required; other explicit layouts require separate validation.

## Validation and limitations

The layout suite collected 421 cases: 408 passed, four skipped, seven expected failures and two failures already present on the baseline. The expected failures are three ragged reductions with 64 threads, two 2×100 Copy/Elementwise cases and two 2×100 reductions. The two baseline failures require PDL unavailable on A100. The reduction suite passed 142 cases with 13 skips.

FP32, FP16 and INT32 Sum/Min/Max, positive and negative inputs, explicit interleaved replicas, scalar-output higher-rank layouts, overwrite and accumulation, and the original implicit layouts have been checked. Memcheck and Racecheck reported no errors for the tested cases. Numerical agreement alone is not sufficient to detect an uninitialized register, so the interleaved regression also checks that the emitted local update is guarded.

The automatic candidate does not yet solve the 64-thread cases or 2×100 layouts. Symbolic or larger replica domains can retain their original non-injective representation when equivalence cannot be proven. Multi-digit reduction ownership, vectorized guarded reductions, other backends and more complex thread ranges require further work.

## Review sequence

Review the standalone containment correctness patch first. Discuss the representation, proof budget, possible conservative rejections and test integration before deciding how to split or combine the layout and reduction changes. The measurements are microbenchmarks, not evidence of an end-to-end model improvement or upstream acceptance.
