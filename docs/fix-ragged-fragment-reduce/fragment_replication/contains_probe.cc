// Expose the existing containment function for an independent experiment.
// This shared library does not replace or modify any compiler implementation.
#include "layout/utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/expr.h>

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def(
      "tl.experiment.ProbeFragmentContains",
      [](tvm::tl::Fragment small, tvm::tl::Fragment large, bool check_index) {
        // Match the 128-thread launch range attached during layout inference.
        small = small->BindThreadRange(tvm::Range(0, 128));
        large = large->BindThreadRange(tvm::Range(0, 128));
        tvm::arith::Analyzer analyzer;
        tvm::ffi::Array<tvm::PrimExpr> indices{
            tvm::IntImm(tvm::DataType::Int(32), 0)};
        return tvm::tl::ProveFragmentContains(small, large, indices, indices,
                                              analyzer, check_index);
      })
      .def("tl.experiment.InverseWithPaddingGuard",
           [](tvm::tl::Fragment fragment) {
             return fragment->InverseWithLevel(true).first;
           });
}
