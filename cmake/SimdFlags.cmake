# Per-translation-unit instruction-set flags.
#
# Runtime dispatch model (docs/DESIGN.md §10.4): generic code is compiled for the baseline ISA;
# only dedicated kernel translation units receive AVX2/FMA flags. Phase 5 adds the AVX2 object
# library; Phase 1 only needs the floating-point contraction controls below.

# Flags that make a scalar reference translation unit a stable numerical oracle:
# no FMA contraction, no reassociation.
function(vf_set_strict_fp_source source)
  if(MSVC)
    # /fp:precise is the default and does not contract since VS 2022; nothing to add.
    return()
  endif()
  set_property(SOURCE ${source} APPEND PROPERTY COMPILE_OPTIONS -ffp-contract=off)
endfunction()

# Flags for the "compiler auto-vectorization" comparison kernels: allow the compiler to
# vectorize float reductions (reordering sums) without enabling -ffast-math globally.
function(vf_set_autovec_source source)
  if(MSVC)
    set_property(SOURCE ${source} APPEND PROPERTY COMPILE_OPTIONS /fp:fast)
  else()
    set_property(SOURCE ${source} APPEND PROPERTY COMPILE_OPTIONS -fopenmp-simd -ffp-contract=off)
    # Clang warns when a requested simd transformation is not applied (e.g. at -O0/-O1).
    set_property(SOURCE ${source} APPEND PROPERTY COMPILE_OPTIONS
                 $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-pass-failed>)
    set_property(SOURCE ${source} APPEND PROPERTY COMPILE_DEFINITIONS VF_USE_OMP_SIMD=1)
  endif()
endfunction()

function(vf_avx2_compile_options out_var)
  if(MSVC)
    set(${out_var} /arch:AVX2 PARENT_SCOPE)
  else()
    set(${out_var} -mavx2 -mfma PARENT_SCOPE)
  endif()
endfunction()
