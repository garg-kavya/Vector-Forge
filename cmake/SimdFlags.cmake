# Per-translation-unit instruction-set flags.
#
# Runtime dispatch model (docs/DESIGN.md §10.4): generic code is compiled for the baseline ISA;
# only dedicated kernel translation units receive AVX2/FMA flags (docs/simd.md).

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
    # /fp:precise (default) does not contract a*b+c into FMA since VS 2022.
    set(flags /arch:AVX2)
  else()
    # No implicit FMA contraction of the scalar tails: the kernels' reduction order is explicit.
    set(flags -mavx2 -mfma -ffp-contract=off)
    if(MINGW)
      # MinGW GCC does not align the stack to 32 bytes on Windows but may spill YMM values with
      # aligned moves (GCC bug 54412); make the assembler emit unaligned moves instead.
      list(APPEND flags -Wa,-muse-unaligned-vector-move)
    endif()
  endif()
  set(${out_var} ${flags} PARENT_SCOPE)
endfunction()

# Sets `out_var` to TRUE when AVX2 kernels should be compiled: VF_ENABLE_AVX2 and an x86-64 target.
function(vf_avx2_kernels_enabled out_var)
  set(enabled FALSE)
  if(VF_ENABLE_AVX2)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64|amd64|x64|X64)$")
      set(enabled TRUE)
    else()
      message(STATUS "VF_ENABLE_AVX2: target processor '${CMAKE_SYSTEM_PROCESSOR}' is not x86-64; AVX2 kernels skipped")
    endif()
  endif()
  set(${out_var} ${enabled} PARENT_SCOPE)
endfunction()

