# Global sanitizer configuration (VF_SANITIZE). Applied with add_compile_options so that
# dependencies built from source (GoogleTest, Google Benchmark) are instrumented as well:
# TSan requires it for correctness and MSVC ASan requires it to avoid annotation mismatches.

function(vf_configure_sanitizers)
  if(NOT VF_SANITIZE)
    return()
  endif()

  if(MSVC)
    foreach(s IN LISTS VF_SANITIZE)
      if(NOT s STREQUAL "address")
        message(FATAL_ERROR "MSVC supports only VF_SANITIZE=address (got '${s}')")
      endif()
    endforeach()
    add_compile_options(/fsanitize=address)
    add_link_options(/INCREMENTAL:NO)
    # Run-time checks (/RTC1) are incompatible with /fsanitize=address.
    foreach(flag_var IN ITEMS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS)
      string(REGEX REPLACE "/RTC[1csu]*" "" ${flag_var} "${${flag_var}}")
      set(${flag_var} "${${flag_var}}" PARENT_SCOPE)
    endforeach()
    return()
  endif()

  if("thread" IN_LIST VF_SANITIZE AND ("address" IN_LIST VF_SANITIZE))
    message(FATAL_ERROR "ThreadSanitizer cannot be combined with AddressSanitizer")
  endif()

  list(JOIN VF_SANITIZE "," vf_san_list)
  set(vf_san_flags -fsanitize=${vf_san_list} -fno-omit-frame-pointer -fno-sanitize-recover=all -g)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND vf_san_flags -O1)
  endif()
  add_compile_options(${vf_san_flags})
  add_link_options(-fsanitize=${vf_san_list})
endfunction()
