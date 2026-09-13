# Per-target warning and conformance settings. Applied only to VectorForge's own targets;
# third-party code is consumed as SYSTEM so its warnings never break our build.

function(vf_set_project_options target)
  if(MSVC)
    set(vf_warnings
        /W4
        /permissive-
        /Zc:__cplusplus
        /Zc:preprocessor
        /utf-8
        /w14242 # conversion, possible loss of data
        /w14254 # larger bit field to smaller, possible loss of data
        /w14263 # member function does not override any base class virtual member function
        /w14265 # class has virtual functions, but destructor is not virtual
        /w14287 # unsigned/negative constant mismatch
        /w14296 # expression is always true/false
        /w14311 # pointer truncation
        /w14545 /w14546 /w14547 /w14549 /w14555 # suspicious comma/operator expressions
        /w14619 # unknown warning number in pragma
        /w14640 # thread-unsafe static member initialization
        /w14826 # sign-extending conversion
        /w14905 /w14906 # string literal casts
        /w14928 # illegal copy-initialization
    )
    if(VF_WARNINGS_AS_ERRORS)
      list(APPEND vf_warnings /WX)
    endif()
  else()
    set(vf_warnings
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wold-style-cast
        -Wnon-virtual-dtor
        -Wcast-align
        -Woverloaded-virtual
        -Wnull-dereference
        -Wdouble-promotion
        -Wimplicit-fallthrough
        -Wformat=2
        -Wundef)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      list(APPEND vf_warnings -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wlogical-op)
    endif()
    if(VF_WARNINGS_AS_ERRORS)
      list(APPEND vf_warnings -Werror)
    endif()
  endif()
  target_compile_options(${target} PRIVATE ${vf_warnings})
  target_compile_features(${target} PUBLIC cxx_std_20)
endfunction()
