# Reject removed targets before CUDA compiler identification obscures the toolkit remedy.
function(ggml_cuda_check_toolkit_architectures toolkit_version architectures)
    if (toolkit_version VERSION_LESS "13.0")
        return()
    endif()
    foreach(architecture IN LISTS architectures)
        if (architecture MATCHES "^([0-9]+)(-real|-virtual)?$")
            if (CMAKE_MATCH_1 LESS 75)
                message(FATAL_ERROR
                    "CUDA Toolkit ${toolkit_version} cannot target architecture ${architecture}. "
                    "For Maxwell, Pascal or Volta use CUDA 12.9 or earlier in a separate build directory, "
                    "or select a supported architecture for your GPU. Do not change your driver to fix this compiler limitation.")
            endif()
        endif()
    endforeach()
endfunction()
