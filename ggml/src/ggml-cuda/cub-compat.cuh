#pragma once
#if defined(GGML_USE_HIP)
#    include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#    include <cub/cub.cuh>
#endif
