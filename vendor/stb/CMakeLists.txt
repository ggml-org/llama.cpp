# header-only: interface target exposing the vendor/ root so consumers
# can include via <stb/stb_image.h>
add_library(stb INTERFACE)
add_library(vendor::stb ALIAS stb)

target_include_directories(stb INTERFACE ..)
