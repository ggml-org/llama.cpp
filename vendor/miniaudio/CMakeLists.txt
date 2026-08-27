# header-only: interface target exposing the vendor/ root so consumers
# can include via <miniaudio/miniaudio.h>
add_library(miniaudio INTERFACE)
add_library(vendor::miniaudio ALIAS miniaudio)

target_include_directories(miniaudio INTERFACE ..)
