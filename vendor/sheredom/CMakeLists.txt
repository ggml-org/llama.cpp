# header-only: interface target exposing the vendor/ root so consumers
# can include via <sheredom/subprocess.h>
add_library(sheredom INTERFACE)
add_library(vendor::sheredom ALIAS sheredom)

target_include_directories(sheredom INTERFACE ..)
