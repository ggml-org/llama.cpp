# header-only: interface target exposing the vendor/ root so consumers
# can include via <nlohmann/json.hpp>
add_library(nlohmann INTERFACE)
add_library(vendor::nlohmann ALIAS nlohmann)

target_include_directories(nlohmann INTERFACE ..)
