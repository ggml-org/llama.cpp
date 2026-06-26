# infcore security — корпоративная лицензия (см. ../../LICENSE для движка; infcore — отдельная лицензия).
add_library(infcore_security STATIC
    authn/authn.cpp rbac/rbac.cpp audit/audit.cpp secrets/secrets.cpp)
target_include_directories(infcore_security PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
