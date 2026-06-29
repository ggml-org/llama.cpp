cmake_minimum_required(VERSION 3.21)
project(infcore VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# Встраиваем движок llama.cpp (родительский каталог) БЕЗ изменений его файлов.
# llama.cpp штатно поддерживает использование через add_subdirectory.
# Флаги бэкендов/состава задаются профилем: cmake -C infcore/cmake/profile-rf.cmake
# -----------------------------------------------------------------------------
set(INFCORE_LLAMA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH
    "Корень дерева llama.cpp (форк)")

# При встраивании через add_subdirectory LLAMA_STANDALONE=OFF, поэтому
# common/server по умолчанию НЕ собираются. Нам они нужны (gateway линкует
# llama-common; сервер — основа OpenAI-API). Форсируем до подключения движка.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)

if(NOT TARGET llama)
    add_subdirectory(${INFCORE_LLAMA_ROOT} ${CMAKE_BINARY_DIR}/_llama)
endif()

# -----------------------------------------------------------------------------
# Слой «своё» (корпоративная лицензия). Общается с движком только через C-API
# llama.h и через хелперы llama-common; файлы апстрима не редактируются.
# -----------------------------------------------------------------------------
add_subdirectory(security)
add_subdirectory(observability)
add_subdirectory(registry)
add_subdirectory(runtime)
add_subdirectory(gateway)
add_subdirectory(cli)

message(STATUS "infcore: движок встроен из ${INFCORE_LLAMA_ROOT}")
