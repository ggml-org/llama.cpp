#pragma once

// converts between common_json and the backing JSON library
//
// include this only in a cpp file that touches an internal component of the library,
// never in a header. every use here is a place to fix if the library changes.

#include "json.h"

template <typename T> T       & common_json_raw(common_json & json);
template <typename T> const T & common_json_raw(const common_json & json);

template <typename T> common_json common_json_from_raw(const T & json);

// view over a value of the backing library, it does not copy
template <typename T> common_json & common_json_ref_from_raw(T & json);
