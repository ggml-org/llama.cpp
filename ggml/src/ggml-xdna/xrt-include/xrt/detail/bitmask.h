// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_DETAIL_BITMASK_H
#define XRT_DETAIL_BITMASK_H

#ifdef __cplusplus
# include <type_traits>
#endif

#ifdef __cplusplus
// Allow enum classes to be used at bitmasks
namespace xrt::detail {

template <typename T>
constexpr std::enable_if_t<std::is_enum_v<T>, T>
operator|(T lhs, T rhs)
{
  using U = std::underlying_type_t<T>;
  return static_cast<T>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

template <typename T>
constexpr std::enable_if_t<std::is_enum_v<T>, T>
operator&(T lhs, T rhs)
{
  using U = std::underlying_type_t<T>;
  return static_cast<T>(static_cast<U>(lhs) & static_cast<U>(rhs));
}

template <typename T>
constexpr std::enable_if_t<std::is_enum_v<T>, T>
operator^(T lhs, T rhs)
{
  using U = std::underlying_type_t<T>;
  return static_cast<T>(static_cast<U>(lhs) ^ static_cast<U>(rhs));
}

template <typename T>
constexpr std::enable_if_t<std::is_enum_v<T>, T>
operator~(T rhs)
{
  using U = std::underlying_type_t<T>;
  return static_cast<T>(~static_cast<U>(rhs)); // NOLINT
}

template <typename T>
constexpr std::enable_if_t<std::is_enum_v<T>, T&>
operator|=(T& lhs, T rhs)
{
  using U = std::underlying_type_t<T>;
  lhs = static_cast<T>(static_cast<U>(lhs) | static_cast<U>(rhs));
  return lhs;
}

template <typename T>
constexpr std::enable_if_t<std::is_enum_v<T>, T&>
operator&=(T& lhs, T rhs)
{
  using U = std::underlying_type_t<T>;
  lhs = static_cast<T>(static_cast<U>(lhs) & static_cast<U>(rhs));
  return lhs;
}

template <typename T>
constexpr std::enable_if_t<std::is_enum_v<T>, T&>
operator^=(T& lhs, T rhs)
{
  using U = std::underlying_type_t<T>;
  lhs = static_cast<T>(static_cast<U>(lhs) ^ static_cast<U>(rhs));
  return lhs;
}

template <typename T>
constexpr std::enable_if_t<std::is_enum_v<T>, bool> operator!(T rhs)
{
  return !static_cast<bool>(static_cast<std::underlying_type_t<T>>(rhs));
}
 
} // xrt::detail

#endif

#endif
