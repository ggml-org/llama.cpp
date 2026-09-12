// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_DETAIL_SPAN_H
#define XRT_DETAIL_SPAN_H

#ifdef __cplusplus

#include <array>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>

// XRT c++17 and ealier do not have have std::span, and ABI
// (sizeof(std::span)) is different between gcc10 and gcc11.
//
// This file defines a simple span replacement with limited
// functionality.  When XRT switches to c++20, and uses a
// proper version of GCC, std::span will be used instead.
namespace xrt::detail {

template <typename T>
class span
{
  T* m_data = nullptr;
  std::size_t m_size = 0;

public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = pointer;
  using const_iterator = const pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const iterator>;
  
  constexpr span() = default;
  constexpr span(T* data, std::size_t size)
    : m_data(data)
    , m_size(size)
  {}

  template< class U, std::size_t N >
  constexpr explicit span(std::array<U, N>& arr ) noexcept
    : span(arr.data(), N)
  {}

  template< class U, std::size_t N >
  constexpr explicit span(const std::array<U, N>& arr ) noexcept
    : span(arr.data(), N)
  {}

  constexpr iterator begin() const noexcept { return m_data; }
  constexpr iterator end() const noexcept { return m_data + m_size; }
  constexpr const_iterator cbegin() const noexcept { return begin(); }
  constexpr const_iterator cend() const noexcept { return end(); }
  
  constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
  constexpr reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
  constexpr const_reverse_iterator crbegin() const noexcept { return rbegin(); }
  constexpr const_iterator crend() const noexcept { return rend(); }

  constexpr reference front() const { return *begin(); }
  constexpr reference back() const { return *(end() - 1) ; }
  constexpr reference at(size_type idx) const
  {
    if (idx < size())
      return data()[idx]; 

    throw std::out_of_range("pos (" + std::to_string(idx) + ") >= size() (" + std::to_string(m_size) + ")");
  }
  constexpr reference operator[] (size_type idx) const { return data()[idx]; }
  constexpr pointer data() const noexcept { return m_data; }

  constexpr size_type size() const noexcept { return m_size; }
  constexpr size_type size_bytes() const noexcept { return size() * sizeof(T); }
  constexpr bool empty() const noexcept { return size() == 0; }

  template <std::size_t count>
  constexpr span<element_type> first() const { return {data(), count}; }
  constexpr span<element_type> first(size_type count) const { return {data(), count}; }

  template <std::size_t count>
  constexpr span<element_type> last() const { return {data() + (size() - count), count}; }
  constexpr span<element_type> last(size_type count) const { return {data() + (size() - count), count}; }

  template <std::size_t offset, std::size_t count>
  constexpr span<element_type> subspan() const { return {data() + offset, count}; }
  constexpr span<element_type> subspan(size_type offset, size_type count) const { return {data() + offset, count}; }
};
  

} // xrt::detail

#endif // __cplusplus
#endif
