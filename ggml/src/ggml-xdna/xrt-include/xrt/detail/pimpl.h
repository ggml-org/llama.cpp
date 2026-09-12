/*
 * Copyright (C) 2021, Xilinx Inc - All rights reserved
 * Xilinx Runtime (XRT) Experimental APIs
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#ifndef _XRT_DETAIL_PIMPL_H_
#define _XRT_DETAIL_PIMPL_H_

#ifdef __cplusplus

#include <memory>

namespace xrt { namespace detail {

// Utility to replicate pimpl
template<typename ImplType>
class pimpl
{
protected:
  ~pimpl() = default;

public:
  pimpl() = default;

  explicit
  pimpl(std::shared_ptr<ImplType> handle)
    : handle(std::move(handle))
  {}

  pimpl(const pimpl&) = default;

  pimpl(pimpl&&) = default;

  pimpl&
  operator=(const pimpl&) = default;

  pimpl&
  operator=(pimpl&&) = default;

  const std::shared_ptr<ImplType>&
  get_handle() const
  {
    return handle;
  }

  explicit
  operator bool() const
  {
    return handle != nullptr;
  }

  bool
  operator < (const pimpl& rhs) const
  {
    return handle < rhs.handle;
  }

protected:
  std::shared_ptr<ImplType> handle;
};

}} // detail, xrt

#endif // __cplusplus
#endif
