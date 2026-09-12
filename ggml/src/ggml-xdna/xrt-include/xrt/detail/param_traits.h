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

#ifndef _XRT_DETAIL_PARAM_TRAITS_HPP
#define _XRT_DETAIL_PARAM_TRAITS_HPP

namespace xrt { namespace info {


// Implement a meta-function from (T, value) to T' to express the
// return type value of an OpenCL function of kind (T, value)
template <typename T, T Param>
struct param_traits {
  // By default no return type
};

// To declare a param_traits returning RETURN_TYPE for function of any T
#define XRT_INFO_PARAM_TRAITS_ANY_T(T, RETURN_TYPE)  \
  template <T Param>                                 \
  struct param_traits<T, Param> {                    \
    using return_type = RETURN_TYPE;                 \
  };


// To declare a param_traits returning RETURN_TYPE for function taking
// a VALUE of type T
#define XRT_INFO_PARAM_TRAITS(VALUE, RETURN_TYPE)    \
  template <>                                        \
  struct param_traits<decltype(VALUE), VALUE> {      \
    using return_type = RETURN_TYPE;                 \
  };

}} // info, xrt

#endif
