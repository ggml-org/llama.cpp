// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.

#ifndef xrt_version_h_
#define xrt_version_h_

#include "xrt/detail/config.h"

#ifdef __cplusplus

/*!
 * @namespace xrt::version
 *
 * @brief
 * APIs for version queries
 */
namespace xrt::version {

/**
 * code() - Returns the version code for the library.
 *
 * The version code is a combination of major and minor version.
 * The major version is shifted left by 16 bits and the minor version
 * is added.
 */
XRT_API_EXPORT
unsigned int
code();

/**
 * major() - Returns the major version for the library.
 *
 * The major version indicates ABI compatibility.  The major version
 * is incremented by 1 only if a release that breaks ABI compatibility.
 */
XRT_API_EXPORT
unsigned int
major();

/**
 * minor() - Returns the minor version for the library.
 *
 * The minor version is incremented by 1 for each release within a
 * major version.  
 */
XRT_API_EXPORT
unsigned int
minor();

/**
 * patch() - Returns the patch version for the library.
 *
 * The patch number defaults to 0 for local builds, but is otherwise
 * controlled by CI and incremented by 1 for each build.
 *
 * The patch number is reset to 0 when the minor version is
 * incremented.
 */
XRT_API_EXPORT
unsigned int
patch();

/**
 * build() - Returns the build number for the library.
 *
 * The build number is the total number of commits to XRT
 * on current branch.
 */
XRT_API_EXPORT
unsigned int
build();

/**
 * feature() - Returns the feature number for the library.
 *
 * The feature number is the total number of commits to XRT
 * main branch.  For branches off of XRT's main branch, the
 * feature number is the total number of commits at the time
 * the branch diverged from XRT's main branch.
 */ 
XRT_API_EXPORT
unsigned int
feature();
 
} // namespace xrt::version

/// @cond
extern "C" {
#endif
  
/**
 * See xrt::version::code()
 */
XRT_API_EXPORT
unsigned int
xrtVersionCode();

/**
 * See xrt::version::major()
 */
XRT_API_EXPORT
unsigned int
xrtVersionMajor();

/**
 * See xrt::version::minor()
 */
XRT_API_EXPORT
unsigned int
xrtVersionMinor();
  
/**
 * See xrt::version::patch()
 */
XRT_API_EXPORT
unsigned int
xrtVersionPatch();

/**
 * See xrt::version::build()
 */
XRT_API_EXPORT
unsigned int
xrtVersionBuild();
  
/**
 * See xrt::version::feature()
 */
XRT_API_EXPORT
unsigned int
xrtVersionFeature();

/// @endcond
#ifdef __cplusplus
}
#endif
  

#endif
