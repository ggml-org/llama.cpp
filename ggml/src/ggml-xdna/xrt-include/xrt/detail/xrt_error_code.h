/*
 *  Copyright (C) 2020, Xilinx Inc
 *
 *  This file is dual licensed.  It may be redistributed and/or modified
 *  under the terms of the Apache 2.0 License OR version 2 of the GNU
 *  General Public License.
 *
 *  Apache License Verbiage
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  GPL license Verbiage:
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License as
 *  published by the Free Software Foundation; either version 2 of the
 *  License, or (at your option) any later version.  This program is
 *  distributed in the hope that it will be useful, but WITHOUT ANY
 *  WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
 *  License for more details.  You should have received a copy of the
 *  GNU General Public License along with this program; if not, write
 *  to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
 *  Boston, MA 02111-1307 USA
 *
 */
#ifndef INCLUDE_XRT_DETAIL_XRT_ERROR_CODE_H_
#define INCLUDE_XRT_DETAIL_XRT_ERROR_CODE_H_

#if defined(__KERNEL__)
# include <linux/types.h>
#elif defined(__cplusplus)
# include <cstdint>
#else
# include <stdint.h>
#endif

/**
 * xrtErrorCode layout
 *
 * This layout is internal to XRT (akin to a POSIX error code).
 *
 * The error code is populated by driver and consumed by XRT
 * implementation where it is translated into an actual error / info /
 * warning that is propagated to the end user.
 *
 * 63 - 48  47 - 40   39 - 32   31 - 24   23 - 16    15 - 0
 * --------------------------------------------------------
 * |    |    |    |    |    |    |    |    |    |    |----| xrtErrorNum
 * |    |    |    |    |    |    |    |    |----|---------- xrtErrorDriver
 * |    |    |    |    |    |    |----|-------------------- xrtErrorSeverity
 * |    |    |    |    |----|------------------------------ xrtErrorModule
 * |    |    |----|---------------------------------------- xrtErrorClass
 * |----|-------------------------------------------------- reserved
 *
 */
typedef uint64_t xrtErrorCode;
typedef uint64_t xrtErrorTime;

#define XRT_ERROR_NUM_MASK		0xFFFFUL
#define XRT_ERROR_NUM_SHIFT		0
#define XRT_ERROR_DRIVER_MASK		0xFUL
#define XRT_ERROR_DRIVER_SHIFT		16
#define XRT_ERROR_SEVERITY_MASK		0xFUL
#define XRT_ERROR_SEVERITY_SHIFT	24
#define XRT_ERROR_MODULE_MASK		0xFUL
#define XRT_ERROR_MODULE_SHIFT		32
#define XRT_ERROR_CLASS_MASK		0xFUL
#define XRT_ERROR_CLASS_SHIFT		40

#define	XRT_ERROR_CODE_BUILD(num, driver, severity, module, eclass) \
	((((num) & XRT_ERROR_NUM_MASK) << XRT_ERROR_NUM_SHIFT) | \
	(((driver) & XRT_ERROR_DRIVER_MASK) << XRT_ERROR_DRIVER_SHIFT) | \
	(((severity) & XRT_ERROR_SEVERITY_MASK) << XRT_ERROR_SEVERITY_SHIFT) | \
	(((module) & XRT_ERROR_MODULE_MASK) << XRT_ERROR_MODULE_SHIFT) | \
	(((eclass) & XRT_ERROR_CLASS_MASK) << XRT_ERROR_CLASS_SHIFT))

#define XRT_ERROR_NUM(code) (((code) >> XRT_ERROR_NUM_SHIFT) & XRT_ERROR_NUM_MASK)
#define XRT_ERROR_DRIVER(code) (((code) >> XRT_ERROR_DRIVER_SHIFT) & XRT_ERROR_DRIVER_MASK)
#define XRT_ERROR_SEVERITY(code) (((code) >> XRT_ERROR_SEVERITY_SHIFT) & XRT_ERROR_SEVERITY_MASK)
#define XRT_ERROR_MODULE(code) (((code) >> XRT_ERROR_MODULE_SHIFT) & XRT_ERROR_MODULE_MASK)
#define XRT_ERROR_CLASS(code) (((code) >> XRT_ERROR_CLASS_SHIFT) & XRT_ERROR_CLASS_MASK)

/**
 * xrt_error_num - XRT specific error numbers
 */
enum xrtErrorNum {
  XRT_ERROR_NUM_FIRWWALL_TRIP = 1,
  XRT_ERROR_NUM_TEMP_HIGH,
  XRT_ERROR_NUM_AIE_SATURATION,
  XRT_ERROR_NUM_AIE_FP,
  XRT_ERROR_NUM_AIE_STREAM,
  XRT_ERROR_NUM_AIE_ACCESS,
  XRT_ERROR_NUM_AIE_BUS,
  XRT_ERROR_NUM_AIE_INSTRUCTION,
  XRT_ERROR_NUM_AIE_ECC,
  XRT_ERROR_NUM_AIE_LOCK,
  XRT_ERROR_NUM_AIE_DMA,
  XRT_ERROR_NUM_AIE_MEM_PARITY,
  XRT_ERROR_NUM_KDS_CU,
  XRT_ERROR_NUM_KDS_EXEC,
  XRT_ERROR_NUM_UNKNOWN
};

enum xrtErrorDriver {
  XRT_ERROR_DRIVER_XOCL = 1,
  XRT_ERROR_DRIVER_XCLMGMT,
  XRT_ERROR_DRIVER_ZOCL,
  XRT_ERROR_DRIVER_AIE,
  XRT_ERROR_DRIVER_UNKNOWN
};

enum xrtErrorSeverity {
  XRT_ERROR_SEVERITY_EMERGENCY = 1,
  XRT_ERROR_SEVERITY_ALERT,
  XRT_ERROR_SEVERITY_CRITICAL,
  XRT_ERROR_SEVERITY_ERROR,
  XRT_ERROR_SEVERITY_WARNING,
  XRT_ERROR_SEVERITY_NOTICE,
  XRT_ERROR_SEVERITY_INFO,
  XRT_ERROR_SEVERITY_DEBUG,
  XRT_ERROR_SEVERITY_UNKNOWN
};

enum xrtErrorModule {
  XRT_ERROR_MODULE_FIREWALL = 1,
  XRT_ERROR_MODULE_CMC,
  XRT_ERROR_MODULE_AIE_CORE,
  XRT_ERROR_MODULE_AIE_MEMORY,
  XRT_ERROR_MODULE_AIE_SHIM,
  XRT_ERROR_MODULE_AIE_NOC,
  XRT_ERROR_MODULE_AIE_PL,
  XRT_ERROR_MODULE_UNKNOWN
};

enum xrtErrorClass {
  XRT_ERROR_CLASS_FIRST_ENTRY = 1,
  XRT_ERROR_CLASS_SYSTEM = XRT_ERROR_CLASS_FIRST_ENTRY,
  XRT_ERROR_CLASS_AIE,
  XRT_ERROR_CLASS_HARDWARE,
  XRT_ERROR_CLASS_UNKNOWN,
  XRT_ERROR_CLASS_LAST_ENTRY = XRT_ERROR_CLASS_UNKNOWN
};

typedef uint64_t xrtExErrorCode;

/**
*xrtExErrorCode layout
*
* This layout is internal to XRT(akin to a POSIX error code).
*
* The error code is populated by driver and consumed by XRT
* implementation where it is translated into an actual error / info /
* warning that is propagated to the end user.
*
*63 - 48   47 - 32   31 - 16   15 - 0
* --------------------------------------
* |    |    |    |    |    |    |----|  ExErrorID
* |    |    |    |    |----|----------- AIE_LOC_COL
* |    |    |----|----------------------AIR_LOC_ROW
* |----|--------------------------------RESERVED
*
*/

#define XRT_EX_ERROR_ID_MASK          0xFFFFUL
#define XRT_EX_ERROR_ID_SHIFT         0
#define XRT_EX_ERROR_LOC_COL_MASK     0xFFFFUL
#define XRT_EX_ERROR_LOC_COL_SHIFT    16
#define XRT_EX_ERROR_LOC_ROW_MASK     0xFFFFUL
#define XRT_EX_ERROR_LOC_ROW_SHIFT    32
#define XRT_EX_ERROR_RESERVED_MASK    0xFFFFUL
#define XRT_EX_ERROR_RESERVED_SHIFT   48

#define  XRT_EX_ERROR_CODE_BUILD(ID, COL, ROW, RESERVED) \
    ((static_cast<uint64_t>((ID) & XRT_EX_ERROR_ID_MASK) << XRT_ERROR_NUM_SHIFT) | \
    (static_cast<uint64_t>((COL) & XRT_EX_ERROR_LOC_COL_MASK) << XRT_EX_ERROR_LOC_COL_SHIFT) | \
    (static_cast<uint64_t>((ROW) & XRT_EX_ERROR_LOC_ROW_MASK) << XRT_EX_ERROR_LOC_ROW_SHIFT) | \
    (static_cast<uint64_t>((RESERVED) & XRT_EX_ERROR_RESERVED_MASK) << XRT_EX_ERROR_RESERVED_SHIFT))

#define XRT_EX_ERROR_ID(code) (((code) >> XRT_EX_ERROR_ID_SHIFT) & XRT_EX_ERROR_ID_MASK)
#define XRT_EX_ERROR_LOC_COL(code) (((code) >> XRT_EX_ERROR_LOC_COL_SHIFT) & XRT_EX_ERROR_LOC_COL_MASK)
#define XRT_EX_ERROR_LOC_ROW(code) (((code) >> XRT_EX_ERROR_LOC_ROW_SHIFT) & XRT_EX_ERROR_LOC_ROW_MASK)


#endif
