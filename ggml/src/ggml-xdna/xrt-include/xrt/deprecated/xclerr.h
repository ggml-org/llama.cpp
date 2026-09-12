/*
 * Copyright (C) 2017 Xilinx, Inc
 * Author: Umang Parekh
 *
 * This file is dual licensed.  It may be redistributed and/or modified
 * under the terms of the Apache 2.0 License OR version 2 of the GNU
 * General Public License.
 *
 * Apache License Verbiage
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
 *
 * GPL license Verbiage:
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by the Free Software Foundation;
 * either version 2 of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with this program;
 * if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

/**
 * DOC: Device error status related structs and defines
 * This file is used by both userspace and xclmgmt kernel driver.
 */

#ifndef INCLUDE_XRT_DEPRECATED_XCLERR_H_
#define INCLUDE_XRT_DEPRECATED_XCLERR_H_

/**
 * enum xclFirewallID - AXI Firewall IDs used to identify individual AXI Firewalls
 *
 * @XCL_FW_MGMT_CONTROL:  MGMT BAR AXI-Lite BAR access protection
 * @XCL_FW_USER_CONTROL:  USER BAR AXI-Lite BAR access protection
 * @XCL_FW_DATAPATH:      DMA data path protection
 */
enum xclFirewallID {
        XCL_FW_MGMT_CONTROL = 0,
        XCL_FW_USER_CONTROL,
        XCL_FW_DATAPATH,
        XCL_FW_MAX_LEVEL // always the last one
};

/**
 * struct xclAXIErrorStatus - Record used to capture specific error
 *
 * @mErrFirewallTime:    Timestamp of when Firewall tripped
 * @mErrFirewallStatus:  Error code obtained from the Firewall
 * @mErrFirewallID:      Firewall ID
 */
struct xclAXIErrorStatus {
        unsigned long       mErrFirewallTime;
        unsigned            mErrFirewallStatus;
        enum xclFirewallID  mErrFirewallID;
};

struct xclPCIErrorStatus {
        unsigned mDeviceStatus;
        unsigned mUncorrErrStatus;
        unsigned mCorrErrStatus;
        unsigned rsvd1;
        unsigned rsvd2;
};

/**
 * struct xclErrorStatus - Container for all error records
 *
 * @mNumFirewalls:    Count of Firewalls in the record (max is 8)
 * @mAXIErrorStatus:  Records holding Firewall information
 * @mPCIErrorStatus:  Unused
 */
struct xclErrorStatus {
        unsigned  mNumFirewalls;
        struct xclAXIErrorStatus mAXIErrorStatus[8];
        struct xclPCIErrorStatus mPCIErrorStatus;
        unsigned mFirewallLevel;
};

#endif /* XCLERR_H_ */
