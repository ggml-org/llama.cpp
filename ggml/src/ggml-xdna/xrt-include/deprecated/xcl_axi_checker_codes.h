/**
 * Copyright (C) 2016-2017 Xilinx, Inc
 * Copyright (C) 2022 Advanced Micro Devices, Inc. - All rights reserved
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

/**
 * Xilinx SDAccel HAL userspace driver extension APIs
 * Performance Monitoring Exposed Parameters
 * Copyright (C) 2015-2017, Xilinx Inc - All rights reserved
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
#ifndef _AXICHECKERCODES_H_
#define _AXICHECKERCODES_H_

#include "core/include/xcl_app_debug.h"

class xclAXICheckerCodes {
  enum class AXICheckerCodes {
    AXI_ERRM_AWADDR_BOUNDARY,
    AXI_ERRM_AWADDR_WRAP_ALIGN,
    AXI_ERRM_AWBURST,
    AXI_ERRM_AWLEN_LOCK,
    AXI_ERRM_AWCACHE,
    AXI_ERRM_AWLEN_FIXED,
    AXI_ERRM_AWLEN_WRAP,
    AXI_ERRM_AWSIZE,
    AXI_ERRM_AWVALID_RESET,
    AXI_ERRM_AWADDR_STABLE,
    AXI_ERRM_AWBURST_STABLE,
    AXI_ERRM_AWCACHE_STABLE,
    AXI_ERRM_AWID_STABLE,
    AXI_ERRM_AWLEN_STABLE,
    AXI_ERRM_AWLOCK_STABLE,
    AXI_ERRM_AWPROT_STABLE,
    AXI_ERRM_AWSIZE_STABLE,
    AXI_ERRM_AWQOS_STABLE,
    AXI_ERRM_AWREGION_STABLE,
    AXI_ERRM_AWVALID_STABLE,
    AXI_RECS_AWREADY_MAX_WAIT,
    AXI_ERRM_WDATA_NUM,
    AXI_ERRM_WSTRB,
    AXI_ERRM_WVALID_RESET,
    AXI_ERRM_WDATA_STABLE,
    AXI_ERRM_WLAST_STABLE,
    AXI_ERRM_WSTRB_STABLE,
    AXI_ERRM_WVALID_STABLE,
    AXI_RECS_WREADY_MAX_WAIT,
    AXI_ERRS_BRESP_WLAST,
    AXI_ERRS_BRESP_EXOKAY,
    AXI_ERRS_BVALID_RESET,
    AXI_ERRS_BRESP_AW,
    AXI_ERRS_BID_STABLE,
    AXI_ERRS_BRESP_STABLE,
    AXI_ERRS_BVALID_STABLE,
    AXI_RECM_BREADY_MAX_WAIT,
    AXI_ERRM_ARADDR_BOUNDARY,
    AXI_ERRM_ARADDR_WRAP_ALIGN,
    AXI_ERRM_ARBURST,
    AXI_ERRM_ARLEN_LOCK,
    AXI_ERRM_ARCACHE,
    AXI_ERRM_ARLEN_FIXED,
    AXI_ERRM_ARLEN_WRAP,
    AXI_ERRM_ARSIZE,
    AXI_ERRM_ARVALID_RESET,
    AXI_ERRM_ARADDR_STABLE,
    AXI_ERRM_ARBURST_STABLE,
    AXI_ERRM_ARCACHE_STABLE,
    AXI_ERRM_ARID_STABLE,
    AXI_ERRM_ARLEN_STABLE,
    AXI_ERRM_ARLOCK_STABLE,
    AXI_ERRM_ARPROT_STABLE,
    AXI_ERRM_ARSIZE_STABLE,
    AXI_ERRM_ARQOS_STABLE,
    AXI_ERRM_ARREGION_STABLE,
    AXI_ERRM_ARVALID_STABLE,
    AXI_RECS_ARREADY_MAX_WAIT,
    AXI_ERRS_RDATA_NUM,
    AXI_ERRS_RID,
    AXI_ERRS_RRESP_EXOKAY,
    AXI_ERRS_RVALID_RESET,
    AXI_ERRS_RDATA_STABLE,
    AXI_ERRS_RID_STABLE,
    AXI_ERRS_RLAST_STABLE,
    AXI_ERRS_RRESP_STABLE,
    AXI_ERRS_RVALID_STABLE,
    AXI_RECM_RREADY_MAX_WAIT,
    AXI_ERRM_EXCL_ALIGN,
    AXI_ERRM_EXCL_LEN,
    AXI_RECM_EXCL_MATCH,
    AXI_ERRM_EXCL_MAX,
    AXI_RECM_EXCL_PAIR,
    AXI_ERRM_AWUSER_STABLE,
    AXI_ERRM_WUSER_STABLE,
    AXI_ERRS_BUSER_STABLE,
    AXI_ERRM_ARUSER_STABLE,
    AXI_ERRS_RUSER_STABLE,
    AXI_AUXM_RCAM_OVERFLOW,
    AXI_AUXM_RCAM_UNDERFLOW,
    AXI_AUXM_WCAM_OVERFLOW,
    AXI_AUXM_WCAM_UNDERFLOW,
    AXI_AUXM_EXCL_OVERFLOW,
    AXI4LITE_ERRS_BRESP_EXOKAY,
    AXI4LITE_ERRS_RRESP_EXOKAY,
    AXI4LITE_AUXM_DATA_WIDTH,
    XILINX_AW_SUPPORTS_NARROW_BURST,
    XILINX_AR_SUPPORTS_NARROW_BURST,
    XILINX_AW_SUPPORTS_NARROW_CACHE,
    XILINX_AR_SUPPORTS_NARROW_CACHE,
    XILINX_AW_MAX_BURST,
    XILINX_AR_MAX_BURST,
    XILINX_AWVALID_RESET,
    XILINX_WVALID_RESET,
    XILINX_BVALID_RESET,
    XILINX_ARVALID_RESET,
    XILINX_RVALID_RESET,
    RECS_CONTINUOUS_RTRANSFERS_MAX_WAIT,
    RECM_CONTINUOUS_WTRANSFERS_MAX_WAIT,
    RECM_WLAST_TO_AWVALID_MAX_WAIT,
    RECS_WRITE_TO_BVALID_MAX_WAIT
  };
public:
  static bool isValidAXICheckerCodes(unsigned int aOverallStatus, unsigned int aSnapshot[XLAPC_STATUS_REG_NUM], unsigned int aCumulative[XLAPC_STATUS_REG_NUM]) {
    //Validate the codes read from the device.
    //overallstatus could be 0 or 1
    if (aOverallStatus > 1) return false;
    //There are 101 possible codes and hence there cannot be any bits set beyond bit-100
    if (aSnapshot[3]>>5 != 0) return false;
    //There should be only 1 bit set in Snapshot
    int nonzero_snapshots = 0;
    for (int i=0; i<XLAPC_STATUS_REG_NUM; ++i) {
      if (aSnapshot[i]!=0) {
        if ((aSnapshot[i] & (aSnapshot[i]-1)) != 0) return false;
        if (nonzero_snapshots) return false;
        ++nonzero_snapshots;
        //Bit set in snapshot must also be set in cumulative
        if ((aSnapshot[i] & aCumulative[i]) == 0) return false;
      }
    }
    //if snapshot is all zero then overallstatus and cumulative should be zero
    if (!nonzero_snapshots) {
      if (aOverallStatus) return false;
      for (int i=0; i<XLAPC_STATUS_REG_NUM; ++i) {
        if (aCumulative[i]) return false;
      }
    }
    return true;
  }
  static std::string decodeAXICheckerCodes(unsigned int aWord[4]) {
  static const char* AXICheckerStrings [101] = {
    "AXI_ERRM_AWADDR_BOUNDARY",
    "AXI_ERRM_AWADDR_WRAP_ALIGN",
    "AXI_ERRM_AWBURST",
    "AXI_ERRM_AWLEN_LOCK",
    "AXI_ERRM_AWCACHE",
    "AXI_ERRM_AWLEN_FIXED",
    "AXI_ERRM_AWLEN_WRAP",
    "AXI_ERRM_AWSIZE",
    "AXI_ERRM_AWVALID_RESET",
    "AXI_ERRM_AWADDR_STABLE",
    "AXI_ERRM_AWBURST_STABLE",
    "AXI_ERRM_AWCACHE_STABLE",
    "AXI_ERRM_AWID_STABLE",
    "AXI_ERRM_AWLEN_STABLE",
    "AXI_ERRM_AWLOCK_STABLE",
    "AXI_ERRM_AWPROT_STABLE",
    "AXI_ERRM_AWSIZE_STABLE",
    "AXI_ERRM_AWQOS_STABLE",
    "AXI_ERRM_AWREGION_STABLE",
    "AXI_ERRM_AWVALID_STABLE",
    "AXI_RECS_AWREADY_MAX_WAIT",
    "AXI_ERRM_WDATA_NUM",
    "AXI_ERRM_WSTRB",
    "AXI_ERRM_WVALID_RESET",
    "AXI_ERRM_WDATA_STABLE",
    "AXI_ERRM_WLAST_STABLE",
    "AXI_ERRM_WSTRB_STABLE",
    "AXI_ERRM_WVALID_STABLE",
    "AXI_RECS_WREADY_MAX_WAIT",
    "AXI_ERRS_BRESP_WLAST",
    "AXI_ERRS_BRESP_EXOKAY",
    "AXI_ERRS_BVALID_RESET",
    "AXI_ERRS_BRESP_AW",
    "AXI_ERRS_BID_STABLE",
    "AXI_ERRS_BRESP_STABLE",
    "AXI_ERRS_BVALID_STABLE",
    "AXI_RECM_BREADY_MAX_WAIT",
    "AXI_ERRM_ARADDR_BOUNDARY",
    "AXI_ERRM_ARADDR_WRAP_ALIGN",
    "AXI_ERRM_ARBURST",
    "AXI_ERRM_ARLEN_LOCK",
    "AXI_ERRM_ARCACHE",
    "AXI_ERRM_ARLEN_FIXED",
    "AXI_ERRM_ARLEN_WRAP",
    "AXI_ERRM_ARSIZE",
    "AXI_ERRM_ARVALID_RESET",
    "AXI_ERRM_ARADDR_STABLE",
    "AXI_ERRM_ARBURST_STABLE",
    "AXI_ERRM_ARCACHE_STABLE",
    "AXI_ERRM_ARID_STABLE",
    "AXI_ERRM_ARLEN_STABLE",
    "AXI_ERRM_ARLOCK_STABLE",
    "AXI_ERRM_ARPROT_STABLE",
    "AXI_ERRM_ARSIZE_STABLE",
    "AXI_ERRM_ARQOS_STABLE",
    "AXI_ERRM_ARREGION_STABLE",
    "AXI_ERRM_ARVALID_STABLE",
    "AXI_RECS_ARREADY_MAX_WAIT",
    "AXI_ERRS_RDATA_NUM",
    "AXI_ERRS_RID",
    "AXI_ERRS_RRESP_EXOKAY",
    "AXI_ERRS_RVALID_RESET",
    "AXI_ERRS_RDATA_STABLE",
    "AXI_ERRS_RID_STABLE",
    "AXI_ERRS_RLAST_STABLE",
    "AXI_ERRS_RRESP_STABLE",
    "AXI_ERRS_RVALID_STABLE",
    "AXI_RECM_RREADY_MAX_WAIT",
    "AXI_ERRM_EXCL_ALIGN",
    "AXI_ERRM_EXCL_LEN",
    "AXI_RECM_EXCL_MATCH",
    "AXI_ERRM_EXCL_MAX",
    "AXI_RECM_EXCL_PAIR",
    "AXI_ERRM_AWUSER_STABLE",
    "AXI_ERRM_WUSER_STABLE",
    "AXI_ERRS_BUSER_STABLE",
    "AXI_ERRM_ARUSER_STABLE",
    "AXI_ERRS_RUSER_STABLE",
    "AXI_AUXM_RCAM_OVERFLOW",
    "AXI_AUXM_RCAM_UNDERFLOW",
    "AXI_AUXM_WCAM_OVERFLOW",
    "AXI_AUXM_WCAM_UNDERFLOW",
    "AXI_AUXM_EXCL_OVERFLOW",
    "AXI4LITE_ERRS_BRESP_EXOKAY",
    "AXI4LITE_ERRS_RRESP_EXOKAY",
    "AXI4LITE_AUXM_DATA_WIDTH",
    "XILINX_AW_SUPPORTS_NARROW_BURST",
    "XILINX_AR_SUPPORTS_NARROW_BURST",
    "XILINX_AW_SUPPORTS_NARROW_CACHE",
    "XILINX_AR_SUPPORTS_NARROW_CACHE",
    "XILINX_AW_MAX_BURST",
    "XILINX_AR_MAX_BURST",
    "XILINX_AWVALID_RESET",
    "XILINX_WVALID_RESET",
    "XILINX_BVALID_RESET",
    "XILINX_ARVALID_RESET",
    "XILINX_RVALID_RESET",
    "RECS_CONTINUOUS_RTRANSFERS_MAX_WAIT",
    "RECM_CONTINUOUS_WTRANSFERS_MAX_WAIT",
    "RECM_WLAST_TO_AWVALID_MAX_WAIT",
    "RECS_WRITE_TO_BVALID_MAX_WAIT"
  };
  static const char* AXICheckerExplanations [101] = {
    "A write burst cannot cross a 4KB boundary",
    "A write transaction with burst type WRAP has an aligned address",
    "A value of 2'b11 on AWBURST is not permitted when AWVALID is High",
    "Exclusive access transactions cannot have a length greater than 16 beats. This check is not implemented",
    "If not cacheable (AWCACHE[1] == 1'b0), AWCACHE = 2'b00",
    "Transactions of burst type FIXED cannot have a length greater than 16 beats",
    "A write transaction with burst type WRAP has a length of 2, 4, 8, or 16",
    "The size of a write transfer does not exceed the width of the data interface",
    "AWVALID is Low for the first cycle after ARESETn goes High",
    "Handshake Checks: AWADDR must remain stable when AWVALID is asserted and AWREADY Low",
    "Handshake Checks: AWBURST must remain stable when AWVALID is asserted and AWREADY Low",
    "Handshake Checks: AWCACHE must remain stable when AWVALID is asserted and AWREADY Low",
    "Handshake Checks: AWID must remain stable when AWVALID is asserted and AWREADY Low",
    "Handshake Checks: AWLEN must remain stable when AWVALID is asserted and AWREADY Low",
    "Handshake Checks: AWLOCK must remain stable when AWVALID is asserted and AWREADY Low. This check is not implemented",
    "Handshake Checks: AWPROT must remain stable when AWVALID is asserted and AWREADY Low",
    "Handshake Checks: AWSIZE must remain stable when AWVALID is asserted and AWREADY Low",
    "Handshake Checks: AWQOS must remain stable when AWVALID is asserted and AWREADY Low",
    "Handshake Checks: AWREGION must remain stable when ARVALID is asserted and AWREADY Low",
    "Handshake Checks: Once AWVALID is asserted, it must remain asserted until AWREADY is High",
    "Recommended that AWREADY is asserted within MAXWAITS cycles of AWVALID being asserted",
    "The number of write data items matches AWLEN for the corresponding address. This is triggered when any of the following occurs: a) Write data arrives, WLAST is set, and the WDATA count is not equal to AWLEN. b) Write data arrives, WLAST is not set, and the WDATA count is equal to AWLEN. c) ADDR arrives, WLAST is already received, and the WDATA count is not equal to AWLEN",
    "Write strobes must only be asserted for the correct byte lanes as determined from the: Start Address, Transfer Size and Beat Number",
    "WVALID is LOW for the first cycle after ARESETn goes High",
    "Handshake Checks: WDATA must remain stable when WVALID is asserted and WREADY Low",
    "Handshake Checks: WLAST must remain stable when WVALID is asserted and WREADY Low",
    "Handshake Checks: WSTRB must remain stable when WVALID is asserted and WREADY Low",
    "Handshake Checks: Once WVALID is asserted, it must remain asserted until WREADY is High",
    "Recommended that WREADY is asserted within MAXWAITS cycles of WVALID being asserted",
    "A slave must not take BVALID HIGH until after the last write data handshake is complete",
    "An EXOKAY write response can only be given to an exclusive write access. This check is not implemented",
    "BVALID is Low for the first cycle after ARESETn goes High",
    "A slave must not take BVALID HIGH until after the write address handshake is complete",
    "Handshake Checks: BID must remain stable when BVALID is asserted and BREADY Low ",
    "Checks BRESP must remain stable when BVALID is asserted and BREADY Low",
    "Once BVALID is asserted, it must remain asserted until BREADY is High",
    "Recommended that BREADY is asserted within MAXWAITS cycles of BVALID being asserted",
    "A read burst cannot cross a 4KB boundary",
    "A read transaction with a burst type of WRAP must have an aligned address",
    "A value of 2'b11 on ARBURST is not permitted when ARVALID is High",
    "Exclusive access transactions cannot have a length greater than 16 beats. This check is not implemented",
    "When ARVALID is HIGH, if ARCACHE[1] is LOW, then ARCACHE[3:2] must also be Low",
    "Transactions of burst type FIXED cannot have a length greater than 16 beats",
    "A read transaction with burst type of WRAP must have a length of 2, 4, 8, or 16",
    "The size of a read transfer must not exceed the width of the data interface",
    "ARVALID is Low for the first cycle after ARESETn goes High",
    "ARADDR must remain stable when ARVALID is asserted and ARREADY Low",
    "ARBURST must remain stable when ARVALID is asserted and ARREADY Low",
    "ARCACHE must remain stable when ARVALID is asserted and ARREADY Low",
    "ARID must remain stable when ARVALID is asserted and ARREADY Low",
    "ARLEN must remain stable when ARVALID is asserted and ARREADY Low",
    "ARLOCK must remain stable when ARVALID is asserted and ARREADY Low. This check is not implemented",
    "ARPROT must remain stable when ARVALID is asserted and ARREADY Low",
    "ARSIZE must remain stable when ARVALID is asserted and ARREADY Low",
    "ARQOS must remain stable when ARVALID is asserted and ARREADY Low",
    "ARREGION must remain stable when ARVALID is asserted and ARREADY Low",
    "Once ARVALID is asserted, it must remain asserted until ARREADY is High",
    "Recommended that ARREADY is asserted within MAXWAITS cycles of ARVALID being asserted",
    "The number of read data items must match the corresponding ARLEN",
    "The read data must always follow the address that it relates to. If IDs are used, RID must also match ARID of an outstanding address read transaction. This violation can also occur when RVALID is asserted with no preceding AR transfer",
    "An EXOKAY write response can only be given to an exclusive read access. This check is not implemented",
    "RVALID is Low for the first cycle after ARESETn goes High",
    "RDATA must remain stable when RVALID is asserted and RREADY Low",
    "RID must remain stable when RVALID is asserted and RREADY Low",
    "RLAST must remain stable when RVALID is asserted and RREADY Low",
    "RRESP must remain stable when RVALID is asserted and RREADY Low",
    "Once RVALID is asserted, it must remain asserted until RREADY is High",
    "Recommended that RREADY is asserted within MAXWAITS cycles of RVALID being asserted",
    "The address of an exclusive access is aligned to the total number of bytes in the transaction. This check is not implemented",
    "The number of bytes to be transferred in an exclusive access burst is a power of 2, that is, 1, 2, 4, 8, 16, 32, 64, or 128 bytes. This check is not implemented",
    "Recommended that the address, size, and length of an exclusive write with a given ID is the same as the address, size, and length of the preceding exclusive read with the same ID. This check is not implemented",
    "128 is the maximum number of bytes that can be transferred in an exclusive burst. This check is not implemented",
    "Recommended that every exclusive write has an earlier outstanding exclusive read with the same ID. This check is not implemented",
    "AWUSER must remain stable when AWVALID is asserted and AWREADY Low",
    "WUSER must remain stable when WVALID is asserted and WREADY Low",
    "BUSER must remain stable when BVALID is asserted and BREADY Low",
    "ARUSER must remain stable when ARVALID is asserted and ARREADY Low",
    "RUSER must remain stable when RVALID is asserted and RREADY Low",
    "Read CAM overflow, increase MAXRBURSTS parameter",
    "Read CAM underflow",
    "Write CAM overflow, increase MAXWBURSTS parameter",
    "Write CAM underflow",
    "Exclusive access monitor overflow",
    "A slave must not give an EXOKAY response on an AXI4-Lite interface",
    "A slave must not give an EXOKAY response on an AXI4-Lite interface",
    "DATA_WIDTH parameter is 32 or 64",
    "When the connection does not support narrow transfers, the AW Master cannot issue a transfer with AWLEN > 0 and AWSIZE is less than the defined interface DATA_WIDTH",
    "When the connection does not support narrow transfers, the AR Master cannot issue a transfer with ARLEN > 0 and ARSIZE is less than the defined interface DATA_WIDTH",
    "When the connection does not support narrow transfers, the AW Master cannot issue a transfer with AWLEN > 0 and AWCACHE modifiable bit is not asserted",
    "When the connection does not support narrow transfers, the AR Master cannot issue a transfer with ARLEN > 0 and ARCACHE modifiable bit is not asserted",
    "AW Master cannot issue AWLEN greater than the configured maximum burst length",
    "AR Master cannot issue ARLEN greater than the configured maximum burst length",
    "AWREADY is Low for the first cycle after ARESETn goes High",
    "WREADY is Low for the first cycle after ARESETn goes High",
    "BREADY is Low for the first cycle after ARESETn goes High",
    "ARREADY is Low for the first cycle after ARESETn goes High",
    "RREADY is Low for the first cycle after ARESETn goes High",
    "RVALID should be asserted within MAXWAITS cycles of either AR command transfer or previous R transfer while there are outstanding AR commands.",
    "WVALID should be asserted within MAXWAITS cycles of either AW command transfer or previous W transfer while there are outstanding AW commands.",
    "AWVALID should be asserted within MAXWAITS cycles of WLAST transfer or previous AW transfer if there are yet more WLAST transfers outstanding.",
    "BVALID should be asserted within MAXWAITS cycles of AW command transfer or WLAST transfer (whichever is later), or previous B transfer if there are yet more AW and WLAST transfers outstanding."
  };
    std::string s;
    for (unsigned int j = 0; j<4; ++j) {
      unsigned int i = 0;
      unsigned int w = aWord[j];
      while (w) {
        if (w & ((unsigned int)0x1)) {
          s.append(AXICheckerStrings[j*32+i]).append(":").append(AXICheckerExplanations[j*32+i]).append("\n");
        }
        w = w>>1;
        ++i;
      }
    }
    return s;
  }
};

class xclStreamingAXICheckerCodes
{
 private:

  enum StreamingAXICheckerCodes {
    AXI4STREAM_ERRM_TVALID_RESET    = 0x001,
    AXI4STREAM_ERRM_TID_STABLE      = 0x002,
    AXI4STREAM_ERRM_TDEST_STABLE    = 0x004,
    AXI4STREAM_ERRM_TKEEP_STABLE    = 0x008,
    AXI4STREAM_ERRM_TDATA_STABLE    = 0x010,
    AXI4STREAM_ERRM_TLAST_STABLE    = 0x020,
    AXI4STREAM_ERRM_TSTRB_STABLE    = 0x040,
    AXI4STREAM_ERRM_TVALID_STABLE   = 0x080,
    AXI4STREAM_RECS_TREADY_MAX_WAIT = 0x100,
    AXI4STREAM_ERRM_TUSER_STABLE    = 0x200,
    AXI4STREAM_ERRM_TKEEP_TSTRB     = 0x400,
    XILINX_ARESETN_PULSE_WIDTH      = 0x800
  } ;

 public:

  static bool isValidStreamingAXICheckerCodes(unsigned int pc_asserted,
					      unsigned int current_pc,
					      unsigned int snapshot_pc) 
  {
    if (pc_asserted) 
    {
      // The snapshot PC register should only be one of the valid error codes
      switch ((StreamingAXICheckerCodes)(snapshot_pc))
      {
      case AXI4STREAM_ERRM_TVALID_RESET:
      case AXI4STREAM_ERRM_TID_STABLE:
      case AXI4STREAM_ERRM_TDEST_STABLE:
      case AXI4STREAM_ERRM_TKEEP_STABLE:
      case AXI4STREAM_ERRM_TDATA_STABLE:
      case AXI4STREAM_ERRM_TLAST_STABLE:
      case AXI4STREAM_ERRM_TSTRB_STABLE:
      case AXI4STREAM_ERRM_TVALID_STABLE:
      case AXI4STREAM_RECS_TREADY_MAX_WAIT:
      case AXI4STREAM_ERRM_TUSER_STABLE:
      case AXI4STREAM_ERRM_TKEEP_TSTRB:
      case XILINX_ARESETN_PULSE_WIDTH:
	break ;
      default:
	return false ; 
      }

      // The current_pc can have any combination of the low-order 12 bits set
      if ((current_pc >> 12) != 0) return false ;
      if ((current_pc & 0xfff) == 0) return false ;

      return true ;
    }
    else
    {
      // If pc_asserted is false, then there should be nothing on
      //  the other registers.
      return (current_pc == 0 && snapshot_pc == 0) ;
    }
  }

  static std::string decodeStreamingAXICheckerCodes(unsigned int aWord) {
    static const char* StreamingAXICheckerStrings[12] = {
      "AXI4STREAM_ERRM_TVALID_RESET",
      "AXI4STREAM_ERRM_TID_STABLE",
      "AXI4STREAM_ERRM_TDEST_STABLE",
      "AXI4STREAM_ERRM_TKEEP_STABLE",
      "AXI4STREAM_ERRM_TDATA_STABLE",
      "AXI4STREAM_ERRM_TLAST_STABLE",
      "AXI4STREAM_ERRM_TSTRB_STABLE",
      "AXI4STREAM_ERRM_TVALID_STABLE",
      "AXI4STREAM_RECS_TREADY_MAX_WAIT",
      "AXI4STREAM_ERRM_TUSER_STABLE",
      "AXI4STREAM_ERRM_TKEEP_TSTRB",
      "XILINX_ARESETN_PULSE_WIDTH"
    };
    static const char* StreamingAXICheckerExplanations[12] = {
      "TVALID is low for the first cycle after aresetn goes high",
      "TID remains stable when TVALID is asserted and TREADY is low",
      "TDEST remains stable when TVALID is asserted and TREADY is low",
      "TKEEP remains stable when TVALID is asserted and TREADY is low",
      "TDATA reamins stable when TVALID is asserted and TREADY is low",
      "TLAST reamins stable when TVALID is asserted and TREADY is low",
      "TSTRB reamins stable when TVALID is asserted and TREADY is low",
      "When TVALID is asserted, it must remain asserted until TREADY is high",
      "It is recommended that TREADY is asserted within MAXWAITS cycles of TVALID being asserted",
      "TUSER reamins stable when TVALID is asserted and TREADY is low",
      "If TKEEP is de-asserted, then TSTRB must also be de-asserted",
      "ARESETn must be low for at least 16 ACLKn cycles"
    };
    std::string s = "" ;
    unsigned int w = aWord;
    unsigned int i = 0 ;
    while (w)
    {
      if (w & (unsigned int)(0x1))
      {
	s.append(StreamingAXICheckerStrings[i]);
	s.append(":");
	s.append(StreamingAXICheckerExplanations[i]);
	s.append("\n");
      }
      w = w >> 1;
      ++i;
    }
    return s;
  }

};

#endif
