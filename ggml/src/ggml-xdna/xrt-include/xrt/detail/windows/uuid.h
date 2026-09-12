/**
 * Copyright (C) 2019 Xilinx, Inc
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

#ifndef core_include_windows_uuid_h_
#define core_include_windows_uuid_h_

#pragma warning( push )
#pragma warning ( disable : 4100 4996 4244 )

typedef unsigned char xuid_t[16];

#ifdef __cplusplus
#include <string>
#include <cstring>
#include <cstdio>
#include <cctype>
#include <stdexcept>

inline void
uuid_copy(xuid_t dst, const xuid_t src)
{
  std::memcpy(dst,src,sizeof(xuid_t));
}

inline void
uuid_clear(xuid_t uuid)
{
  std::memset(uuid, 0, sizeof(xuid_t));
}

inline int
uuid_compare(const xuid_t uuid1, const xuid_t uuid2)
{
  return memcmp(uuid1, uuid2, sizeof(xuid_t));
}

inline int
uuid_is_null(const xuid_t uuid)
{
  for (int i=0; i<sizeof(xuid_t) ;++i)
    if (uuid[i])
      return 0;
  return 1;
}

inline void
uuid_unparse_lower(const xuid_t uuid, char* str)
{
  std::sprintf(str,"%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
	       uuid[0], uuid[1], uuid[2], uuid[3],
	       uuid[4], uuid[5],
	       uuid[6], uuid[7],
	       uuid[8], uuid[9],
	       uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]);
}

inline void
uuid_parse(const char* str, xuid_t uuid)
{
  //381d7988-e0b4-421b-811a-cdcf83ad2764
  constexpr int uuid_str_sz = 36;

  for (int i=0; i<uuid_str_sz; ++i) {
    if (str[i]==0)
      throw std::runtime_error("invalid uuid: " + std::string(str));
    if ((i==8 || i==13 || i==18 || i==23)) {
      if (str[i] != '-')
        throw std::runtime_error("invalid uuid: " + std::string(str));
    }
    else if (!std::isxdigit(str[i]))
      throw std::runtime_error("invalid uuid: " + std::string(str));
  }

  for (int i=0,u=0; i<uuid_str_sz; i+=2) {
    if (str[i] == '-')
      ++i;
    auto hi = std::stoi(std::string(1,str[i]),0,16);
    auto lo = std::stoi(std::string(1,str[i+1]),0,16);
    uuid[u++] = (hi << 4) | lo;
  }

#if 0
  char cmp[uuid_str_sz+1] = {0};
  uuid_unparse_lower(uuid, cmp);
  if (strncmp(str, cmp, 36) != 0)
    throw std::runtime_error("uuid_parse error (" + std::string(str) + ")(" + cmp + ")");
#endif
}

#endif 

#pragma warning ( pop )
#endif
