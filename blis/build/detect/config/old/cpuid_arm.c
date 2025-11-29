/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2015, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <stdio.h>
#include <string.h>

#define CPUNAME_REFERENCE     	0
#define CPUNAME_ARMV7       	1
#define CPUNAME_CORTEXA9       	2
#define CPUNAME_CORTEXA15      	3

static char *cpuname[] = {
  "reference",
  "armv7a",
  "cortex-a9",
  "cortex-a15",
};


int get_feature(char *search)
{
  FILE *infile;
  char buffer[2048], *p,*t;
  p = (char *) NULL;

  infile = fopen("/proc/cpuinfo", "r");
  if (infile == NULL) {
    return 0;
  }

  while (fgets(buffer, sizeof(buffer), infile)) {
    if (!strncmp("Features", buffer, 8)) {
      p = strchr(buffer, ':') + 2;
      break;
    }
  }
  fclose(infile);

  if( p == NULL ) return 0;

  t = strtok(p," ");
  if (t != NULL) {
    if (!strcmp(t, search)) { return 1; }
  }
  while( t = strtok(NULL," ")){
    if (!strcmp(t, search)) { return 1; }
  }

  return 0;
}

int cpu_detect(void)
{
  FILE *infile;
  char buffer[512], *p;
  p = (char *) NULL ;

  infile = fopen("/proc/cpuinfo", "r");
  if (infile == NULL) {
    return CPUNAME_REFERENCE;
  }
  while (fgets(buffer, sizeof(buffer), infile)) {
    if (!strncmp("CPU part", buffer, 8)) {
      p = strchr(buffer, ':') + 2;
      break;
    }
  }
  fclose(infile);

  if(p != NULL) {
    if (strstr(p, "0xc09")) {
      if(get_feature("neon"))
	return CPUNAME_CORTEXA9;
      else
	return CPUNAME_ARMV7;
    }
    if (strstr(p, "0xc0f")) {
      if(get_feature("neon"))
	return CPUNAME_CORTEXA15;
      else
	return CPUNAME_ARMV7;
    }
  }

  p = (char *) NULL ;
  infile = fopen("/proc/cpuinfo", "r");
  if (infile == NULL) {
    return CPUNAME_REFERENCE;
  }

  while (fgets(buffer, sizeof(buffer), infile)) {
    if ((!strncmp("model name", buffer, 10)) || (!strncmp("Processor", buffer, 9))) {
      p = strchr(buffer, ':') + 2;
      break;
    }
  }
  fclose(infile);

  if(p != NULL) {
    if (strstr(p, "ARMv7")) {
      return CPUNAME_ARMV7;
    }
  }

  return CPUNAME_REFERENCE;
}

int main()
{
  int cpuname_id;

  cpuname_id=cpu_detect();
  printf("%s\n", cpuname[cpuname_id]);
  return 0;
}
