#!/usr/bin/env python3
"""Compile the production selection prefix with small Vulkan type stubs.

This exercises actual pointer/length wiring and platform/option switches; it
neither compiles a Vulkan pipeline nor claims device execution.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--compiler', default='c++')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = (root / 'ggml/src/ggml-vulkan/ggml-vulkan.cpp').read_text()
start = source.index('static void ggml_vk_create_pipeline_func(')
end = source.index('    vk::ShaderModuleCreateInfo shader_module_create_info', start)
prefix = source[start:end]
stub = r'''
#include "ggml-vulkan-adreno-compat.hpp"
#include <array>
#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include <cstdlib>
#define VK_LOG_DEBUG(...) do {} while (0)
#define GGML_ASSERT(x) assert(x)
constexpr int MAX_PARAMETER_COUNT=16;
static const unsigned char mul_mat_vec_f32_f32_f32_subgroup_no_shmem_data[]={1,1};
static const unsigned char mul_mat_vec_f32_f32_f32_data[]={2,2,2};
static const size_t mul_mat_vec_f32_f32_f32_len=3;
struct properties {uint32_t vendorID=0x5143,driverVersion=2150604839u;
 std::array<char,64> deviceName{};struct{uint32_t maxComputeSharedMemorySize=32768;}limits;};
struct device {struct properties properties;uint32_t driver_id=8;};
struct pipeline {};
using vk_device=std::shared_ptr<device>;using vk_pipeline=std::shared_ptr<pipeline>;
const void * observed_data=nullptr;size_t observed_size=0;
'''
ending = r'''
    observed_data=spv_data;observed_size=spv_size;
}
int main(){
 for(int mutation=0;mutation<17;++mutation){
  auto d=std::make_shared<device>();auto p=std::make_shared<pipeline>();
  std::strcpy(d->properties.deviceName.data(),"Adreno (TM) 750");
  std::vector<uint32_t> specs={64,1,5};uint32_t subgroup=64;bool full=true;
  const void * input=mul_mat_vec_f32_f32_f32_subgroup_no_shmem_data;
  switch(mutation){
   case 1:d->properties.vendorID=0x1002;break;
   case 2:d->driver_id=1;break;
   case 3:d->properties.driverVersion--;break;
   case 4:std::strcpy(d->properties.deviceName.data(),"Adreno (TM) 740");break;
   case 5:specs[0]=32;break;
   case 6:specs[1]=2;break;
   case 7:specs[2]=1;break;
   case 8:specs[2]=6;break;
   case 9:subgroup=32;break;
   case 10:full=false;break;
   case 11:d->properties.limits.maxComputeSharedMemorySize=1279;break;
   case 12:specs.clear();break;
   case 13:specs.push_back(0);break;
   case 14:input=mul_mat_vec_f32_f32_f32_data;break;
   case 15:d->properties.vendorID=0x10de;break;
   case 16:d->properties.vendorID=0x8086;break;
  }
  ggml_vk_create_pipeline_func(d,p,2,input,"main",5,{64,1,1},specs,false,full,subgroup);
  const bool changed=EXPECT_ENABLED && mutation==0;
  if(observed_data!=(changed?mul_mat_vec_f32_f32_f32_data:input) || observed_size!=(changed?3u:2u))return 1;
 }
}
'''
with tempfile.TemporaryDirectory() as directory:
    directory=Path(directory);code=directory/'routing.cpp';exe=directory/'routing'
    code.write_text(stub+prefix+ending)
    for name,defines,enabled in [
        ('android_on',['-D__ANDROID__','-DGGML_VULKAN_ADRENO_750_SHMEM'],1),
        ('android_off',['-D__ANDROID__','-UGGML_VULKAN_ADRENO_750_SHMEM'],0),
        ('non_android_on',['-U__ANDROID__','-DGGML_VULKAN_ADRENO_750_SHMEM'],0),
    ]:
        subprocess.run([args.compiler,'-std=c++17','-O2',f'-DEXPECT_ENABLED={enabled}',*defines,
            '-I'+str(root/'ggml/src/ggml-vulkan'),str(code),'-o',str(exe)],check=True)
        subprocess.run([str(exe)],check=True)
        print(f'PASS: {name},17 production call-site routing cases')
