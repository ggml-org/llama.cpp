#!/usr/bin/env python3
"""Compile the production mul_mat_vec reduction selection with small Vulkan type stubs.

This exercises the actual guard wiring and platform/option switches; it
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
source = source[source.index('void ggml_vk_load_shaders('):]
start = source.index('    bool use_subgroups = device->subgroup_arithmetic;')
end = source.index('#endif\n', start) + len('#endif\n')
selection = source[start:end]
stub = r'''
#include "ggml-vulkan-adreno-compat.hpp"
#include <array>
#include <cstring>
#include <memory>
struct properties {uint32_t vendorID=0x5143,driverVersion=2150604839u;std::array<char,256> deviceName{};};
struct device {struct properties properties;uint32_t driver_id=8;bool subgroup_arithmetic=true;};
using vk_device=std::shared_ptr<device>;
static bool dmmv_use_subgroups(vk_device & device) {
'''
ending = r'''
    return use_subgroups;
}
int main(){
 for(int mutation=0;mutation<9;++mutation){
  auto d=std::make_shared<device>();
  std::strcpy(d->properties.deviceName.data(),"Adreno (TM) 750");
  switch(mutation){
   case 1:d->properties.vendorID=0x1002;break;
   case 2:d->driver_id=1;break;
   case 3:d->properties.driverVersion--;break;
   case 4:d->properties.driverVersion++;break;
   case 5:std::strcpy(d->properties.deviceName.data(),"Adreno (TM) 740");break;
   case 6:d->properties.vendorID=0x10de;break;
   case 7:d->properties.vendorID=0x8086;break;
   case 8:d->subgroup_arithmetic=false;break;
  }
  const bool expected=mutation==8 ? false : !(EXPECT_ENABLED && mutation==0);
  if(dmmv_use_subgroups(d)!=expected)return 1;
 }
}
'''
with tempfile.TemporaryDirectory() as directory:
    directory=Path(directory);code=directory/'routing.cpp';exe=directory/'routing'
    code.write_text(stub+selection+ending)
    for name,defines,enabled in [
        ('android_on',['-D__ANDROID__','-DGGML_VULKAN_ADRENO_750_SHMEM'],1),
        ('android_off',['-D__ANDROID__','-UGGML_VULKAN_ADRENO_750_SHMEM'],0),
        ('non_android_on',['-U__ANDROID__','-DGGML_VULKAN_ADRENO_750_SHMEM'],0),
    ]:
        subprocess.run([args.compiler,'-std=c++17','-O2',f'-DEXPECT_ENABLED={enabled}',*defines,
            '-I'+str(root/'ggml/src/ggml-vulkan'),str(code),'-o',str(exe)],check=True)
        subprocess.run([str(exe)],check=True)
        print(f'PASS: {name},9 production mul_mat_vec reduction selection cases')
