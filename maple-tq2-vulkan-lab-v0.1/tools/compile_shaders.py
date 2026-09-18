#!/usr/bin/env python3
"""Compile every supplied GLSL variant; fail on errors, never mark missing tools PASS."""
from pathlib import Path
import argparse,shutil,subprocess

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--glslc',default=shutil.which('glslc'))
    ap.add_argument('--output',type=Path,default=Path('build/shaders'))
    ap.add_argument('--spirv-val',default=shutil.which('spirv-val'))
    a=ap.parse_args()
    if not a.glslc: ap.error('glslc is unavailable; install Vulkan SDK or pass --glslc')
    src=Path(__file__).resolve().parents[1]/'shaders'
    a.output.mkdir(parents=True,exist_ok=True)
    variants=[]
    for f in src.glob('*.comp'):
        if f.stem=='project':
            for q8 in (0,1):
                for fusion in (0,1,2):
                    for reuse in (0,1):
                        variants.append((f,f'project_q{q8}_f{fusion}_r{reuse}',[f'-DQ8={q8}',f'-DFUSION={fusion}',f'-DREUSE={reuse}']))
        else: variants.append((f,f.stem,[]))
    for source,name,defs in variants:
        dst=a.output/(name+'.spv')
        cmd=[a.glslc,'--target-env=vulkan1.2','-O','-I',str(src),*defs,str(source),'-o',str(dst)]
        print('COMPILE',name,flush=True)
        subprocess.run(cmd,check=True)
        if a.spirv_val: subprocess.run([a.spirv_val,'--target-env','vulkan1.2',str(dst)],check=True)
    print(f'Compiled {len(variants)} variants; SPIR-V validation: {bool(a.spirv_val)}')
if __name__=='__main__': main()
