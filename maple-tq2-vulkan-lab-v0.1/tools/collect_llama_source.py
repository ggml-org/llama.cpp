#!/usr/bin/env python3
"""Collect actual source files (including local changes), not a build directory."""
from pathlib import Path
import argparse,hashlib,json,subprocess,zipfile

SOURCE_EXT={'.c','.cc','.cpp','.cxx','.h','.hpp','.hxx','.cu','.cuh','.m','.mm','.metal','.comp','.glsl','.cl','.py','.cmake','.md','.rst','.txt','.toml','.yaml','.yml','.json','.sh','.bat','.cmd','.ps1','.in','.patch','.diff','.jinja'}
EXCLUDED={'bin','models','.git','.venv','venv','node_modules','__pycache__','.cache','dist','target','vendor-binaries','fixtures'}

def git(root,*args):
    try:return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.DEVNULL,timeout=15).strip()
    except (OSError,subprocess.SubprocessError):return None

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('llama-actual-source.zip'));a=ap.parse_args();root=a.root.resolve();out=a.out.resolve()
    required=['ggml/src/ggml-vulkan/ggml-vulkan.cpp','CMakeLists.txt','ggml/include/ggml.h']
    missing=[x for x in required if not (root/x).is_file()]
    if missing:ap.error('Not a llama.cpp source root. Missing: '+', '.join(missing)+'. Use C:\\AI\\llama-src, not its build-sycl-vulkan child.')
    if out.exists():ap.error('Output already exists; choose a new --out path')
    files=[]
    for p in root.rglob('*'):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(root)
        if any(part in EXCLUDED or part.lower().startswith('build') or part.lower().startswith('cmake-build') for part in rel.parts[:-1]):continue
        if p.name.startswith('.env') or p.suffix.lower() not in SOURCE_EXT and p.name not in ('CMakeLists.txt','LICENSE','Makefile'):continue
        if p.name in ('compile_commands.json','CMakeCache.txt'):continue
        if p.stat().st_size>32*2**20:continue
        files.append((p,rel))
    out.parent.mkdir(parents=True,exist_ok=True)
    manifest={'source_root':str(root),'git_head':git(root,'rev-parse','HEAD'),'git_status':git(root,'status','--short'),'files':[]}
    try:
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p,rel in sorted(files,key=lambda x:str(x[1])):
                data=p.read_bytes();name=rel.as_posix();z.writestr('llama-source/'+name,data)
                manifest['files'].append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            z.writestr('SOURCE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    except Exception:
        out.unlink(missing_ok=True);raise
    print(f'Created {out} ({len(files)} source files, {out.stat().st_size/2**20:.2f} MiB)')
    print('Included current working-tree text, including untracked source. Review before sharing.')
if __name__=='__main__':main()
