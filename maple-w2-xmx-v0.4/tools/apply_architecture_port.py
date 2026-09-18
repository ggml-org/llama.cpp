# SPDX-License-Identifier: MIT
"""Apply only verified v0.4 module files. Default is a no-write preflight.
No clean-tree/reset/rebase requirement; unrelated llama.cpp changes are ignored.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
import uuid
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def safe_path(root, name):
    p=PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(s in ('..','.git') for s in p.parts) or '\\' in name:
        raise ValueError('unsafe relative path in patch metadata')
    dest=root.joinpath(*p.parts)
    for item in [dest,*dest.parents]:
        if item==root:break
        if item.is_symlink():raise ValueError('symlink in patch target: '+name)
    if root not in dest.resolve().parents:raise ValueError('path escapes module root')
    return dest

def apply(target, payload=ROOT, commit=False):
    target=target.resolve();payload=payload.resolve()
    if target==payload or not target.is_dir():raise ValueError('target must be an existing separate v0.4 module root')
    manifest=json.loads((payload/'PORT_PATCH_FILES.json').read_text(encoding='utf-8'))
    changes=[];skipped=0
    for item in manifest['files']:
        name=item['path'];src=safe_path(payload,name);dst=safe_path(target,name)
        if not src.is_file() or digest(src)!=item['new_sha256']:raise ValueError('payload identity mismatch: '+name)
        current=digest(dst) if dst.is_file() else None
        if current==item['new_sha256']:skipped+=1;continue
        if dst.exists() and not dst.is_file():raise ValueError('target is not a file: '+name)
        if current!=item['old_sha256']:raise ValueError('not the uploaded v0.4 content: '+name+'; inspect diff, do not force overwrite')
        changes.append((name,src,dst,current))
    report={'checked':len(manifest['files']),'changes':len(changes),'already_applied':skipped,'applied':False}
    if not commit or not changes:return report
    backup=target/'_w2_port_backups'/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    backup.mkdir(parents=True)
    written=[]
    try:
        for name,src,dst,old in changes:
            current=digest(dst) if dst.is_file() else None
            if current!=old:raise ValueError('target changed during apply: '+name)
            if old is not None:
                bp=safe_path(backup,name);bp.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,bp)
            dst.parent.mkdir(parents=True,exist_ok=True)
            fd,tmp=tempfile.mkstemp(prefix='.w2-port-',dir=dst.parent);os.close(fd)
            try:
                shutil.copy2(src,tmp);os.replace(tmp,dst)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
            written.append((name,dst,old))
    except BaseException:
        for name,dst,old in reversed(written):
            if old is None:dst.unlink(missing_ok=True)
            else:shutil.copy2(safe_path(backup,name),dst)
        raise
    report.update(applied=True,backup=str(backup))
    (backup/'apply-receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target',required=True,type=Path)
    ap.add_argument('--apply',action='store_true',help='write after all hashes pass; backups and rollback on failure')
    a=ap.parse_args()
    print(json.dumps(apply(a.target,commit=a.apply),indent=2))
if __name__=='__main__':main()
