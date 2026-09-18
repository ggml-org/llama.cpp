#!/usr/bin/env python3
"""Fail closed on missing outputs, nonfinite values, status bits or numerical errors."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('fixture',type=Path)
    ap.add_argument('--rtol',type=float,default=3e-4);ap.add_argument('--atol',type=float,default=3e-4)
    a=ap.parse_args();root=a.fixture
    manifest_path=root/'manifest.json'
    m=json.loads(manifest_path.read_text());fail=[]
    receipt_path=root/'execution.json'
    backend=json.loads(receipt_path.read_text()).get('backend','unspecified') if receipt_path.exists() else 'UNKNOWN (missing execution receipt)'
    print('Validation backend:',backend)
    if not receipt_path.exists():raise SystemExit('FAIL: execution receipt missing')
    status_path=root/'control.actual.bin'
    if not status_path.exists():raise SystemExit('FAIL: shader outputs missing; references alone are NOT an execution pass')
    if status_path.stat().st_mtime_ns<manifest_path.stat().st_mtime_ns:raise SystemExit('FAIL: shader outputs predate the current fixture')
    status=np.fromfile(status_path,dtype='<u4')
    if len(status)!=4 or status[2]!=0:fail.append('shader status flags: '+str(status.tolist()))
    for name,meta in m['expected'].items():
        p=root/(name+'.actual.bin')
        if name=='logits' and m['full_router']:continue
        if not p.exists():fail.append('missing '+name);continue
        dtype=np.dtype(meta['dtype']);expected=np.fromfile(root/(name+'.expected.bin'),dtype=dtype)
        got=np.fromfile(p,dtype=dtype)
        if got.shape!=expected.shape:fail.append(name+' shape mismatch');continue
        if dtype.kind in 'ui':
            ok=np.array_equal(got,expected);info=f'{np.count_nonzero(got!=expected)} differing ids'
        else:
            finite=np.isfinite(got).all();ok=finite and np.allclose(got,expected,rtol=a.rtol,atol=a.atol)
            delta=got.astype(np.float64)-expected.astype(np.float64)
            info=f'max_abs={np.max(np.abs(delta)):.6g}, NMSE={np.sum(delta*delta)/(np.sum(expected.astype(np.float64)**2)+1e-30):.6g}'
        print(('PASS' if ok else 'FAIL'),name,info)
        if not ok:fail.append(name)
    if m['grouped']:
        try:
            ids=np.fromfile(root/'ids.actual.bin',dtype='<u4');s=np.fromfile(root/'sorted.actual.bin',dtype='<u4');o=np.fromfile(root/'offsets.actual.bin',dtype='<u4')
            if len(o)!=m['experts']+1 or len(s)!=len(ids) or o[-1]!=len(ids) or not np.array_equal(np.sort(s),np.arange(len(ids))):
                fail.append('bucket permutation/offsets')
            else:
                for e in range(m['experts']):
                    if not np.all(ids[s[o[e]:o[e+1]]]==e):fail.append('bucket expert mapping');break
        except (OSError,IndexError) as exc:fail.append('bucket output: '+str(exc))
    if fail:raise SystemExit('VALIDATION FAILED: '+', '.join(fail))
    print('All standalone shader fixture checks passed for '+backend+'. This does not establish llama-server/model-level correctness.')
if __name__=='__main__':main()
