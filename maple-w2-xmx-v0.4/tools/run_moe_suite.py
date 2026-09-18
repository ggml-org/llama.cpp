#!/usr/bin/env python3
"""MoE chain checks and A16/A8 wall-time comparison. No server modification."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
from run_suite import ROOT, run_logged

def cases_for(suite: str) -> list[dict[str, Any]]:
    cases = [dict(name='moe-smoke-q1', k=256, hidden=256, experts=8, tokens=1, gate_split=1, down_split=1, order=1),
             dict(name='moe-smoke-q4-ooo', k=256, hidden=256, experts=8, tokens=4, gate_split=1, down_split=1, order=0)]
    if suite in ('quick','full'):
        cases.append(dict(name='moe-maple-q1', k=2048, hidden=512, experts=256, tokens=1, gate_split=1, down_split=1, order=1, real=True))
    if suite == 'full':
        cases += [dict(name='moe-maple-q4',k=2048,hidden=512,experts=256,tokens=4,gate_split=1,down_split=1,order=1,real=True),
                  dict(name='moe-maple-q1-split',k=2048,hidden=512,experts=256,tokens=1,gate_split=4,down_split=2,order=1,real=True),
                  dict(name='moe-maple-q1-g32-h128',k=2048,hidden=512,experts=256,tokens=1,gate_split=1,down_split=1,order=1,real=True,groups='32',down_group=128),
                  dict(name='moe-smoke-outlier-q8',k=256,hidden=256,experts=8,tokens=8,gate_split=1,down_split=1,order=1,pattern='outliers')]
    return cases

def case_args(case: dict[str,Any], caps: dict[str,Path], repeats: int, scrub: int) -> list[str]:
    args=['--k',str(case['k']),'--hidden',str(case['hidden']),'--experts',str(case['experts']),
          '--tokens',str(case['tokens']),'--topk','8','--layout','s2tile8','--groups',case.get('groups','32,128,256'),
          '--host-waits','both','--gate-split',str(case['gate_split']),'--down-split',str(case['down_split']),
          '--local','4','--in-order',str(case['order']),'--pattern',case.get('pattern','normal'),
          '--warmup','3','--repeats',str(min(repeats,5) if 'smoke' in case['name'] else repeats),
          '--scrub-mib',str(scrub),'--skip-probe','0']
    if case.get('down_group'):
        args+=['--down-group',str(case['down_group'])]
    if case.get('real') and caps:
        for key in ('gate','up','down'):
            args += [f'--{key}',str(caps[key])]
    return args

def main(argv: list[str] | None=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe',type=Path,default=ROOT/'build'/('maple-moe-compare.exe' if os.name=='nt' else 'maple-moe-compare'))
    ap.add_argument('--suite',choices=('smoke','quick','full'),default='quick')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='A750')
    source=ap.add_mutually_exclusive_group();source.add_argument('--model',type=Path);source.add_argument('--capsule-dir',type=Path)
    ap.add_argument('--layer',type=int,default=0);ap.add_argument('--repeats',type=int,default=28)
    ap.add_argument('--scrub-mib',type=int,default=0);ap.add_argument('--dry-run',action='store_true')
    ns=ap.parse_args(argv)
    if ns.layer<0 or not 1<=ns.repeats<=10000 or not 0<=ns.scrub_mib<=128:ap.error('invalid layer/repeats/scrub')
    plan=cases_for(ns.suite)
    if ns.dry_run:
        print(json.dumps(plan,indent=2));return 0
    if not ns.exe.is_file():ap.error(f'missing executable: {ns.exe}')
    if ns.model and not ns.model.is_file():ap.error(f'missing GGUF: {ns.model}')
    if (ns.out/'suite_status.json').exists():ap.error('use a fresh output directory')
    ns.out.mkdir(parents=True,exist_ok=True)
    status: dict[str,Any]={'version':'0.3','suite':ns.suite,'passed':False,'complete':False,'cases':[],
        'executable_sha256':hashlib.sha256(ns.exe.read_bytes()).hexdigest(),
        'scope':'one MoE layer; excludes attention/router/norm/residual/Vulkan interop/server TG',
        'activation_and_routes':'synthetic; direct moe_compare CLI accepts captured files'}
    rows=[]
    try:
        caps={}
        if ns.capsule_dir:
            for key in ('gate','up','down'):
                caps[key]=(ns.capsule_dir/f'{key}.mw2').resolve()
                if not caps[key].is_file():raise ValueError(f'missing capsule: {caps[key]}')
        elif ns.model and ns.suite!='smoke':
            for key in ('gate','up','down'):
                caps[key]=(ns.out/f'weights/{key}.mw2').resolve();caps[key].parent.mkdir(parents=True,exist_ok=True)
                run_logged([sys.executable,str(ROOT/'tools/extract_tq2.py'),'--model',str(ns.model.resolve()),
                    '--tensor',f'blk.{ns.layer}.ffn_{key}_exps.weight','--output',str(caps[key])],ns.out/f'extract-{key}.log')
        status['capsules']={key:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for key,p in caps.items()}
        for case in plan:
            directory=(ns.out/case['name']).resolve()
            cmd=[str(ns.exe.resolve()),'--device',ns.device,'--out',str(directory)]+case_args(case,caps,ns.repeats,ns.scrub_mib)
            run_logged(cmd,ns.out/f'{case["name"]}.log')
            with (directory/'summary.csv').open(newline='',encoding='utf-8') as inp:new=list(csv.DictReader(inp))
            if not new or any(row.get('kernel_pass')!='1' for row in new):raise RuntimeError(f'failed chain: {case["name"]}')
            for row in new:row.update(case=case['name'],tokens=case['tokens'],k=case['k'],hidden=case['hidden'],source='real_weights' if caps and case.get('real') else 'synthetic')
            rows.extend(new)
            with (ns.out/'all_moe_summary.csv').open('w',newline='',encoding='utf-8') as dest:
                wr=csv.DictWriter(dest,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
            status['cases'].append({'name':case['name'],'passed':True})
        status['passed']=True;status['complete']=True
        print(f'PASS: {len(plan)} MoE cases / {len(rows)} variants. {ns.out / "all_moe_summary.csv"}')
        print('No Vulkan handoff measured, no model-quality certificate, no server changes.')
        return 0
    except (OSError,ValueError,RuntimeError) as exc:
        status['error']=str(exc);print(f'FAIL: {exc}',file=sys.stderr);return 1
    finally:
        (ns.out/'suite_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
if __name__=='__main__':raise SystemExit(main())
