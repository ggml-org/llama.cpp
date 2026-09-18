#!/usr/bin/env python3
"""v0.4: paired baseline/grouping Q sweep. Exactly one optimization changes."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any
from run_suite import ROOT, run_logged

QS = (1,2,4,8,16,32,64,128,256,512,1024,2048)

def source_files(root: Path=ROOT) -> list[Path]:
    return sorted([*root.glob('include/*.hpp'), *root.glob('src/*.cpp'),root/'tools/grouping_compare.cpp'])

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(4*1024*1024): h.update(chunk)
    return h.hexdigest()

def sources(root: Path=ROOT) -> dict[str,str]:
    return {p.relative_to(root).as_posix():sha256(p) for p in source_files(root)}

def cases_for(suite: str) -> list[dict[str,Any]]:
    cases=[dict(name='group-smoke-q1',tokens=1,k=256,hidden=256,experts=8,order=1),
           dict(name='group-smoke-q13-ooo',tokens=13,k=256,hidden=256,experts=8,order=0)]
    qs=() if suite=='smoke' else (1,4,64,256,2048) if suite=='quick' else QS
    return cases+[dict(name=f'q{q:04d}',tokens=q,k=2048,hidden=512,experts=256,order=1,model_shape=True) for q in qs]

def case_args(c:dict[str,Any], caps:dict[str,Path], repeats:int, scrub:int) -> list[str]:
    args=['--tokens',str(c['tokens']),'--k',str(c['k']),'--hidden',str(c['hidden']),'--experts',str(c['experts']),
          '--topk','8','--gate-split','1','--down-split','1','--local','4',
          '--in-order',str(c['order']),'--repeats',str(min(4,repeats) if not c.get('model_shape') else repeats),
          '--warmup','3','--pattern','normal','--id-mode','rotate','--scrub-mib',str(scrub),'--skip-probe','1']
    if c.get('model_shape') and caps:
        for key,path in caps.items(): args.extend([f'--{key}',str(path)])
    return args

def verify_build(exe: Path) -> dict[str,Any]:
    path=exe.parent/'grouping-build.json'
    if not path.is_file(): raise ValueError(f'missing build manifest {path}; rebuild v0.4 first')
    built=json.loads(path.read_text(encoding='utf-8-sig'))
    if built.get('version')!='0.4': raise ValueError('not a v0.4 build')
    if built.get('source_sha256')!=sources(): raise ValueError('source changed after build; do not use -NoBuild')
    if built.get('executable_sha256')!=sha256(exe): raise ValueError('executable changed after build')
    return built

def paired_stats(path:Path) -> dict[str,Any]:
    with path.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    rounds:dict[str,dict[str,float]]={}
    for row in rows:
        if row['phase']!='repeat': continue
        arm='grouped' if row['variant'].endswith('expert-grouped') else 'baseline'
        if arm in rounds.setdefault(row['round'],{}): raise ValueError('duplicate arm in round')
        rounds[row['round']][arm]=float(row['wall_us'])
    if not rounds or any(set(r)!= {'baseline','grouped'} for r in rounds.values()): raise ValueError('unpaired samples')
    delta=[r['baseline']-r['grouped'] for r in rounds.values()]
    return dict(paired_rounds=len(delta),grouped_wins=sum(x>0 for x in delta),paired_saved_median_us=statistics.median(delta),
        paired_speedup_median=statistics.median(r['baseline']/r['grouped'] for r in rounds.values()))

def write_csv(path:Path,rows:list[dict[str,Any]]) -> None:
    if not rows: return
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(argv:list[str]|None=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe',type=Path,default=ROOT/'build'/('maple-grouping-compare.exe' if os.name=='nt' else 'maple-grouping-compare'))
    ap.add_argument('--suite',choices=('smoke','quick','full'),default='full');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='A750');ap.add_argument('--repeats',type=int,default=28);ap.add_argument('--scrub-mib',type=int,default=0)
    ap.add_argument('--layer',type=int,default=0);ap.add_argument('--dry-run',action='store_true')
    group=ap.add_mutually_exclusive_group();group.add_argument('--model',type=Path);group.add_argument('--capsule-dir',type=Path);group.add_argument('--synthetic',action='store_true')
    ns=ap.parse_args(argv)
    if ns.layer<0 or not 2<=ns.repeats<=10000 or not 0<=ns.scrub_mib<=128: ap.error('invalid layer/repeats/scrub')
    plan=cases_for(ns.suite)
    if ns.dry_run:
        print(json.dumps({'version':'0.4','only_change':'per-expert job grouping','arms':2,'fixed':'s2tile8 G32/H32 gluquant async split1/1 local4','cases':plan},indent=2));return 0
    if ns.suite!='smoke' and not(ns.model or ns.capsule_dir or ns.synthetic): ap.error('provide --model/--capsule-dir, or explicitly --synthetic')
    if not ns.exe.is_file(): ap.error(f'missing executable: {ns.exe}')
    if (ns.out/'suite_status.json').exists(): ap.error('choose fresh output directory')
    ns.out.mkdir(parents=True,exist_ok=True)
    rows=[];speeds=[]
    status:dict[str,Any]={'version':'0.4','complete':False,'passed':False,'cases':[],
        'scope':'one layer, grouping included EVERY call; no multi-token GEMM, no Vulkan or server TG',
        'source_note':'based on attached v0.3 source; local Q-sweep source diff was not supplied'}
    try:
        built=verify_build(ns.exe);status['build']=built
        run_logged([str(ns.exe.resolve()),'--device',ns.device,'--probe-only','1'],ns.out/'probe.log')
        caps={}
        if ns.capsule_dir:
            for key in ('gate','up','down'):
                path=(ns.capsule_dir/f'{key}.mw2').resolve()
                if not path.is_file(): raise ValueError(f'missing capsule {path}')
                caps[key]=path
        elif ns.model and ns.suite!='smoke':
            if not ns.model.is_file(): raise ValueError(f'missing model {ns.model}')
            for key in ('gate','up','down'):
                path=(ns.out/f'weights/{key}.mw2').resolve();path.parent.mkdir(parents=True,exist_ok=True)
                run_logged([sys.executable,str(ROOT/'tools/extract_tq2.py'),'--model',str(ns.model.resolve()),'--tensor',f'blk.{ns.layer}.ffn_{key}_exps.weight','--output',str(path)],ns.out/f'extract-{key}.log')
                caps[key]=path
        status['capsules']={k:dict(path=str(p),sha256=sha256(p)) for k,p in caps.items()}
        (ns.out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
        for c in plan:
            directory=(ns.out/c['name']).resolve()
            cmd=[str(ns.exe.resolve()),'--device',ns.device,'--out',str(directory)]+case_args(c,caps,ns.repeats,ns.scrub_mib)
            run_logged(cmd,ns.out/f'{c["name"]}.log')
            with (directory/'summary.csv').open(newline='',encoding='utf-8') as f: new=list(csv.DictReader(f))
            if len(new)!=2 or {r['expert_grouping'] for r in new}!={'0','1'} or any(r['kernel_pass']!='1' or r['pair_pass']!='1' for r in new):
                raise ValueError(f'invalid/failed two-arm summary: {c["name"]}')
            for r in new:r['case']=c['name']
            rows.extend(new);write_csv(ns.out/'all_grouping_summary.csv',rows)
            b=next(r for r in new if r['expert_grouping']=='0');g=next(r for r in new if r['expert_grouping']=='1')
            speeds.append(dict(case=c['name'],Q=c['tokens'],source=g['source'],baseline_us=b['wall_median_us'],grouped_us=g['wall_median_us'],
                speedup=g['speedup_vs_ungrouped_same_build'],group_kernel_us=g['group_kernel_median_us'],
                baseline_p95_us=b['wall_p95_us'],grouped_p95_us=g['wall_p95_us'],**paired_stats(directory/'samples.csv')))
            write_csv(ns.out/'q_speedup.csv',speeds)
            status['cases'].append({'name':c['name'],'passed':True})
        status['passed']=True;status['complete']=True
        print(f'PASS {len(plan)} cases/{len(rows)} variants. Read {ns.out / "q_speedup.csv"}.')
        print('No performance winner or automatic cutover is preselected. Group construction is included. Not server TG.')
        return 0
    except (OSError,ValueError,RuntimeError) as exc:
        status['error']=str(exc);print(f'FAIL {exc}',file=sys.stderr);return 1
    finally:
        (ns.out/'suite_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
if __name__=='__main__':raise SystemExit(main())
