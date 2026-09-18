#!/usr/bin/env python3
"""Build once externally, run W2A16/W2A8 comparisons and collect evidence. Stdlib only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

def cases_for(suite: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        dict(name='smoke-pair-f32', k=256, m=16, experts=8, tokens=1, pair=1, per=0, split=1, input='f32', pattern='normal'),
        dict(name='smoke-down-q4-f16-s2', k=512, m=16, experts=8, tokens=4, pair=0, per=1, split=2, input='f16', pattern='normal'),
        dict(name='smoke-zero', k=256, m=16, experts=8, tokens=1, pair=1, per=0, split=1, input='f32', pattern='zeros'),
        dict(name='smoke-rne-ties', k=512, m=16, experts=8, tokens=1, pair=1, per=0, split=2, input='f32', pattern='ties'),
    ]
    if suite in ('quick', 'full'):
        for kind in ('gate-up', 'down'):
            cases.append(dict(name=f'maple-{kind}-q1-s1', k=2048 if kind=='gate-up' else 512,
                m=512 if kind=='gate-up' else 2048, experts=256, tokens=1, pair=int(kind=='gate-up'),
                per=int(kind=='down'), split=1, input='f32', pattern='normal', model_kind=kind))
    if suite == 'full':
        for kind, token, split in [('gate-up',1,2),('gate-up',1,4),('gate-up',1,8),('down',1,2),
                                    ('gate-up',4,1),('down',4,1),('gate-up',8,2),('down',8,2)]:
            cases.append(dict(name=f'maple-{kind}-q{token}-s{split}', k=2048 if kind=='gate-up' else 512,
                m=512 if kind=='gate-up' else 2048, experts=256, tokens=token, pair=int(kind=='gate-up'),
                per=int(kind=='down'), split=split, input='f32', pattern='normal', model_kind=kind))
        cases.append(dict(name='smoke-outliers-q8',k=512,m=64,experts=256,tokens=8,pair=1,per=0,split=2,input='f32',pattern='outliers'))
        for local in (1,8):
            cases.append(dict(name=f'maple-gate-up-local{local}',k=2048,m=512,experts=256,tokens=1,pair=1,per=0,split=1,input='f32',pattern='normal',model_kind='gate-up',local=local))
    return cases

def case_args(case: dict[str, Any], capsules: dict[str, Path] | None = None) -> list[str]:
    args=['--k',str(case['k']),'--m',str(case['m']),'--experts',str(case['experts']),
          '--tokens',str(case['tokens']),'--topk','8','--per-selection',str(case['per']),
          '--split',str(case['split']),'--input',case['input'],'--x-pattern',case['pattern'],
          '--local',str(case.get('local',4))]
    if capsules and case.get('model_kind'):
        if case['model_kind']=='gate-up':
            args+=['--weights',str(capsules['gate']),'--weights2',str(capsules['up'])]
        else:
            args+=['--weights',str(capsules['down'])]
    else:
        args+=['--synthetic-pair',str(case['pair'])]
    return args

def run_logged(cmd: list[str], logfile: Path) -> None:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print('+',subprocess.list2cmdline(cmd),flush=True)
    with logfile.open('w',encoding='utf-8') as out:
        out.write('COMMAND: '+subprocess.list2cmdline(cmd)+'\n');out.flush()
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        assert proc.stdout is not None
        for line in proc.stdout:
            out.write(line);out.flush();print(line,end='',flush=True)
        code=proc.wait()
    if code:
        raise RuntimeError(f'Exit {code}: {cmd[0]}. Preserve {logfile}.')

def collect(case: dict[str, Any], directory: Path) -> list[dict[str, Any]]:
    with (directory/'summary.csv').open(newline='',encoding='utf-8') as inp:
        rows=list(csv.DictReader(inp))
    if not rows:
        raise RuntimeError(f'No summary rows for {case["name"]}')
    for row in rows:
        if row.get('kernel_pass')!='1' or row.get('quant_pass')!='1':
            raise RuntimeError(f'Numerical failure in {case["name"]}: {row.get("variant")}')
        row.update(case=case['name'],k=case['k'],m=case['m'],tokens=case['tokens'],per_selection=case['per'],pair=case['pair'])
    return rows

def write_all(rows: list[dict[str, Any]], dest: Path) -> None:
    if not rows:
        return
    with dest.open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main(argv: list[str] | None = None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe',type=Path,default=ROOT/'build'/('maple-w2-compare.exe' if os.name=='nt' else 'maple-w2-compare'))
    ap.add_argument('--suite',choices=('smoke','quick','full'),default='quick')
    ap.add_argument('--out',type=Path,default=ROOT/'build/results/suite')
    ap.add_argument('--device',default='A750')
    ap.add_argument('--model',type=Path)
    ap.add_argument('--layer',type=int,default=0)
    ap.add_argument('--repeats',type=int,default=28)
    ap.add_argument('--scrub-mib',type=int,default=0)
    ap.add_argument('--dry-run',action='store_true')
    ns=ap.parse_args(argv)
    if ns.layer<0 or not 1<=ns.repeats<=10000 or not 0<=ns.scrub_mib<=128:
        ap.error('invalid layer/repeats/scrub-mib')
    plan=cases_for(ns.suite)
    if ns.dry_run:
        print(json.dumps(plan,indent=2));return 0
    ns.exe=ns.exe.resolve();ns.out=ns.out.resolve()
    if not ns.exe.is_file():
        ap.error(f'Executable missing: {ns.exe}. Build with oneAPI first.')
    if ns.model and not ns.model.is_file():
        ap.error(f'Model missing: {ns.model}')
    if (ns.out/'suite_status.json').exists():
        ap.error('Output already contains a suite; use a new --out to preserve prior evidence.')
    ns.out.mkdir(parents=True,exist_ok=True)
    with ns.exe.open('rb') as binary:
        digest=hashlib.sha256()
        for block in iter(lambda: binary.read(1024*1024), b''):
            digest.update(block)
        exe_hash=digest.hexdigest()
    status: dict[str, Any]={'version':'0.3','suite':ns.suite,'complete':False,'passed':False,
        'executable':str(ns.exe),'executable_sha256':exe_hash,
        'model':str(ns.model) if ns.model else None,'activation':'synthetic (use compare --x-file for captured traces)',
        'scope':'standalone SYCL microkernels; NOT Vulkan bridge or end-to-end Maple', 'cases':[]}
    rows: list[dict[str,Any]]=[]
    try:
        run_logged([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_*.py'],ns.out/'python-tests.log')
        run_logged([str(ns.exe),'--device',ns.device,'--probe-only','1'],ns.out/'native-int2-probe.log')
        caps: dict[str,Path]={}
        if ns.model and ns.suite!='smoke':
            for kind in ('gate','up','down'):
                caps[kind]=ns.out/f'weights/{kind}.mw2'
                caps[kind].parent.mkdir(parents=True,exist_ok=True)
                run_logged([sys.executable,str(ROOT/'tools/extract_tq2.py'),'--model',str(ns.model.resolve()),
                    '--tensor',f'blk.{ns.layer}.ffn_{kind}_exps.weight','--output',str(caps[kind])],ns.out/f'extract-{kind}.log')
        for case in plan:
            directory=ns.out/case['name']
            command=[str(ns.exe),'--device',ns.device,'--skip-probe','1','--out',str(directory),
                '--repeats',str(min(ns.repeats,5) if case['name'].startswith('smoke-') else ns.repeats),
                '--warmup','3','--scrub-mib',str(ns.scrub_mib),'--include-fma','1' if case['name']=='smoke-pair-f32' else '0']+case_args(case,caps)
            run_logged(command,ns.out/f'{case["name"]}.log')
            rows.extend(collect(case,directory));write_all(rows,ns.out/'all_summary.csv')
            status['cases'].append(dict(name=case['name'],pass_checks=True))
        status['complete']=True;status['passed']=True
        print(f'PASS. {len(plan)} cases; {len(rows)} variant summaries. {ns.out / "all_summary.csv"}')
        print('A8 precision-only error is reported, not certified as model-quality acceptable. No server files changed.')
        return 0
    except (OSError,RuntimeError,ValueError) as exc:
        status['error']=str(exc);print(f'FAIL: {exc}',file=sys.stderr);return 1
    finally:
        (ns.out/'suite_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')

if __name__=='__main__':
    raise SystemExit(main())
