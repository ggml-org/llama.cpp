#!/usr/bin/env python3
"""CMake/Linux helper: record current source and executable hashes AFTER rebuilding."""
import argparse
import json
from pathlib import Path
from run_grouping_suite import sources,sha256
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--exe',type=Path,required=True);a=p.parse_args()
if not a.exe.is_file(): p.error('executable not found')
r={'version':'0.4','source_sha256':sources(),'executable_sha256':sha256(a.exe),
   'compiler':'user-managed CMake build; consult CMakeCache.txt'}
(a.exe.parent/'grouping-build.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
