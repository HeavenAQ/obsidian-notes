#!/usr/bin/env python3
"""Download or repair the open-access PDFs listed in the tracked CVPR manifest."""
from pathlib import Path
import csv, subprocess, sys
VAULT = Path(__file__).resolve().parents[2]
MANIFEST = VAULT / '99 Assets/Bibliography/CVPR 2027 — PDF Download Manifest.tsv'
failures=[]
for row in csv.DictReader(MANIFEST.open(), delimiter='\t'):
    target=VAULT / row['pdf']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes()[:4] == b'%PDF':
        print('OK     ', target.relative_to(VAULT))
        continue
    partial=target.with_suffix(target.suffix+'.part')
    print('FETCH  ', target.relative_to(VAULT))
    result=subprocess.run([
        'curl','-fL','--retry','4','--retry-delay','2','--connect-timeout','20',
        '--max-time','240','-A','Mozilla/5.0 research-library/1.0',
        '-o',str(partial),row['source']
    ])
    if result.returncode == 0 and partial.exists() and partial.read_bytes()[:4] == b'%PDF':
        partial.replace(target)
    else:
        partial.unlink(missing_ok=True)
        failures.append(str(target.relative_to(VAULT)))
if failures:
    print('\nFailed:', *failures, sep='\n- ', file=sys.stderr)
    raise SystemExit(1)
print('\nLocal CVPR paper library is complete.')
