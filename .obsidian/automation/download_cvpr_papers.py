#!/usr/bin/env python3
"""Download or repair the open-access CVPR paper library.

Resolution is delegated to backfill_paper_library.py so DOI/arXiv/publisher
landing pages and verified repository overrides are handled consistently.
"""
from pathlib import Path
import csv, subprocess, sys
VAULT = Path(__file__).resolve().parents[2]
MANIFEST = VAULT / '99 Assets/Bibliography/CVPR 2027 — PDF Download Manifest.tsv'

result = subprocess.run([
    sys.executable,
    str(VAULT / '.obsidian/automation/backfill_paper_library.py'),
    '--workers', '4',
], cwd=VAULT)
if result.returncode:
    raise SystemExit(result.returncode)

failures=[]
for row in csv.DictReader(MANIFEST.open(), delimiter='\t'):
    target=VAULT / row['pdf']
    if not (target.exists() and target.stat().st_size > 10_000 and target.read_bytes()[:5] == b'%PDF-'):
        failures.append(str(target.relative_to(VAULT)))
if failures:
    print('\nFailed:', *failures, sep='\n- ', file=sys.stderr)
    raise SystemExit(1)
print('\nLocal CVPR paper library is complete.')
