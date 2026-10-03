#!/usr/bin/env python3
"""Download and verify every release part, then reconstruct the receipt archive."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,default=Path('.runs/downloaded-receipts'))
args = parser.parse_args()
manifest=json.loads((Path(__file__).resolve().parents[1]/'results/artifact-sha256.json').read_text())
args.out.mkdir(parents=True,exist_ok=True)
archive=args.out/manifest['artifact']
temporary=archive.with_suffix(archive.suffix+'.tmp')
whole=hashlib.sha256()
with temporary.open('wb') as combined:
    for part in manifest['parts']:
        path=args.out/part['name']
        valid=path.exists() and path.stat().st_size==part['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==part['sha256']
        if not valid:
            partial=path.with_suffix(path.suffix+'.tmp')
            with urllib.request.urlopen(part['url'],timeout=600) as source,partial.open('wb') as target:
                while block:=source.read(1024*1024):target.write(block)
            if partial.stat().st_size!=part['bytes'] or hashlib.sha256(partial.read_bytes()).hexdigest()!=part['sha256']:
                raise ValueError('release part checksum differs: '+part['name'])
            partial.replace(path)
        with path.open('rb') as source:
            while block:=source.read(1024*1024):combined.write(block);whole.update(block)
        print('Verified '+part['name'],flush=True)
if temporary.stat().st_size!=manifest['bytes'] or whole.hexdigest()!=manifest['sha256']:
    raise ValueError('combined archive checksum differs')
temporary.replace(archive)
print('Verified full receipt archive: '+str(archive))
