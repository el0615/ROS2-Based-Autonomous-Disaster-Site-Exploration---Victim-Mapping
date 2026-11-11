#!/usr/bin/env python3
"""Zone-Based v2.4 add-only installer. Copies a new isolated folder; never patches the old manager.
Default is a read-only check. --install is explicit. Destination collisions fail closed.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import shutil, sys, tempfile
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True,type=Path)
    p.add_argument('--install',action='store_true')
    a=p.parse_args();project=a.project.expanduser().resolve()
    parent=project/'scripts/exploration';dest=parent/'zone_exploration'
    if not parent.is_dir():raise RuntimeError('Project scripts/exploration not found')
    manifest=json.loads((ROOT/'PACKAGE_SHA256.json').read_text())
    for name,h in manifest.items():
        f=ROOT/name
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:
            raise RuntimeError('Package checksum mismatch: '+name)
    if dest.exists():
        identical=all((dest/n).is_file() and hashlib.sha256((dest/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
        if identical:print('[ALREADY_INSTALLED]',dest);return 0
        raise RuntimeError('Destination exists with changed files. No overwrite performed: '+str(dest))
    print('[INSTALL_CHECK_OK] add-only destination:',dest)
    if not a.install:return 0
    temp=Path(tempfile.mkdtemp(prefix='.zone_based_install_',dir=parent))
    try:
        for name in [*manifest,'PACKAGE_SHA256.json']:
            target=temp/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        os.rename(temp,dest)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True);raise
    print('[ZONE_BASED_INSTALLED]',dest)
    print('[UNCHANGED] Original DN-011, Nav2 YAML, Coverage and maps were not edited.')
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as exc:
        print('[INSTALL_REFUSED]',exc,file=sys.stderr);raise SystemExit(2)
