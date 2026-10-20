"""Relaunch unchanged Raygun science after restoring the original import paths."""
import argparse
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha
import recomb_confirm_v2_general_raygun as original

BASE=Path('reports/recomb_confirm_v2_General100_raygun_v2')


def main(a):
    f=read(a.root/'configs/recomb_confirm_v2_General100_raygun_dispatch_v3.json')
    if sha(a.root/original.FZ)!=f['parent_science_freeze_sha256']:raise ValueError('Parent science changed')
    for p,h in f['files'].items():
        if sha(a.root/p)!=h:raise ValueError('Technical relaunch source/input differs')
    parent=read(a.root/original.FZ)
    for name in ['raygun_tasks.json','raygun_checkpoint_load_contract.json']:
        source=str(original.BASE/name)
        if sha(a.root/BASE/name)!=parent['files'][source]:raise ValueError('Relaunch tasks not byte identical')
    original.BASE=BASE
    original.run(a.root,a.shard_index,a.shards)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shard-index',type=int,required=True);p.add_argument('--shards',type=int,default=4)
    main(p.parse_args())
