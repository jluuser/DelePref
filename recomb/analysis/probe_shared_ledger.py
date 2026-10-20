"""Two-node CPU check of shared-file locks and reservation caps; fake calls only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time


def wait_for(path, timeout=30):
    start=time.monotonic()
    while not path.exists():
        if time.monotonic()-start>timeout:raise TimeoutError(str(path))
        time.sleep(.02)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['prepare','worker','summarize'],required=True);a=p.parse_args()
    root=Path(a.root).resolve();sys.path.insert(0,str(root/'code'))
    from common import AUTHORIZED_ROOT,write_json
    from budget_ledger import BudgetLedger,BudgetExceeded,_lock
    if root!=AUTHORIZED_ROOT.resolve():raise ValueError('unexpected test root')
    job=os.environ['SLURM_JOB_ID']
    if not job.isdigit():raise ValueError('invalid allocated job identity')
    work=root/'cpu_infrastructure_tests'/('multinode_'+job);work.mkdir(parents=True,exist_ok=True)
    protocol_path=work/'protocol.json';ledger_path=work/'fake_structure_calls.jsonl'
    if a.stage=='prepare':
        config=json.loads((root/'configs/protocol_v2.json').read_text())
        config['compute']['structure_call_limit']=16
        write_json(protocol_path,config);return
    if a.stage=='worker':
        rank=int(os.environ['SLURM_PROCID']);host=socket.gethostname()
        write_json(work/f'ready_{rank}.json',{'host':host,'rank':rank})
        for other in (0,1):wait_for(work/f'ready_{other}.json')
        lockpath=str(work/'cross_host_probe.lock')
        blocked=None
        if rank==0:
            with _lock(lockpath):
                (work/'holding').write_text('test-only')
                wait_for(work/'attempting')
                time.sleep(2)
            (work/'released').write_text('test-only')
        else:
            wait_for(work/'holding');(work/'attempting').write_text('test-only')
            begin=time.monotonic()
            with _lock(lockpath):blocked=time.monotonic()-begin
            if blocked<1.:raise RuntimeError('cross-host lock did not block while held')
        ledger=BudgetLedger(protocol_path,ledger_path);reserved=0;capped=0
        for i in range(80):
            task=f'test_{rank}_{i}'
            try:
                row=ledger.reserve(task_id=task,sequence_sha=hashlib.sha256(task.encode()).hexdigest(),
                    wt_variant='variant',predictor_weights_sha='a'*64,profile_sha='b'*64,
                    seed=20261013,category='development',job_id=job)
                reserved+=int(row['new_reservation'])
            except BudgetExceeded:capped+=1
        write_json(work/f'worker_{rank}.json',{'rank':rank,'host':host,'new_fake_reservations':reserved,
                   'capped_requests':capped,'blocked_seconds':blocked,'actual_predictor_calls':0})
        return
    rows=[json.loads((work/f'worker_{rank}.json').read_text()) for rank in (0,1)]
    if len({r['host'] for r in rows})!=2:raise RuntimeError('probe did not run on two distinct hosts')
    snapshot=BudgetLedger(protocol_path,ledger_path).snapshot()
    if snapshot['consumed']!=16 or sum(r['new_fake_reservations'] for r in rows)!=16:
        raise RuntimeError('concurrent reservation limit failed')
    report={'status':'CROSS_NODE_LEDGER_LOCK_VERIFIED','job_id':job,'hosts':[r['host'] for r in rows],
            'workers':rows,'test_cap':16,'fake_reservations':snapshot['consumed'],
            'actual_predictor_calls':0,'test_directory':str(work)}
    write_json(root/'reports/ledger_multinode_probe.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
