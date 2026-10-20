"""Read-only runtime accounting across both structure families and actual Slurm slots.

Keep the original General-only snapshot helper intact. Timing-only E8 model
calls are not counted as structure predictions. No inputs or jobs are changed.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')


def journal_snapshot(path):
    latest = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                latest[row['task_id'], row['attempt']] = row
    counts = Counter(row['state'] for row in latest.values())
    return dict(states=dict(counts), actual_started=sum(counts[k] for k in ['inflight', 'succeeded', 'failed']),
                successful=counts['succeeded'], failed=counts['failed'], inflight=counts['inflight'],
                reserved_not_started=counts['reserved'])


def main():
    general = journal_snapshot(ROOT/'reports/recomb_confirm_v2_General100_structures/prediction_journal.jsonl')
    application = journal_snapshot(ROOT/'reports/recomb_confirm_v2_application_structures/prediction_journal.jsonl')
    hardware = journal_snapshot(ROOT/'reports/recomb_confirm_v2_application_structures_hardware_v3/prediction_journal.jsonl')
    for key in ['actual_started', 'successful', 'failed', 'inflight', 'reserved_not_started']:
        application[key] += hardware[key]
    states = Counter(application['states']); states.update(hardware['states'])
    application['states'] = dict(states)
    application['hardware_v3'] = hardware
    raw = subprocess.check_output(['/opt/gridview/slurm/bin/squeue', '-h', '-u', 'chengshiz',
                                   '-o', '%i|%j|%T|%b'], text=True)
    jobs = []
    for line in raw.splitlines():
        jid, name, state, gres = line.split('|')
        if name.startswith('recomb-v2-') and 'gpu:' in gres.lower():
            jobs.append(dict(job_id=jid, parent_id=int(jid.split('_')[0]), name=name,
                             state=state, requested_TRES=gres))
    running = [r for r in jobs if r['state'] == 'RUNNING']
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(), clock_origin='cu9 UTC',
                  **general, General=general, application=application,
                  app_Raygun_conditions=len(list((ROOT/'reports/recomb_confirm_v2_application_baselines/raygun').glob('condition_*.json'))),
                  General_main_job=1119261, General_extended_job=1119263, General_second_job=1119264,
                  historical_before_v2=1177, v2_before_confirmation=160,
                  v2_actual_started_total=160+general['actual_started']+application['actual_started'],
                  v2_successful_total=160+general['successful']+application['successful'],
                  new_structure_failed_total=general['failed']+application['failed'],
                  new_structure_inflight_total=general['inflight']+application['inflight'],
                  cumulative_actual_started_total=1337+general['actual_started']+application['actual_started'],
                  task_GPU_jobs=jobs, task_GPU_running_count=len(running),
                  task_GPU_running_parent_jobs=sorted({r['parent_id'] for r in running}),
                  task_GPU_reserved_parent_jobs=sorted({r['parent_id'] for r in jobs}),
                  GPU_allocation_note='Each current project worker requests1GPU; array pending rows share its frozen dispatch throttle, not extra concurrent slots',
                  timing_only_E8_calls_not_in_structure_ledger=True)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
