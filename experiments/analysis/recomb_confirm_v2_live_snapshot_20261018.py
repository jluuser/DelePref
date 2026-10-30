"""Read-only runtime snapshot; no changes to prediction inputs or selection."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

root = Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
latest = {}
for line in (root/'reports/recomb_confirm_v2_General100_structures/prediction_journal.jsonl').open():
    if line.strip():
        row = json.loads(line)
        latest[row['task_id'], row['attempt']] = row
states = Counter(x['state'] for x in latest.values())
started = sum(states[k] for k in ('inflight', 'succeeded', 'failed'))
snapshot = dict(timestamp=datetime.now(timezone.utc).isoformat(), clock_origin='cu9 UTC',
    states=dict(states), actual_started=started, successful=states['succeeded'],
    failed=states['failed'], inflight=states['inflight'], reserved_not_started=states['reserved'],
    app_Raygun_conditions=len(list((root/'reports/recomb_confirm_v2_application_baselines/raygun').glob('condition_*.json'))),
    General_main_job=1119261, General_extended_job=1119263, General_second_job=1119264,
    historical_before_v2=1177, v2_before_confirmation=160,
    v2_actual_started_total=160+started, cumulative_actual_started_total=1337+started)
print(json.dumps(snapshot, ensure_ascii=False))
