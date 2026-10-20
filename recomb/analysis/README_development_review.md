# CPU development artifact review

This engineering report implements the authorized development-review subtask.
It reads existing artifacts, imports the verified frozen `paired_statistics.py`,
and never deserializes a checkpoint, imports torch, runs inference, trains,
folds, publishes, or modifies `paper/` or the frozen computational code.

Implementation sequence (completed with synthetic tests before implementation):

- [x] Exercise missing-run, corrupt-hash, missing-candidate, duplicate-decision,
  earliest-minimum, LR-tie, and hierarchical-pairing failures.
- [x] Validate the frozen inputs, snapshot, every completed run, its traces,
  selected validation artifacts, and all primary validation WTs/candidates.
- [x] Select the lower numerical LR on an exact minimum-CE tie; estimate all
  C-B, C-S, C-A, C-D diagnostics with the frozen paired bootstrap.
- [x] Keep every expected WT within every reported source/budget stratum;
  undefined bounds or absent pools yield explicit undefined strata, never
  a complete-case bootstrap.
- [x] Write JSON, per-WT inputs, CSV, and an ENGINEERING Markdown report;
  self-review the implementation and run the CPU test suite.

The frozen primary manifest supplies the WT roster: currently 10 validation WTs
and 420 candidates. The global split also includes the sensitivity cohort and
has 48 validation WTs; those extra WTs are not silently imported into primary
comparisons. CE within WT balances budget, generator, then candidate; the paired
bootstrap subsequently averages WTs within homology components and components
equally. T/G diagnostics stay separate by budget and historical generator pool
(including each mixed pool), and use historical That-selected decisions. They
are not matched G-selected results or independent confirmation. Selection and
estimation share validation data. No scientific pass decision is produced.

From the experiment root on cu9:

```sh
PY=/public/home/zhangyangroup/chengshiz/keyuan.zhou/evocompress/.venv-v7/bin/python
ROOT=/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013
"$PY" analysis/development_review.py \
  --root "$ROOT" --config "$ROOT/configs/protocol_v2.json" \
  --manifest "$ROOT/manifests/development_primary.jsonl" \
  --splits "$ROOT/manifests/splits.json" \
  --label-summary "$ROOT/reports/label_candidates.jsonl" \
  --cache-index "$ROOT/caches/embeddings/full_shard0_index.jsonl" \
                "$ROOT/caches/embeddings/full_shard1_index.jsonl" \
  --output "$ROOT/reports/development_review"
```

Optional `--dispatch`, `--freeze`, and `--run-root` specify alternate locations
for the same frozen identities; they do not relax identity validation.
The default dispatch is `reports/dispatch_after_pilot.json`, freeze is
`configs/execution_freeze.json`, and run root is `checkpoints/`.

Absent status files and non-complete statuses produce
`DEVELOPMENT_REVIEW_WAITING`, with all ten runs inventoried and no LR selections
or efficacy comparisons. A completed run with malformed or inconsistent
artifacts raises an error. Ten validated completed runs produce
`DEVELOPMENT_REVIEW_READY_FOR_HUMAN_OR_AGENT_EVIDENCE_REVIEW`.
Both statuses are successful CPU report executions. The report records opaque
checkpoint file SHA256 values; it does not inspect serialized checkpoint tensors.

Outputs are `report.json`, `paired_inputs.jsonl`, `engineering.csv`, and
`ENGINEERING.md` under the explicit output directory. The resolved output must
be `reports/development_review` or a descendant of that dedicated subtree,
confined through common helpers. Run validation directories, frozen inputs,
and `paper/` cannot be output destinations. Explicit dispatch/freeze inputs and
alternate run roots receive the same collision protection; output cannot
contain those input files or overlap run roots or the frozen code snapshot.
CSV and Markdown are written to fresh temporary files and atomically replace
their output entries, preserving protected targets even if existing output
names are hardlinks or in-root symlinks.

Local CPU validation command (bundled runtime, standard library only):

```powershell
& 'C:\Users\ArmStrong\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s analysis/tests -p test_development_review.py -v
```

Remote equivalent after synchronization:

```sh
"$PY" -m unittest discover -s analysis/tests -p test_development_review.py -v
```

The core report compares the five frozen trained arms. Separate
`review_scalar_baselines.py` now validates the original utility/local-cosine/
static-coverage artifacts and produces C-U/C-L/C-V comparisons under
`reports/development_review/scalar_baselines/`; job 1109421 completed successfully.
