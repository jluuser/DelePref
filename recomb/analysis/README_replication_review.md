# CPU-only three-seed replication review

Run after or during the frozen prospective array, using the existing experiment
Python with NumPy. This command does not submit jobs or run inference:

```sh
ROOT=/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013
PYTHON=/public/home/zhangyangroup/chengshiz/keyuan.zhou/evocompress/.venv-v7/bin/python
"$PYTHON" "$ROOT/analysis/replication_review.py" \
  --root "$ROOT" \
  --freeze "$ROOT/configs/prospective_replication_freeze.json" \
  --output "$ROOT/reports/replication_review"
```

The verifier audits the five original selected seed-20261013 runs and ten frozen
seed-20261014/20261015 tasks at LR 0.001. Missing/running tasks produce
`REPLICATION_REVIEW_WAITING` without efficacy comparisons or ensemble results.
Malformed completed artifacts raise an error. A repeat audit may add completed
tasks but rejects changes to already audited run artifacts, prediction files or
opaque checkpoint bytes.

Every original seed-20261013 prediction NPZ is bound, on the first replication
audit, to the prior CPU prediction diagnostic ledger. Its SHA is sealed by that
diagnostic report, which is itself bound by the prospective freeze. The ledger
must contain exactly the five-arm by 420-candidate roster and original selected
validation paths. Recomputed CE and Ghat supplement this byte-level check; they
cannot replace it because different class probabilities can share those scores.

Once all fifteen runs validate, `THREE_SEED_REPLICATION_REVIEW_READY` records
every seed individually and a separate five-arm ensemble. Each ensemble averages
the three aligned five-class probability vectors per retained pair, before CE,
Ghat, That or candidate selection. The procedure and weights are identical for
S/A/B/C/D. It never selects seeds, averages winning candidates, or substitutes
mean per-seed CE for mixture CE.

Outputs include `report.json`, `ENGINEERING.md`, an artifact hash ledger,
per-seed paired inputs, and ensemble candidate/protein/decision reports plus
per-candidate probability NPZs. C-B/C-S/C-A/C-D intervals use the frozen paired
bootstrap helper with 10,000 replicates and seed 20261013, retaining the full ten
WT/component roster. Unknown-label bounds retain the original fixed denominators.
Seeds do not become extra WT observations. Validation model selection and effect
estimation share data; the results are historical training-seed stability evidence,
not independent confirmation. No scientific pass decision, confirmation folding,
or paper change is authorized.

Checkpoints are read only as opaque bytes. Original selected SHA values are
checked against the sealed original core review. Prospective checkpoint SHA values
are computed and rechecked during the first audit; the frozen trainer did not
independently seal or expose their internal parameter identity. Ensemble checkpoint
identity is an explicit ordered tuple of the three source checkpoint hashes, not
a purported checkpoint file. GPU/wall cost remains undefined until Slurm accounting
is supplied separately; trace steps and evaluation counts are recorded.

Local tests require only the standard library and NumPy:

```sh
python -m unittest discover -s analysis/tests -p test_replication_review.py -v
```
