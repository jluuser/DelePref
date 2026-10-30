# Selection bottleneck analysis

`selection_bottleneck.py` reads the completed, sealed, three-seed ensemble audit.
It does not call the ensemble builder, recover checkpoints, train models, run
embeddings, read variant structures, or replace frozen Top1 choices. Every report
is exploratory analysis of reused primary validation.

Run on the authorized experiment host with NumPy and the frozen CPU helpers:

```sh
python analysis/selection_bottleneck.py \
  --root /public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013 \
  --freeze /public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013/configs/prospective_replication_freeze.json \
  --output /public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013/reports/replication_review/selection_bottleneck
python -m unittest discover -s analysis/tests -p test_selection_bottleneck.py -v
```

The implementation follows three checked stages:

1. Validate the prospective freeze and original input identities using the
   read-only context loader. Validate the actual replication report, its sealed
   ledger, all five exact 420-candidate prediction rosters, ensemble paths/output
   hashes, ordered source hashes, original labels, and native WT relationship
   mappings. Recompute CE/Ghat/That and observed-pair errors from each sealed NPZ.
2. Compute candidate calibration and within-WT/budget/source ordering, original
   regret and oracle ties, TopK recall, winner margins, and coverage/G-only post
   hoc diagnostics. Compute C−B/D−C candidate changes and CE changes on the union
   of the two models' predicted top-two candidates versus all other candidates.
3. Summarize with equal WT → budget → source → candidate weighting and run the
   frozen paired component bootstrap for C−B, D−C and D−B, retaining the entire
   WT roster (10,000 replicates, seed 20261013). Write a dedicated new output tree.

Outputs:

- `report.json`: identities, definitions, all-arm calibration/ranking summaries,
  D comparisons, CE translation, agreement, and paired intervals.
- `summary.csv`: compact all-arm/global/budget/source metrics.
- `per_wt_pools.jsonl`: unchanged frozen choices, observed truth or bounds,
  original regret bounds, post hoc selectors, oracle ties, rankings and margins.
- `candidate_deltas.jsonl`: per-candidate C−B/D−C CE, G/T prediction and absolute
  error differences; negative CE or absolute-error changes indicate improvement.
- `pool_comparisons.jsonl`: CE improvement with unchanged Top1, plus both
  candidate-mean and known-pair-weighted decision-relevant CE changes.

The `all/all` summaries use source-internal pools and count each candidate once.
Mixed pools are separate, with 14 candidates instead of the seven per source.
Pooled Pearson/Spearman are candidate-level descriptive diagnostics. Within-pool
Spearman uses average tied ranks; constants are undefined and counted. A tied
truth oracle is hit by any truth-maximizing candidate; its predicted rank is the
best rank among tied oracle candidates under the original That/candidate-ID tie
rule. Generator oracle counts use that explicit candidate-ID tie rule.

`pair_preservation_brier` is the squared error between expected preservation and
observed preservation value; `multiclass_brier` is the categorical Brier score.
Observed-pair metrics are labeled as such when unknown retained pairs exist.
Unknown G/T bounds disable point summaries for the whole affected stratum;
there is no complete-case paired inference or imputation. Zero oracle improvement
over static coverage gives a null headroom ratio. A margin/error-SD ratio is a
descriptive scale comparison, not a calibrated uncertainty probability.
`realized_T_gap_between_predicted_top2` compares the realized T of the frozen
predicted winner and runner-up and can be negative. `oracle_sorted_true_top2_gap`
compares the two highest true T values in the pool and is zero for an oracle tie.

Static coverage uses sealed C, the original native-WT relationship coverage.
Coverage-only and G-only selectors are diagnostics and are never production
outputs. Original U is optional and is not loaded; historical first-seed C−U
results are not mislabeled as ensemble evidence. D uses edited embeddings plus
WT structure/deletion features, so it is not a pure sequence model.

Tests cover global/within-pool correlation reversal, CE improvement at unchanged
expected preservation/Top1, conflicting G/T oracles, ties and zero headroom,
source dominance, WT rather than variant weighting, D's lower-CE/worse-choice
case, unknown-truth handling, protected outputs, sealed hashes/rosters/labels,
missing probabilities and incorrect pair mappings, and a complete CPU fixture.

To reproduce the existing inline pool MSE decomposition and 15 checkpoint curves
as clearly labeled posthoc CPU reports in an isolated child directory:

```sh
python analysis/selection_bottleneck_posthoc.py \
  --root /public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013 \
  --freeze /public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013/configs/prospective_replication_freeze.json \
  --output /public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013/reports/replication_review/selection_bottleneck/posthoc_reproduction \
  --decomposition-sha256 0fa1c412c2a276392bcc17ec2f36a85d9fe244b2015db8b2a1e7ee6df845fd19 \
  --curve-sha256 ec328f822e8ba29f75eed06c1af946149951d85aa803f2576bbe2c3763c72e9c
python -m unittest discover -s analysis/tests -p test_selection_bottleneck_posthoc.py -v
```

The prior audit seals the 281 historical checkpoint `report.json` files, training
traces and selected-checkpoint outputs. Companion candidate/protein/decision
files at unselected epochs are checked semantically against those pinned reports
and frozen labels; their byte hashes are first recorded by this reproduction.
Those new hashes verify stability during this read, not retrospective byte
identity with the original training run. Whitespace-only changes in an unsealed
companion cannot be detected retrospectively and do not change the curve result.
