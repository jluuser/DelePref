# First-seed CPU diagnostics

Run from the canonical experiment root after the core and scalar development
reviews are ready:

```text
python analysis/cpu_prediction_diagnostics.py --config configs/development_diagnostics_first_seed.json
python analysis/cpu_decision_diagnostics.py --config configs/development_diagnostics_first_seed.json
```

These programs read NumPy predictions, frozen labels, sealed WT native coordinate
JSONs, and historical reviews. Checkpoints are hashed as bytes; they are never
deserialized. No variant structure, model forward, GPU, training, or folding is
used. Outputs are atomic and confined to the configured prediction/decision
subtrees of `reports/cpu_development_diagnostics`.
Historical relative `labels_path` metadata is anchored explicitly at the frozen
creation directory `ROOT/code`, regardless of the current working directory;
every resolved path must equal the configured labels directory and candidate ID.

Prediction metrics average known relations within candidate, candidates within
generator, generators within budget, budgets within WT, and WTs equally. Brier is
the sum over all five classes. Confusion rows are true classes. Calibration uses
ten fixed bins, including 1.0 in the final bin, and computes ECE from aggregated
bin masses. Unknown labels are excluded from correctness metrics and counted in
coverage. Contact and noncontact summaries remain undefined if any candidate has
zero known eligible relations, while retaining every candidate and WT.

Decision diagnostics evaluate the existing choices on the intersection of all
candidate retained relationships within each WT/budget/source or mixed pool.
Unknown relationships remain in that intersection with outcome bounds [0,1].
Empty intersections retain the full WT roster and suppress complete-case
bootstrapping. All seven comparisons use the verified frozen bootstrap helper,
10,000 replicates and seed 20261013. Static matching is descriptive feasibility
under exact morphology and fixed 0.05 coverage/contact bins; zero support never
relaxes the rule. Source summaries and CE leave-one-WT-out influence remain
descriptive historical diagnostics.

Reports preserve both favorable and unfavorable results. Definitions were
specified after first-seed validation choices and before this pair-level audit;
these diagnostics are not independent confirmation and produce no scientific
pass decision or folding authorization.

Synthetic contracts:

```text
python -m unittest discover -s analysis/tests -p test_cpu_diagnostics.py
```
