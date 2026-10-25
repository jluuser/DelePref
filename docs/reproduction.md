# Reproduction scope and sources

Follow the root README to verify the four small assets, then run
`delepref reproduce --data-dir release-data --out reproduced --sec6`.

| Output | Frozen source (under original reports) |
| --- | --- |
| General100 main/all-predictor CSVs | `recomb_confirm_v2_General100_full_presentation/main_and_second_predictor_table.json` |
| Enzyme80/Binder60 CSVs | `recomb_confirm_v2_application_evaluation_serialization_v5/main/mean_tables.json` |
| Uniform completion CSV | `recomb_confirm_v2_E9_result/paired_effect_table.json` |
| Morphology completion CSV | `recomb_E9_shape_evaluation_20261021/paired_comparisons.json` |
| Runtime CSV | `recomb_confirm_v2_E10_result/descriptive_summary.json` |
| General40/completion/changed-design supplementary plots | Budget/case/mechanism data in `paper_tables.json.gz` |
| Sec6 `Figure2.pdf`, `.svg`, `_600dpi.png` | Frozen real alignment/masks/coordinates/contact fates and existing PyMOL PNGs |

Source artifact SHA256 values are in the data and output reproduction report.
The older multi-figure plotter remains in `recomb/visualization/`.

Sec6 is 174 × 120 mm. Matplotlib creates vector labels, tracks and bars;
embedded molecular PNGs remain raster content. WT ribbon and predicted Cα
traces are distinguished. Unobserved atoms are not fabricated. There are no
new model calls, folding or bootstrap in this reproduction command.

The CPU demo replays trained-head inference from real cached ESMC outputs.
Fresh `shrink` uses the pinned backbone and optional reviewed profiles;
every candidate, winner, seed and model hash is emitted.

Full native mapping, folding, paired statistics and baseline runners remain
in `recomb/analysis/`, `recomb/code/` and `recomb/docs/protocols/`. They preserve
original paths/SHA bindings and are an audit snapshot, not an automatic portable
cluster command. SCISOR-L, Raygun, ESMFold/Boltz and the retrieval database
require their separately obtained runtimes/assets and licenses. The release
does not host full training shards or every raw structure cache.
