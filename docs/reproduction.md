# Reproducing manuscript readouts

The repository provides trained heads, real cached-embedding demos, frozen
evaluation data, and scripts for table and figure regeneration. Start with
the installation and CPU demo in the [root README](../README.md).

## Tables and Sec6 figure

From a checkout at the version recorded in `benchmarks/manifest.json`:

```bash
python -m pip install '.[evaluation]'
delepref assets download --asset all --source-dir . --out release-data
delepref reproduce --data-dir release-data --out reproduced --sec6
```

For a standalone installation, omit `--source-dir .` to download the pinned
assets. Every asset is checked against its declared size and SHA256. Use a
fresh output directory for Sec6, since existing case files are not overwritten.

| Output | Source and meaning |
| --- | --- |
| `table_general100_main.csv`, `table_general100_all_predictors.csv` | Frozen General100 measurements in `paper_tables.json.gz` |
| `table_enzyme80.csv`, `table_binder60.csv` | Frozen application measurements, plus `strict_identity__*` Raygun point estimates |
| `table_raygun_strict_identity.csv` | Four deterministic exact-identity macro summaries used in the manuscript |
| `table_raygun_strict_per_target.csv` | Corresponding 140 per-target readouts; the alternative mapping bounds remain in the application tables |
| `table_completion_uniform.csv` | Original fixed-seed uniform-completion comparison |
| `table_completion_morphology.csv` | Post-outcome five-draw morphology comparison, averaging five draws within each WT |
| `table_completion_morphology_per_WT_draw.csv` | All 500 endpoint/WT/draw differences used to audit the five-draw results |
| `table_completion_morphology_single_draw.csv` | Earlier single-draw analysis, retained for historical comparison |
| `table_runtime_descriptive.csv` | Frozen first- and reused-request timing summary |
| `sec6_case/Figure2.pdf`, `.svg`, `_600dpi.png` | Vector layout and labels composed with the archived molecular renders |

The five-draw paired bootstrap is recomputed from the frozen per-WT
differences with 10,000 draws and seed 20261025, and checked against the
recorded results. The Raygun means are recomputed from 140 per-target
exact-identity measurements and checked against their frozen summary. Other
confidence intervals are loaded from the original archived results. The
command performs **no new model inference, structure prediction or training**;
it does not recompute contact metrics from raw structures.

## Evidence and historical source

`benchmarks/assets/paper_evidence_v2.zip` includes the five-draw protocol,
all per-WT and per-draw differences, selected candidate records, the 18 new
follow-up ESMFold artifacts, the exact-identity Raygun mappings and metrics,
and source checksums. `scripts/build_paper_evidence.py` documents how the
archive is assembled from the original research exports. JSON paths under the
original research root are normalized to `research-source/`; they are source
identifiers, not guaranteed downloadable paths.

Historical experiment scripts are retained under `recomb/analysis/`, including
`recomb_evidence_followup_20261025.py` and the Raygun canonical audit scripts.
These runners require their original inputs and scientific software and are not
the portable CPU table-regeneration command. The five-draw archive is a
post-outcome sensitivity analysis, not a preregistered confirmation. The older
single-draw result and historical alignment-ambiguity bounds are not the same
analyses as the current manuscript results.

## Primary structures and native references

The frozen [Hugging Face structure archive](https://huggingface.co/datasets/Zhouky/DelePref-data/blob/7ab323fefffc0087fcce2f819450be8d5a37567b/structures/primary-structures.tar.gz)
contains 2,469 physical prediction records and 240 native references for the
3,474 logical measurement rows. It retains the two failed prediction statuses.
Download the archive and verify it without extraction:

```bash
python scripts/verify_structure_archive.py primary-structures.tar.gz
```

The archive manifest maps physical IDs to their prediction records and
coordinate files; join those IDs with `benchmark_manifests.json.gz` to recover
the WT, method, budget and prediction seed. Source-reference JSON paths are
normalized; the manifest retains original and normalized hashes. The archive
does not include model weights, all possible candidate predictions or every
historical experiment cache. Full metric recomputation still requires the
documented evaluation software and matching reference/chain conventions.
The separate [Cas9 structure archive](https://huggingface.co/datasets/Zhouky/DelePref-data/blob/efe17817db199a30a99dc850f4391aab5bf3dcff/cas9/cas9-structures-v1.tar.gz)
contains 88 predicted complexes and 18 reference files. The same verifier
supports both archives.

## Model and training scope

The CPU demos use authentic cached ESMC outputs. Fresh `shrink` requires the
pinned upstream encoder and CUDA and keeps the 1022-residue limit. Historical
Cas9 full-length cached embeddings can be replayed explicitly with
`--allow-long-cached-embedding`; the five Base/guided selected masks were
checked against their frozen records. This is length extrapolation and does not
run a new long-sequence encoder. See [Model Card](../MODEL_CARD.md) and [Data Card](../DATA_CARD.md)
for model, training-data and structure-availability limits. The 600 training
shards are distributed separately via the public dataset, with download and
preparation instructions in the root README. The raw UniRef database and every
historical structure cache are not included in this source checkout. A score regenerated from a frozen table is
not an independent structure-based validation.

The manuscript figures supplied with the paper project are under
`benchmarks/figures/`. Their scientific source data and third-party attribution
are described in [Data Card](../DATA_CARD.md) and
[Third-party notices](../THIRD_PARTY_NOTICES.md).
