# DelePref: homology-guided protein shrinking

DelePref shortens a protein to an exact deletion budget using natural homolog
alignments and a learned deletion planner. Internal gaps in a single homolog
provide deletion seeds; observed flanking residues provide retained anchors.
The Policy completes the budget, and the complete-plan Scorer selects a
shortened sequence. Without a legal homolog proposal, the system returns the
same candidate pool and winner as the learned base engine.

The frozen ESMC-600M encoder runs once per WT. Policy and Scorer operate on
its residue representations without re-encoding variants. Default inference
ranks eight candidates: up to four homolog-guided plans, with Policy plans
filling the remaining slots. There is no inference deletion-span count cap.

## Real CPU demo

```bash
git clone https://github.com/jluuser/DelePref.git DelePref
cd DelePref
python -m venv .venv
# Activate .venv with the command appropriate for your shell.
python -m pip install .
delepref demo --case all --out demo-output
delepref demo --case sec6 --without-homology --out fallback-output
```

Python 3.10+ is required. The package includes **9.58 MB trained heads** and
two authentic cached ESMC embeddings. This CPU demo does **not** encode a new
sequence or substitute synthetic representations. Sec6 / 2fji101 shortens
209 to 167 residues; Af1862 / 2oebA00 shortens 152 to 122. Output folders
contain `results.json` (all candidates, masks, scores and provenance) and
`shortened.fasta`. Coordinates are **zero-based, closed/inclusive**.

## New sequence inference

Install compatible CUDA PyTorch first. Use an independent release environment,
not an existing experiment environment:

```bash
python -m pip install '.[encoder]'
# Explicit download: approximately 2.30 GB, checked against a pinned hash.
delepref assets download --asset encoder --out models/esmc_600m
delepref shrink --fasta input.fasta --budget 0.20 --seed 13 \
  --encoder-snapshot models/esmc_600m --profile retrieval/profiles.json \
  --out shrink-output
```

Omit `--profile` to use the learned base engine. Alternatively set
`DELEPREF_ESMC_SNAPSHOT` to an existing compatible snapshot. Encoder weights
must match the revision in the [Model Card](MODEL_CARD.md). Fresh encoding
requires CUDA; the cached demo runs on CPU. No model is downloaded implicitly.

## Replaying the long Cas9 cases

The manuscript's five full-length Cas9 inputs used a process-local encoder limit
of 2048. Their real frozen embeddings, input sequences, homolog profiles and
selected plans are available in the [public dataset](https://huggingface.co/datasets/Zhouky/DelePref-data).
Download and extract the pinned archive, then replay a case without re-encoding:

```bash
delepref assets download --asset cas9_evidence --out cas9-data
python scripts/verify_cas9_replay.py --evidence cas9-data/cas9-evidence-v1.zip
python -m zipfile -e cas9-data/cas9-evidence-v1.zip cas9-data/cases
delepref shrink --fasta cas9-data/cases/SpCas9/input.fasta --budget 0.20 \
  --embedding cas9-data/cases/SpCas9/embedding.safetensors \
  --profile cas9-data/cases/SpCas9/profile.json --allow-long-cached-embedding \
  --out cas9-replay
```

This explicit option accepts cached WT embeddings for up to 2048 residues
and marks lengths above 1022 as extrapolation. It does not enable fresh
long-sequence encoding. See [Model Card](MODEL_CARD.md).

## Homolog retrieval

Search is local and sequence based, with no structure prediction. Install
[MMseqs2](https://github.com/soedinglab/MMseqs2), obtain a versioned protein
FASTA under its source terms, and build a database:

```bash
mmseqs createdb homologs.fasta homolog_db
delepref retrieve --fasta input.fasta --database homolog_db \
  --threads 4 --out retrieval
```

The output retains real alignments, full-target identity checks and source IDs.
Unaligned termini remain unknown. A raw search is marked as sequence-quality
filtered, **not biologically source-reviewed**. See [evidence review](docs/retrieval.md)
before treating fragments or domain-only hits as evolutionary deletion examples.
The demos and released General100 profiles use frozen reviewed evidence.

## CPU table and figure reproduction

```bash
python -m pip install '.[evaluation]'
delepref assets download --asset all --source-dir . --out release-data
delepref reproduce --data-dir release-data --out reproduced --sec6
```

Without a checkout, omit `--source-dir` to download the small assets from the
pinned GitHub tag. Private repositories require an authorized local checkout
or an explicit `DELEPREF_GITHUB_TOKEN`. Every asset is checked for size and
SHA256. `all` excludes the large encoder and training corpus.

This recreates General100, Enzyme80, Binder60, completion and cost CSVs, the
General40/completion supplements, and the 174 mm Sec6 figure (`Figure2.pdf`,
`.svg`, `_600dpi.png`). It recalculates the five-draw paired bootstrap and
Raygun exact-identity summaries from frozen per-target measurements. The
earlier single-draw control and alignment-ambiguity bounds are retained as
separate outputs; they are not interchangeable with the manuscript values.
No model, folding or training is run. See the [reproduction guide](docs/reproduction.md)
for the scope, input provenance and output names.

## Training data

The [public dataset](https://huggingface.co/datasets/Zhouky/DelePref-data)
provides the 600 frozen preference shards and their sequence and protocol
metadata. They contain 24,386,017 pairs (36.01 GB before compression). Download
the `training/` folder, then verify and prepare it with:

```bash
hf download Zhouky/DelePref-data --repo-type dataset \
  --revision 634b96176b78988961535d30effbb23970748c7e \
  --include 'training/**' --local-dir dataset
python scripts/restore_training_data.py --dataset dataset/training \
  --out restored-training --encoder-snapshot models/esmc_600m
python scripts/train_final.py --config restored-training/training_config.yaml --preflight-only
```

The preparation step verifies the released shards and writes a local training
configuration; it does not download the encoder or run training. The original
UniRef release identity is not fully recovered, and the selected sequences
are included to support reuse of the frozen pairs. See the [Data Card](DATA_CARD.md)
and [training guide](docs/legacy_training.md) for scope and source terms.

## Contents and availability

| Material | Location / availability |
| --- | --- |
| Trained Policy/Scorer, config and SHA256 | `src/delepref/assets/`; installed with package |
| Real cached-encoder demos and expected candidates | `src/delepref/assets/examples/` |
| 240 WT manifests, frozen measurements and General100 profiles | `benchmarks/assets/` |
| Five-draw completion and Raygun exact-identity evidence | `benchmarks/assets/paper_evidence_v2.zip` |
| Sec6 coordinates and existing molecular renders | `benchmarks/assets/sec6_case.zip` |
| Primary benchmark structures and native references | [Hugging Face dataset](https://huggingface.co/datasets/Zhouky/DelePref-data) |
| Five full-length Cas9 embeddings, selections and complex structures | Hugging Face dataset, `cas9/` |
| Manuscript figures from the provided paper project | `benchmarks/figures/` |
| ESMC-600M backbone | Explicit pinned upstream download; not bundled |
| Training/teacher data construction | `scripts/`, `configs/`, [training guide](docs/legacy_training.md) |
| 600 preference shards (36.01 GB uncompressed) | Hugging Face dataset, `training/` |
| Archived scientific runners and experiment record | `experiments/`; historical runners may require their original inputs |

Structural geometry, contact recovery, site identity and interface proxies
are separate outcomes; none alone establishes biological function. Heads
were trained with at most three spans. Unlimited-span inference preserves
the Scorer's original count/3 scale and is flagged as extrapolation. See
[Model Card](MODEL_CARD.md), [Data Card](DATA_CARD.md) and [notices](THIRD_PARTY_NOTICES.md).

## Compatibility and tests

The release uses an independent clone. Existing experiment directories,
checkpoints and paths can stay in place. Explicit legacy checkpoint/config
paths remain accepted. See [compatibility](docs/compatibility.md).

```bash
python -m pip install '.[test]'
python -m pytest -q
```

Author code and exported heads use the existing MIT license. Upstream assets
retain their own terms. [CITATION.cff](CITATION.cff) provides the software
citation; the manuscript citation can be added when its public record exists.
