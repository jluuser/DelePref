# Historical training and auxiliary evaluation guide

This preserves the pre-portable-release instructions for the original
training/data-construction entrypoints. References below to excluded weights
and a three-span inference limit describe those historical entrypoints. The
current root README, Model Card and Data Card specify packaged heads and
unlimited-span homology-guided inference. The root README also explains how to
download and prepare the now-published preference pairs. Existing research paths
need no change.

DelePref learns budget-conditioned protein deletion plans from offline
counterfactual preferences. A frozen protein language model compares a
natural sequence with deletion variants during supervision construction. At
inference, the natural sequence is encoded once, a learned policy proposes
exact-budget multi-span masks, and a learned scorer reranks the complete plans
without encoding deletion variants.

Teacher scores are representation-level perturbation evidence. They are not
ground-truth functional labels, and DelePref does not guarantee preservation
of protein function or structure.

## RECOMB / DelePref-RAG extension (October 2026)

The [RECOMB extension](../recomb/README.md) adds homolog-guided deletion seeds,
retained anchors, unlimited-span exact-budget completion, frozen-Scorer
selection, and exact fallback to the original model. It includes the completed
General100 / Enzyme80 / Binder60 evaluations, E9 completion and E10 request
cost results, final figure sources and the
[complete experimental report](../recomb/artifacts/RECOMB_complete_experiments_20261019.pdf).
The original model and training implementation below are preserved.

## Repository Contents

```text
configs/                 DelePref model and training configurations
scripts/                 data construction, training, inference, and evaluation
src/delepref/            model, data, decoder, and streaming modules
tests/                   focused unit and integration tests
recomb/                  homology-guided extension, frozen summaries and figure assets
```

The historical source layout excluded checkpoints, model weights, generated
preference data, raw benchmark datasets and raw prediction outputs. For the
current availability of trained heads, frozen pairs and structure archives,
see the root README and Data Card.

## Installation

Python 3.10 or newer and a CUDA-enabled PyTorch installation are recommended.
Install the package and evaluation dependencies with:

```bash
python3 -m pip install -e '.[evaluation,test]'
```

Install the official `esm` package that provides ESMC separately, then place an
ESMC-600M safetensors snapshot at `models/esmc_600m/` or update the relevant
configuration fields. The expected snapshot contains `model.safetensors`.

## Datasets and Pretrained Assets

The original base release contained source code and small configuration files.
The current release also distributes selected trained heads, preference pairs
and structure artifacts as described in the root README and Data Card. Full
upstream databases and every historical structure cache are not included;
obtain additional material under its own terms.

### Required for Training

Core DelePref training requires the following:

| Asset | Purpose | How to obtain it |
| --- | --- | --- |
| UniRef50 FASTA | Natural protein sequences used to build the offline deletion-query bank | Download the release used for your experiment from the [UniProt UniRef50 FTP directory](https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref50/). The manifest builder accepts `.fasta` and `.fasta.gz`. |
| ESMC-600M runtime and weights | Frozen encoder and offline teacher | Install the official [EvolutionaryScale ESM package](https://github.com/evolutionaryscale/esm), then obtain the `biohub/ESMC-600M` snapshot from [Hugging Face](https://huggingface.co/biohub/ESMC-600M). The configured local directory must contain `model.safetensors`. |

For example, after obtaining the files, place them locally as follows:

```text
data/raw/uniref50/uniref50.fasta.gz
models/esmc_600m/model.safetensors
```

One possible command-line download sequence is:

```bash
mkdir -p data/raw/uniref50 models/esmc_600m
curl -L --fail --retry 3 \
  https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref50/uniref50.fasta.gz \
  -o data/raw/uniref50/uniref50.fasta.gz
hf download biohub/ESMC-600M model.safetensors \
  --local-dir models/esmc_600m
```

If the UniProt release changes its file name, use the directory linked above
and update `--input_fasta` accordingly. The `hf` command is provided by the
`huggingface_hub` package; authentication is not normally required for these
public assets.

The exact UniRef release must be recorded for reproducibility. The current
code deterministically filters standard amino-acid sequences with lengths
80--800, assigns sequence-hash train/validation/test partitions, and writes a
frozen manifest. A newer UniRef release can produce different rows and is not
bitwise equivalent to the dataset used in a previous experiment.

Build the natural-protein manifest with:

```bash
python3 scripts/build_uniref50_preference_manifest.py \
  --input_fasta data/raw/uniref50/uniref50.fasta.gz \
  --output_dir data/processed/preference_training/delepref/uniref50_manifest \
  --formal_train_size 300000 \
  --formal_val_size 20000
```

The remaining training data are generated locally. The main stages are:

```text
UniRef50 FASTA
  -> filtered sequence manifests
  -> raw deletion candidates
  -> frozen ESMC teacher impacts
  -> strict same-count preference pairs
  -> 20% supplement, split manifests, and condition catalog
  -> DelePref training
```

Use the `--help` output of the following scripts for the exact paths and
parallel-shard options for the local machine:

```text
scripts/build_stage1_deletion_pretrain_dataset.py
scripts/run_formal_teacher_scoring_shards.py
scripts/build_pair_same_count.py
scripts/build_b20_supplement.py
scripts/score_b20_esmc600m.py
scripts/build_final_split_manifests.py
scripts/build_condition_catalog.py
scripts/build_proposal_validation_manifest.py
```

If the frozen pair dataset and manifests have already been prepared, skip the
construction stages and set the corresponding paths in
`configs/deletion_preference_student.yaml`. The formal trainer expects the
base-pair index, the 20% pair index, condition catalog, train/dev manifest,
proposal-validation manifest, and their validation reports listed in that
configuration.

### Optional Evaluation Assets

The following assets are not required to train DelePref. They are needed only
for particular evaluation scripts or cross-model diagnostics:

| Asset | Used for | Acquisition |
| --- | --- | --- |
| ProteinGym indel bundle | Experimental deletion/indel proxy tables | Obtain the versioned bundle from the [ProteinGym project](https://github.com/OATML-Markslab/ProteinGym), then pass its local directory to the indexing scripts. |
| CATH and DisProt records | Structural or disorder-related proxy analyses | Obtain the relevant release from [CATH](https://www.cathdb.info/) and [DisProt](https://disprot.org/); the evaluation scripts consume local prepared tables. |
| AlphaFold Database structures | Structure-based deletion analyses | Obtain only the required structures or archives from [AlphaFold DB](https://alphafold.ebi.ac.uk/download). |
| Binder and affinity datasets | Binder/interface proxy analyses | Obtain the required AB-Bind, SAbDab, PPB-Affinity, SKEMPI, or related files under their own terms; use the `--help` options of the Binder preparation scripts to provide local paths. |
| ProtT5 weights | Cross-family representation audit | Download `Rostlab/prot_t5_xl_half_uniref50-enc` from [Hugging Face](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc). |
| ESM2-650M weights | Historical cross-teacher diagnostic only | Obtain the `facebook/esm2_t33_650M_UR50D` snapshot locally if that diagnostic is required. It is not used by the DelePref formal trainer. |
| Boltz-2 | Binder complex-prediction proxy | Install Boltz separately and provide its local runtime and input structures. These predictions are proxy evidence, not experimental binding measurements. |
| SCISOR source/checkpoint | Optional baseline comparison | SCISOR is not bundled. See the external-baseline instructions below. |

External datasets and weights are never silently downloaded by the formal
training code. Record their release, local path, and checksum in the relevant
experiment provenance files before reporting results.

## Data Construction

The training data are built without retraining or modifying the frozen teacher:

1. `scripts/build_stage1_deletion_pretrain_dataset.py` creates diverse
   cold-start deletion queries.
2. `scripts/run_formal_teacher_scoring_shards.py` scores natural/deletion-mutant
   counterfactuals with frozen teacher views.
3. `scripts/build_pair_same_count.py` constructs strict same-sequence,
   same-deleted-count preferences.
4. `scripts/build_b20_supplement.py` and `scripts/score_b20_esmc600m.py`
   construct and score the 20% budget supplement.
5. `scripts/build_final_split_manifests.py`,
   `scripts/build_condition_catalog.py`, and
   `scripts/build_proposal_validation_manifest.py` create leakage-controlled
   splits and streaming catalogs.

Every script provides `--help`. Input and output paths are explicit arguments;
the default layout is rooted under `data/processed/preference_training/`.

## Training

Edit `configs/deletion_preference_student.yaml` to point to the local ESMC
snapshot and generated manifests. Validate the inputs before allocating a long
run:

```bash
python3 scripts/train_final.py \
  --config configs/deletion_preference_student.yaml \
  --preflight-only
```

Run the formal two-GPU configuration with:

```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/train_final.py \
  --config configs/deletion_preference_student.yaml \
  --formal-training \
  --run-id delepref_training
```

For a bounded integration run, replace `--formal-training` with
`--smoke --max-steps 20`. Resume from a periodic checkpoint with
`--resume checkpoints/delepref/<run-id>/checkpoint_<step>.pt` and the same
`--run-id`.

## Inference

Generate exact-budget proposals and scorer-selected plans with:

```bash
python3 scripts/generate_plans.py --help
```

DelePref uses at most three non-overlapping deletion spans and supports the
formal 10% and 20% budgets. The exact dynamic-programming decoder enforces the
deleted residue count and span limit. Student checkpoints contain trainable
heads only; the frozen ESMC snapshot is loaded separately.

## Evaluation

The evaluation scripts cover preference fidelity, learned-policy versus
cold-start comparisons, CATH and DisProt proxies, measured enzyme deletions,
Binder analyses, inference and decoder ablations, confirmation evaluation,
split audits, and cross-teacher checks. Evaluation inputs and checkpoints are
passed through command-line arguments or local configuration files; generated
outputs remain outside the repository.

Some Binder analyses include SCISOR as an external reference. The SCISOR
source is not part of this repository. Set `DELEPREF_SCISOR_ROOT` to an
independently obtained SCISOR checkout, and set
`DELEPREF_SCISOR_CHECKPOINT` and `DELEPREF_SCISOR_P0` to its model assets
before running a phase that generates SCISOR plans. DelePref training and
inference do not require SCISOR.

External datasets and tools must be obtained under their own licenses. No
cluster-specific paths are required by the public code.

## Tests

Run the CPU test suite with:

```bash
python3 -m pytest -q
```

GPU-dependent model loading and full teacher rescoring require local ESMC
weights and are intentionally not part of the default unit-test invocation.

## License

The DelePref source in this repository is released under the license in
`LICENSE.md`. External datasets, model weights, and baseline implementations
remain subject to their own licenses.
