# Release acceptance

Validation date: 2026-10-25. New package code was installed from a wheel into
an isolated target directory, then executed outside all source directories.
Dependencies were reused read-only from an existing environment; this is
not a claim that dependency resolution was tested on every operating system.
No shared environment, original checkpoint, research source or paper was modified.

## Reference environment

Python 3.10; PyTorch 2.3.1; NumPy 1.26.4; safetensors 0.4.5; PyYAML 6.0.2;
Transformers 4.45.0; tokenizers 0.20.0; ESM 3.2.1.post1; Matplotlib 3.9.2;
Pillow 10.4.0; pytest 9.1.1. Fresh encoder acceptance used an allocated
NVIDIA A100 80GB PCIe through Slurm. Core cached inference needs no ESM install.

## Recorded checks

- All 32 exported head tensors preserve names/shapes/values and strict loading.
- Two real cached-encoder cases compare all 32 candidate rows, scores/ranks,
  selected masks/sequences and exact no-evidence fallback to frozen outputs.
- The initial full CPU suite passed 110 tests, with one optional test skipped.
- The final expanded suite passed **116 tests**, with one optional test skipped,
  including explicit read-only loading of the original checkpoint/YAML.
- Genuine fresh encoding: 2 WTs in the original runtime and the same 2 in the
  portable runtime, identical A100 profile. Both float32 output tensors were
  **exactly equal**, maximum absolute difference 0.0. All four Base/guided
  selected masks also matched the frozen reference.
- An initial portable load failed before its sequence forward because pip ESM's
  read-only token properties conflict with Transformers 4.45. A release-local
  constructor compatibility fix preserved token IDs and passed the genuine
  encoder comparison. No shared dependency was patched.
- Offline asset checksums and full CPU table/Sec6 figure reproduction passed.
  The rendered 174 × 120 mm case figure was visually inspected.
- A genuine MMseqs2 local search of the real Sec6 WT against its real retrieved
  homolog passed projection/target identity checks. A strict MMseqs argument
  parser required leading zeros (`0.20`, not `.20`); numeric search thresholds
  were unchanged. The raw-search result remains explicitly source-unreviewed.
- Final wheel installation, demo/fallback, cached `shrink`, asset verification
  and full figure/table CLI commands passed outside all source checkouts.
- All 406 protected original source/config/weight files and the original
  server checkout HEAD remained unchanged. Introduced files/decompressed assets
  and both pre-release Git commits passed a credential-pattern scan; this is
  not a claim of exhaustive security review or removal of archived research paths.

Fresh encoding acceptance introduced four WT encodings in total, no variant
encodings, no training and no complete structure prediction. These checks
validate packaging equivalence, not additional biological efficacy.
