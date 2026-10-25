# Compatibility with continuing experiments

Keep active experiment trees, YAMLs, formal checkpoints and frozen paths in
place. Use an independent clone/environment for portable release work. The
existing `recomb/` research snapshot retains its bytes and SHA bindings.

Explicit legacy paths are accepted without rewriting their files:

```bash
delepref demo --case sec6 --out compatibility-check \
  --weights /your/existing/formal_evaluation.pt \
  --model-config /your/existing/student_config.yaml
```

Only use trusted legacy `.pt` files; loading uses `weights_only=True` and
requires the original 32 head tensors/architecture. Safetensors is the
release default. Optional environment variables are `DELEPREF_HEADS` and
`DELEPREF_ESMC_SNAPSHOT`.

The old encoder loader has only a release-local tokenizer compatibility
fallback for ESM/Transformers property APIs. It does not patch site-packages.
Token IDs and actual GPU encoder outputs are validated separately from the
cached-embedding head checks.

Continue manuscript experiments in their frozen environment. Make prospective
changes in another branch with new output directories/versioned configs.
Do not overwrite old reported results to change paths. Export an allowlisted
result bundle, preserve original hashes and update public checksums explicitly.

The asset catalog pins `recomb2027-review-20261025`; do not silently move its
scientific contents. Future data archives may use Hugging Face Datasets or a
DOI archive with immutable revisions and hashes linked from GitHub. Shared
Python environments, original weights and manuscript files need no change.
