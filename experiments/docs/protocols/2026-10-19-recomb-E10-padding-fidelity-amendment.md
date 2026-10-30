# E10 measured scientific-padding fidelity amendment

> Execute inline under the existing two-experiment authorization. Preserve completed E9, original E8, original scientific outputs, and the original staged timing source and its completed clock QA.

**Goal:** Time actual first and further WT requests whose DelePref-family outputs use the same measured encoding layout as the completed scientific evaluation.

**Evidence:** CPU1121813 reproduced all160 fixed20 method/budget conditions exactly. GPU1121801 completed the four encoding controls: original-batch reconstruction and single-WT encoding padded to the original token width reproduced all20 embeddings byte-for-byte and all four methods' complete pools/winners; natural-width singleton encoding reproduced only14/20 winners for original/RAG/gap and19/20 Terminal. The first singleton repeat was exact. Filling width is an input-layout condition under the fixed bf16 implementation; no fp32 intervention was run, so precision alone is not claimed as the cause.

## Unchanged scope

- Same20 cost inputs, six original native method contracts, cold beta .2 and cached beta .3, same allocated A100, real process/request timing and actual uncached query search.
- Existing target DB and weights are reused. The original batch width is determined solely from the already frozen General100 input order, microbatch32 and sequence lengths; it is not selected using structure scores.
- No change to encoding weights, dtype, biological sequence, Policy/Scorer, candidate budgets, RNG, selector or main quality outputs. No new structure prediction for E10.

## Implementation and verification

- [ ] Preserve `analysis/recomb_confirm_v2_E10_timing.py`, original clock QA1121823 and its freeze. They have not been submitted for full timing.
- [ ] Create `analysis/recomb_confirm_v2_E10_padding.py` with a process-local tokenizer adapter that pads tokens/attention masks to the original input-derived width, never truncating or adding biological residues.
- [ ] Create `analysis/recomb_confirm_v2_E10_timing_padding_v2.py`, `slurm/recomb_confirm_v2_E10_timing_padding_v2.sbatch`, and `analysis/tests/test_recomb_confirm_v2_E10_padding_v2.py`. New timing output/freeze namespaces use `E10_timing_padding_v2`.
- [ ] Keep the already tested `observe_process` function AST identical. Test that identity and, without GPU inference, compare the real ESM tokenizer's all20 padded outputs against the exact nested tokenizer implementation used in the completed fidelity experiment. Test attention/special-token preservation and rejection of truncation.
- [ ] Freeze the new padding adapter and entry source before its CPU QA. Require that QA and the completed fidelity diagnosis before the full timing freeze.
- [ ] In each first head-family request, temporarily use the adapter in that process's `FrozenWTEncoder`, call the unchanged scientific `api.encode_chunk`, and restore the tokenizer. Save the actual embedding; verify its raw tensor bytes against the measured scientific embedding identity. Further requests reuse that exact representation.
- [ ] Record padding width, actual WT representation identity, outputs and costs. This is asset-ready process-cold service under the fixed scientific layout; OS caches may be warm, and no offline target DB build is repeated.
- [ ] Submit the full timing job held with explicit one node/one task, inspect actual Slurm GRES and CPU/memory request before release. A required-node list is not an allowed-node list; use the permitted partition and existing QoS. Do not edit live/queued bound files or other projects.
- [ ] Execute all120 native method processes and240 actual requests, summarize full first/cached latency and peak memory, verify all produced identities and both-host SHA, and only then complete E10 and the active goal.

The earlier resource-dispatch correction is archived separately under `reports/recomb_targeted_supplements_20261019/dispatch_resource_repair_1121791_1121801/`. Existing job IDs and dependencies were retained. E9 task0's original extra GPU reservation is disclosed there; no second worker or duplicate structure was launched.
