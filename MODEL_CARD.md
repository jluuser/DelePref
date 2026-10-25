# Model Card

## Identity

- DelePref 1.1.0: homology-guided exact-budget protein shrinking.
- `src/delepref/assets/heads.safetensors`: 9,584,652 bytes, 32 float32
  tensors, 2,395,381 Policy/Scorer parameters.
- SHA256: `a2c29a859dc8b7e82ab38df8cdae18de5324e301ea574ed2c196ec669c584d34`.
- Formal evaluation checkpoint: step 4000; source SHA256
  `1bc7694be77e24e4b671773be39331daed241dcfe101fd6d2d26f8e23fe7ff4d`.
- Every tensor value/name is preserved. No training state, private path or
  encoder parameter is included. Exported author-trained heads use MIT;
  upstream resources remain under their own terms.

## Encoder and inputs

Frozen ESMC-600M, `biohub/ESMC-600M`, revision
`a7e82012c83126b9eedb055fea9fa84b6c02f094`; SHA256
`e4232c30fd35fe2f57051ec88a703996ac94520580b4b836894207a3d45d9ff8`.
Its 2,300,205,696-byte backbone is an explicit upstream download, not bundled.

WT inputs use 20 standard amino acids without silent truncation. Encoder
maximum is 1022 residues; training sequences were filtered to 80–800 residues.
Acceptance of other lengths is not evidence of validation there. BOS/EOS are
excluded. Encoding uses bf16; the formal residue cache is float32. Its actual
dtype is preserved. The pinned model's license and notices apply separately.

## Planning and selection

1. Encode WT once, obtain budget-conditioned learned residue logits.
2. Construct K=8 distinct exact-budget Policy candidates with frozen seeds.
3. Extract internal gap events and observed flanking anchors from one homolog
   at a time; unknown termini do not become natural gaps.
4. Complete/refine each seed under budget and anchor constraints, without a
   deletion-span count cap.
5. Mix at most four homolog proposals with Policy candidates; rank all eight
   with the same trained complete-plan Scorer and fixed tie-break.
6. Without legal retrieved proposals, return the base pool/winner exactly.

Deletion count follows `round(length * budget)`. Intervals are zero-based
closed/inclusive; default seed is 13. Scorer output is a preference scalar,
not a calibrated probability of folding, binding, stability or function.
Homolog evidence enters proposal generation/constraints; no unvalidated
fusion score is added in this release.

## Training and extrapolation

Teacher weak supervision combines global/retained/boundary representation
views with weights 0.35/0.45/0.20. These are not functional assay labels.
Heads retain three latent Policy slots and were trained with at most three
spans at 10%/20% budgets. Unlimited decoding changes the feasible set without
retraining weights. Original span averaging and count/3 scale are retained;
outputs with >3 spans carry `scorer_span_extrapolation=true`. Other budgets
and fragmented plans require this limitation to be disclosed.

## Evidence and limits

General100/Enzyme80/Binder60 contain 240 fixed WT targets. Local Cα geometry,
contacts, enzyme-site identity and interface proxies are reported separately
with missingness and paired uncertainty. Natural homolog gaps may depend on
compensatory substitutions. Neither predicted structures nor homolog gaps
prove preserved function. All six methods and adverse outcomes are retained.

CPU demos use authentic frozen embeddings, explicitly labelled cached inference.
All 32 candidate rows across two demos are compared with frozen records,
along with exact fallback, tokenization and legacy loading. See
`docs/release_acceptance.md` for recorded environment and validation scope.
