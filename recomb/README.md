# DelePref-RAG / RECOMB extension

This directory is a historical research snapshot. For the current manuscript
readouts, including the five-draw completion analysis and deterministic Raygun
identity mapping, use `benchmarks/assets/paper_evidence_v2.zip` and the root
reproduction guide. The older results below remain available for audit.

This directory adds the completed homology-guided deletion-planning study to
the original DelePref repository. The original `src/delepref/`, `scripts/`,
training configurations and tests remain intact. The extension reuses the
step-4000 Policy/Scorer, builds homolog-derived seeds and retained anchors,
completes the exact deletion budget, and ranks complete candidates with the
same Scorer. Without a legal retrieved proposal it returns the original K8
pool exactly.

## Start here

- [Complete 63-page experimental report](artifacts/RECOMB_complete_experiments_20261019.pdf)
- [Figure/table editorial review](artifacts/RECOMB_figure_editorial_review_20261019.pdf)
- [Four final main figures and complete General100 supplement](visualization/figures/)
- [English captions](visualization/captions_en.md)
- [Frozen aggregate results](results/)
- [Scientific protocol and amendments](docs/protocols/)
- [Source/asset snapshot checksums](SOURCE_MANIFEST.json)

The report includes General100, Enzyme80, Binder60, budgets, repeated seeds,
the second predictor, candidate-pool analyses, E9 completion, E10 real request
costs, sixteen traced cases and the complete 100-row atlas. It incorporates
the final four figures and the corrected negative signs. P1 has not been run.

## Canonical implementation

| Component | Source |
| --- | --- |
| Unlimited-span exact-budget completion and deterministic fallback | `analysis/delepref_unlimited_planning.py` |
| Unchanged three-slot Scorer feature calculation and span-count / 3 scale | `analysis/delepref_unlimited_scorer.py` |
| Original checkpoint compatibility and strict identity checks | `analysis/delepref_v8_runtime.py` |
| Retrieval-guided candidate generation | `analysis/delepref_v8_rag_unlimited.py` |
| Confirmation General100 planning/evaluation | `analysis/recomb_confirm_v2_general_plans.py`, `analysis/recomb_confirm_v2_general_evaluate.py` |
| Application assembly, metrics and missingness | `analysis/recomb_confirm_v2_application_*.py` |
| Final NumPy-scalar JSON adapter | `analysis/recomb_confirm_v2_application_serialization.py` |
| Matched-seed completion follow-up | `analysis/recomb_confirm_v2_E9_completion.py` |
| Real first/subsequent-request timing | `analysis/recomb_confirm_v2_E10_*.py` |
| Structure task identity and journal | `code/prediction_journal.py` |

The frozen checkpoint SHA256 is
`1bc7694be77e24e4b671773be39331daed241dcfe101fd6d2d26f8e23fe7ff4d`.
The main pool uses K8 with nominal 4+4 slots, original deduplication/backfill
and no deletion-span cap. Scorer inputs preserve the three original feature
slots and span-count / 3 normalization. More than three spans is explicitly
an extrapolation beyond the original Scorer training range.

Source modules and freeze JSON/YAML files are exported byte-for-byte; the
repository attributes disable newline conversion for this snapshot. Earlier
v2-v8 diagnostic modules are retained for audit. They are historical code;
the canonical completed protocol is the confirmation-v2 plan and amendments
linked above. No Ridge/teacherU replacement selector or new training was used.

## Recorded results

- General100, main 20%: RAG minus DelePref is +0.0082 for common local geometry
  and +0.0128 for common native-contact recovery. Both joint 97.5% intervals
  include zero. Gap-frequency has similar means; Terminal has higher means
  on this panel. All six methods remain in the released tables.
- E9, twenty fixed WT/components: Policy minus Uniform completion is +0.0234
  for common local geometry and +0.0330 for common contacts, with joint 97.5%
  intervals [0.0011, 0.0526] and [0.0019, 0.0962]. All twenty final winners,
  ties and degradations are retained. Completion-only segment lengths were
  not matched; this is the original matched-seed follow-up.
- Binder60: interface-contact recovery decreases by 0.0817, 95% interval
  [-0.1412, -0.0289]. Enzyme80: complete catalytic amino-acid identity is
  61/80 for both DelePref and RAG. Identity and computational structural
  proxies are distinct from experimental activity or binding measurements.
- E10 contains all 240 real requests. First requests use 20% deletion;
  subsequent requests reuse WT assets and use 30%. The tables identify
  seconds versus milliseconds and the cache/input assumptions explicitly.

Full numeric precision, intervals, correspondence definitions and coverage
are in `results/`, the figure JSONs and the complete report. Source-context
flags are not estimates of deletion tolerance or biological correctness.

## Installation and software-only checks

Install the base package using the instructions in the repository root.
The following checks use Python, NumPy and CPU PyTorch for the original
seeded sampling convention, with no checkpoint,
retrieval database, model inference, structure prediction or cluster access:

```bash
cd recomb
PYTHONPATH=analysis:code python -m unittest discover -s analysis/tests -p 'test_delepref_unlimited_planning.py'
PYTHONPATH=analysis:code python -m unittest discover -s analysis/tests -p 'test_recomb_confirm_v2_E9_completion.py'
PYTHONPATH=analysis:code python -m unittest discover -s analysis/tests -p 'test_recomb_confirm_v2_E9_catalog.py'
PYTHONPATH=analysis:code python -m unittest discover -s analysis/tests -p 'test_recomb_confirm_v2_application_serialization.py'
PYTHONPATH=analysis:code python -m unittest discover -s analysis/tests -p 'test_prediction_journal.py'
```

The three small frozen E9 candidate/selection records needed by the catalog
check are included under `reports/recomb_confirm_v2_E9_completion/`. These
are software contract checks, not new biological experiments.
The existing head-interface test additionally requires the original checkpoint
and PyTorch; do not replace these assets with synthetic weights.

## Figure reproduction

```bash
python -m pip install -r recomb/visualization/requirements.txt
python recomb/visualization/render_final_figures.py
```

The plotter reads only the included frozen tables, case-coordinate arrays and
existing CPU structure-render PNGs. It outputs PDF, SVG and PNG at 174 mm width
and minimum 8 pt text; these are report assets to adapt to a submission's
final format. It performs no model or structure inference. The optional
`render_structures_pymol.py` can re-render those same C-alpha coordinates with
an existing PyMOL environment. It does not reconstruct unobserved atoms.

## Full scientific reconstruction

The archived scientific runners retain original machine paths, SHA bindings
and CPU/Slurm release checks. They are a source snapshot, not an automatic
download or dispatch command. Execute them from `recomb/` only after restoring
the exact separately obtained assets and their input manifests. The asset
registry in `provenance/` records checkpoint and source identities; its
historical discovery status is not the final experiment-completion status.

Model/encoder weights, preference-training tables, retrieval databases, raw
structure/MSA caches, cluster logs, user credentials and visualization runtimes
are excluded. Raw third-party data retain their original acquisition terms.
No manuscript source, submission, repository visibility change or additional
P1 calculation is part of this update.
