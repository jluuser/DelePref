# Data Card

The checksum catalog is `benchmarks/manifest.json`, also packaged as
`delepref/assets/catalog.json`. `delepref assets list` prints it.

| Asset | Contents |
| --- | --- |
| `benchmark_manifests.json.gz` | 100 General100, 80 Enzyme80, 60 Binder60 WT sequences/hashes and source identities; 3,474 per-output measurement rows |
| `general100_profiles.json.gz` | Frozen reviewed homolog projections and source metadata |
| `paper_tables.json.gz` | Exact aggregate tables, archived intervals, budgets and case/mechanism plot data |
| `sec6_case.zip` | Real 2FJI alignment, masks/contact data, WT PDB excerpt, predicted Cα coordinates and existing molecular PNGs/cameras |

Budgets, predictors, seeds and candidates are not additional independent
proteins. Unknown/failed status is retained. Preparation-stage permission
flags and private paths are omitted from the normalized export; source
hashes remain available.

## Protocol

Scientific runners and audits remain in `recomb/`. Main General100 budget is
20%; fixed General40 uses 10/20/30/40%. Six methods are retained: DelePref,
DelePref-Base, Gap-frequency, Terminal, SCISOR-L K1 and Raygun. Native search
budgets differ where recorded; K1 is not presented as equal-K/compute.
Completion controls use 20 paired WTs. The morphology control is a post-outcome
supplement, not a separate preregistered confirmation.

Common structure-mapped contacts differ from fixed-WT identity contact fates.
The Sec6 identity example has 536 contacts: Base 137 deleted / 139
retained-unrecovered / 260 recovered; guided 150 / 58 / 328. Do not exchange
these denominators. Table intervals are archived paired intervals; CPU
reproduction does not recompute bootstrap or significance.

## Training availability

The formal training reference contains 269,915 WT sequences. The preference
corpus contains 24,386,017 pairs, 600 CSV shards, 36,013,051,774 bytes before
compression. **The shards are not hosted in this code release.** The catalog
reports `recipe_only`, without a nonexistent URL or claim of complete from-scratch
data availability. Existing construction/training scripts and configs remain;
see `docs/legacy_training.md` and `recomb/provenance/`.

A current UniRef download does not recreate an older snapshot byte-for-byte.
The inherited retrieval cache's exact upstream FASTA release identity is not
fully recovered. Frozen reviewed projections are included for replay. Fresh
retrieval must record database provenance and may produce different proposals.

## Sources and attribution

- [PDB](https://www.rcsb.org/): reference coordinates/construct IDs, CC0;
  retain structure-author citations.
- [CATH](https://www.cathdb.info/): domain identities/boundaries, CC BY 4.0.
- [UniProt/UniRef](https://www.uniprot.org/): sequences/homologs, CC BY 4.0;
  target/source IDs remain in profiles.
- [M-CSA](https://www.ebi.ac.uk/thornton-srv/m-csa/): catalytic sites, CC BY
  4.0; site identity is distinct from measured activity.

Upstream methods/predictors retain their terms in `THIRD_PARTY_NOTICES.md`.
Author-derived plots/measurements do not relicense raw databases. Historical
fixed-variant benchmarks are not new large-budget shortening assays. No
wet-lab validation is introduced by this release.
