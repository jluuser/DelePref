# Retrieval and source review

MMseqs2 searches a local versioned sequence database without structure
prediction. The command records all three MMseqs commands/logs and the raw
17-column `hits.tsv`, including alignments and full target sequence. Query
and target coordinates must reconstruct the actual sequences before projection.

Fixed search settings: e-value 0.001, identity 0.20, search coverage 0.60 with
cov-mode 2, sensitivity 7.5, max 512 hits. Representatives additionally require
80% known WT-coordinate coverage. Exact duplicates and redundant same-gap
patterns are removed; at most 64 representatives remain. `.` denotes unknown;
`-` denotes an observed internal gap. Unobserved termini never supply a seed.

Sequence filters do not establish deletion safety. Fragments, partial domains,
weak flanks and compensation require source review. Raw search status is
`SEQUENCE_QUALITY_ONLY_NOT_BIOLOGICALLY_REVIEWED`. An explicit review JSON can
be supplied with `--reviewed-targets`:

```json
{
  "actual_target_identifier": {
    "status": "RETAIN",
    "representative_sequence_sha256": "full-target-sequence-sha256"
  }
}
```

Only retained targets with matching full-sequence hashes survive. Review is
a source audit, not a deletion-success label; do not use final structure
results to choose reviewed entries.

The generated `profiles.json` is keyed by input FASTA identifier with
`query.sequence_sha256` and real `representatives[].projected`. Pass it to
`delepref shrink --profile`. WT identity and projection alphabet/length are
checked. No legal event yields exact learned-base fallback.

For paper inference replay, use the frozen General100 profile asset and
exact WT IDs/sequences in the benchmark manifest. A new database is a new
retrieval experiment, not the frozen paper input.
