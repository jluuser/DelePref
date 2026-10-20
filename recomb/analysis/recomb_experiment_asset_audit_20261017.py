"""Read-only inventory for experiment planning; no model scoring or designs."""
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

ROOT = Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
SOURCE = Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
AA = set('ACDEFGHIKLMNPQRSTVWY')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for b in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def subsequence_maps(wt, mutant):
    left = []; start = 0
    for aa in mutant:
        pos = wt.find(aa, start)
        if pos < 0:
            return None
        left.append(pos); start = pos + 1
    right = []; end = len(wt)
    for aa in mutant[::-1]:
        pos = wt.rfind(aa, 0, end)
        if pos < 0:
            return None
        right.append(pos); end = pos
    right.reverse()
    return left, right


def main():
    out = ROOT / 'reports/recomb_experiment_design_20261017'
    out.mkdir(parents=True, exist_ok=True)
    legacy = SOURCE / 'data/processed/enzyme_v4_public_measured/enzyme_standardized_measured_variants.csv'
    with legacy.open() as f:
        rows = list(csv.DictReader(f))
    legacy_report = dict(path=str(legacy), sha256=sha(legacy), rows=len(rows),
        proteins=len({r['protein_id'] for r in rows}), datasets=len({r['dataset_id'] for r in rows}),
        variant_types=dict(collections.Counter(r['variant_type'] for r in rows)),
        interval_counts=dict(collections.Counter(r['num_intervals'] for r in rows)),
        caution='Raw mixed-indel rows are not all valid deletion-only observations or independent proteins.')
    reference_path = ROOT / 'provenance/recomb_literature_20261017/ProteinGym_PG_v1_3_DMS_indels_reference.csv'
    with reference_path.open() as f:
        reference = {r['DMS_id']: r for r in csv.DictReader(f)}
    archive = ROOT / 'external/rag_v6/DMS_ProteinGym_indels.zip'
    assay_rows = []; physical = set(); wt_sequences = set(); protein_ids = set()
    with zipfile.ZipFile(archive) as z:
        members = sorted(n for n in z.namelist() if n.endswith('.csv'))
        for member in members:
            assay = Path(member).stem
            if assay not in reference:
                raise ValueError('missing exact release metadata: ' + assay)
            meta = reference[assay]; wt = meta['target_seq']
            counts = collections.Counter(); unique = set(); lengths = collections.Counter()
            with z.open(member) as raw:
                for row in csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig')):
                    counts['all_rows'] += 1
                    mut = row['mutated_sequence']
                    try:
                        value = float(row['DMS_score'])
                    except (ValueError, TypeError):
                        counts['nonfinite_measurement'] += 1; continue
                    if not math.isfinite(value):
                        counts['nonfinite_measurement'] += 1; continue
                    if not wt or not mut or not set(wt).issubset(AA) or not set(mut).issubset(AA):
                        counts['noncanonical_sequence'] += 1; continue
                    deleted = len(wt) - len(mut)
                    if deleted <= 0:
                        counts['not_shorter_than_WT'] += 1; continue
                    maps = subsequence_maps(wt, mut)
                    if maps is None:
                        counts['shorter_but_not_strict_subsequence'] += 1; continue
                    counts['strict_deletion_compatible_rows'] += 1
                    counts['single_residue' if deleted == 1 else 'multiple_residues'] += 1
                    counts['unique_coordinate_map' if maps[0] == maps[1] else 'ambiguous_coordinate_map'] += 1
                    lengths[deleted] += 1
                    if deleted / len(wt) >= .1: counts['deletion_fraction_ge_10pct'] += 1
                    if deleted / len(wt) >= .2: counts['deletion_fraction_ge_20pct'] += 1
                    if maps[0] == maps[1]:
                        retained = set(maps[0]); deleted_ids = [i for i in range(len(wt)) if i not in retained]
                        spans = sum(i == 0 or p != deleted_ids[i - 1] + 1 for i, p in enumerate(deleted_ids))
                        counts['unique_map_multispan' if spans > 1 else 'unique_map_single_span'] += 1
                    unique.add(mut); physical.add((wt, mut)); wt_sequences.add(wt); protein_ids.add(meta['UniProt_ID'])
            assay_rows.append(dict(assay_id=assay, UniProt_ID=meta['UniProt_ID'], WT_length=len(wt),
                WT_sha256=hashlib.sha256(wt.encode()).hexdigest(), selection_type=meta['selection_type'],
                source_organism=meta['source_organism'], counts=dict(counts), unique_deletion_sequences=len(unique),
                deletion_length_counts=dict(lengths)))
    counts = collections.Counter()
    for r in assay_rows: counts.update(r['counts'])
    pg = dict(archive=str(archive), archive_sha256=sha(archive), reference_path=str(reference_path),
        reference_sha256=sha(reference_path), release_tag='PG_v1.3', archive_CSV_files=len(assay_rows),
        assays_with_deletion_compatible_rows=sum(r['counts'].get('strict_deletion_compatible_rows',0)>0 for r in assay_rows),
        unique_WT_sequences=len(wt_sequences), unique_UniProt_IDs=len(protein_ids),
        unique_WT_variant_sequence_pairs=len(physical), counts=dict(counts), assays=assay_rows,
        status='CPU_SUBSEQUENCE_INVENTORY_NOT_FINAL_BENCHMARK',
        remaining_audits=['Original-training homology/exposure on the expanded WT roster',
                         'Assay provenance, score direction and duplicate-release checks',
                         'Fixed treatment of ambiguous multi-deletion coordinate maps',
                         'Competitive-model per-variant score coverage and input-profile matching'],
        no_model_scores_or_variant_recommendations=True)
    assets = []
    for relative in ['cath_s40_domain_benchmark.csv','disprot_idr_benchmark.csv','egfr_binder/egfr_round2_binders.csv',
                     'cas9_landscape_v2/cas9_landscape_normalized.csv']:
        path = SOURCE / 'data/processed' / relative
        with path.open() as f:
            reader=csv.DictReader(f); fields=reader.fieldnames; n=0; quality=collections.Counter()
            for row in reader:
                n+=1
                if 'quality_flag' in row:quality[row['quality_flag']]+=1
        assets.append(dict(path=str(path),sha256=sha(path),rows=n,columns=fields,quality_counts=dict(quality)))
    packages=[]
    for path in sorted((SOURCE/'results/experiments').glob('student_v3_step004000_*/run_summary.json')):
        data=json.loads(path.read_text())
        packages.append(dict(directory=str(path.parent),summary_sha256=sha(path),status=data.get('status'),
            counts={k:v for k,v in data.items() if k in ['case_count','protein_count','row_counts','counts','cohort_counts']},
            table_files=sorted(p.name for p in path.parent.glob('*.csv'))))
    report=dict(date='2026-10-17',legacy_measured_table=legacy_report,ProteinGym=pg,existing_benchmark_files=assets,
        existing_v3_packages=packages,rebuttal_retention_record=str(SOURCE/'rebuttal/README.md'),
        rebuttal_record_is_author_response=False,review_basis='User-supplied AAAI reviews plus retained rebuttal scientific packages',
        model_forward_calls=0,new_structure_predictions=0,paper_changes=False,script_sha256=sha(__file__))
    (out/'asset_and_dataset_audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'legacy_rows':len(rows),'ProteinGym_assays':len(assay_rows),'ProteinGym_counts':dict(counts),
                      'deletion_assays':pg['assays_with_deletion_compatible_rows'],'unique_WT':len(wt_sequences),
                      'unique_WT_variants':len(physical),'saved_packages':len(packages)}))


if __name__=='__main__':main()
