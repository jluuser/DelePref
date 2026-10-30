"""Metadata-only scope curation. No model imports, generation, or score reads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

BASE = Path('reports/recomb_benign_scope_20261017')
POLICY = Path('configs/recomb_benign_scope_20261017.json')
HAZARD = re.compile(r'toxin|\bricin\b|colicin|larvicide|virulence|antibiotic resistance|'
                    r'antimicrobial resistance|beta[- ]lactamase|thiostrepton.resistance|'
                    r'trimethoprim resistance|aminoglycoside.*acetyltransferase|\bermc', re.I)
VIRAL = re.compile(r'virus|viridae|phage', re.I)
PATHOGEN = re.compile(r'Mycoplasmoides pneumoniae|Candida albicans|Klebsiella pneumoniae|'
                     r'Vibrio cholerae|Nocardia farcinica|Leishmania donovani|'
                     r'Providencia |Proteus vulgaris|Citrobacter freundii|'
                     r'Pseudomonas paraeruginosa|Streptococcus mutans|'
                     r'Plasmodium falciparum|Bacillus cereus', re.I)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def describe(entry):
    desc = entry.get('proteinDescription', {})
    return dict(accession=entry.get('primaryAccession'), organism=entry.get('organism', {}).get('scientificName'),
                taxonomy=entry.get('organism', {}).get('lineage', []), description=desc,
                keywords=[k.get('name', k.get('value', '')) for k in entry.get('keywords', [])])


def classify(metadata, reviewed):
    if not metadata:
        return 'HOLD', 'missing_source_or_function_metadata'
    description = json.dumps([dict(description=m.get('description'), keywords=m.get('keywords', [])) for m in metadata])
    organisms = json.dumps([dict(organism=m.get('organism'), taxonomy=m.get('taxonomy', [])) for m in metadata])
    if HAZARD.search(description):
        return 'EXCLUDE', 'toxin_virulence_or_resistance_annotation'
    if VIRAL.search(organisms):
        return 'EXCLUDE', 'viral_source_outside_requested_scope'
    if PATHOGEN.search(organisms):
        return 'EXCLUDE', 'pathogen_source_outside_requested_scope'
    if reviewed and all(m.get('organism') and m.get('description') for m in metadata):
        return 'RETAIN', 'individually_reviewed_ordinary_protein_scope'
    return 'HOLD', 'not_yet_positively_reviewed_or_source_ambiguous'


def main(root):
    policy = read(root / POLICY)
    out = root / BASE
    out.mkdir(parents=True, exist_ok=False)
    panel_path = root / 'reports/delepref_v8_native_panel/main_panel.json'
    mapping_path = root / 'reports/recomb_functional_annotations_20261017/mapped_features.json'
    panel = read(panel_path)
    mapped = {r['protein_id']: r for r in read(mapping_path)}
    reviewed = set(policy['general_reviewed_accessions'])
    rows, retained = [], []
    for p in panel:
        ids = mapped[p['protein_id']].get('candidate_accessions', [])
        metadata, sources = [], []
        for accession in ids:
            path = root / 'provenance/recomb_functional_annotations_20261017/uniprot' / (accession + '.json')
            if path.exists():
                metadata.append(describe(read(path)))
                sources.append(dict(path=str(path.relative_to(root)), sha256=sha(path)))
        positive = bool(ids) and set(ids).issubset(reviewed) and len(metadata) == len(ids)
        status, reason = classify(metadata, positive)
        row = dict(protein_id=p['protein_id'], WT_sha256=p['sequence_sha256'], source=p['stratum'],
                   length=p['sequence_length'], status=status, reason=reason, annotations=metadata,
                   annotation_files=sources)
        rows.append(row)
        if status == 'RETAIN':
            retained.append(p)
    write(out / 'general_audit.json', rows)
    write(out / 'general_panel.json', retained)
    allowed = {r['protein_id'] for r in retained}
    old_subsets = read(root / 'reports/delepref_v8_evaluation_preparation/subsets.json')
    subsets = {k: [pid for pid in value if pid in allowed] for k, value in old_subsets.items() if isinstance(value, list)}
    write(out / 'retained_subsets.json', subsets)
    for name in ['raygun', 'scisor']:
        old = read(root / 'reports/recomb_general100_execution_20261017' / (name + '_tasks.json'))
        write(out / (name + '_eligible_tasks.json'), [t for t in old if t['protein_id'] in allowed])
    application_panel = read(root / 'reports/recomb_general100_applications_20261017/panel.json')
    app_reviewed = set(policy['application_reviewed_ids'])
    write(out / 'application_review_queue.json', [dict(protein_id=p['protein_id'], dataset=p['dataset'],
        WT_sha256=p['sequence_sha256'], proposed_review_status='POSITIVE_METADATA_REVIEW' if p['protein_id'] in app_reviewed else 'EXCLUDE_OR_HOLD',
        executable=False, required='chain-polymer identity, per-accession function and evidence-source review') for p in application_panel])
    summary = dict(status='GENERAL_POSITIVE_SCOPE_CURATED_APPLICATIONS_STILL_GATED', source_WT=len(panel),
        retained_WT=len(retained), disposition=dict(Counter(r['status'] for r in rows)),
        retained_sources=dict(Counter(p['stratum'] for p in retained)),
        retained_subsets={k: len(v) for k, v in subsets.items()},
        parent_roster_sha256=sha(panel_path), mapping_sha256=sha(mapping_path), policy_sha256=sha(root / POLICY),
        model_results_used_for_selection=False, new_model_calls=0, new_structure_calls=0,
        original_files_modified=False, original_100_not_the_new_denominator=True,
        application_jobs_allowed=False, RAG_jobs_allowed=False,
        complete_structure_prediction_limit=None)
    write(out / 'completion.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().root)
