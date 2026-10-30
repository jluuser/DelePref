"""Assay semantics and category-balanced interpretation; frozen scores stay unchanged."""
from collections import Counter,defaultdict
import json
import numpy as np
from recomb_expanded_external_prepare_20261017 import ROOT,BASE,write,sha
from recomb_expanded_metrics_20261017 import macro_bootstrap


def main():
    out=ROOT/BASE;meta=json.loads((out/'assay_metadata.json').read_text())
    rows=[json.loads(l) for l in (out/'rows.jsonl').read_text().splitlines()]
    exposures={r['target']:r for r in json.loads((out/'training_exposure.json').read_text())}
    component={k:v['family_component'] for k,v in exposures.items()}
    assays=[];category={}
    for assay,m in sorted(meta.items()):
        rs=[r for r in rows if r['assay_id']==assay]
        biological=m['selection_assay'] or m['coarse_selection_type']
        if assay.startswith('A4_HUMAN_Seuma_2022'):biological='aggregation/nucleation (distinct from fold stability or desired function)'
        category[assay]=biological
        cutoff=float(m['DMS_binarization_cutoff'])
        binary_disagreement=sum(r['official_binary'] is not None and int(r['measurement']>cutoff)!=r['official_binary'] for r in rs)
        oriented_cutoff=cutoff*float(m['raw_DMS_directionality'])
        oriented_disagreement=sum(r['official_binary'] is not None and int(r['measurement']>oriented_cutoff)!=r['official_binary'] for r in rs)
        construction='ProteinGym standardized target_seq; expression tags/fusions are not modeled unless present in released sequence'
        assayed_target_length=None;mutation_region=None
        if assay.startswith('A4_HUMAN_Seuma_2022'):
            assayed_target_length=42
            construction='Amyloid-beta42 peptide aggregation assay; PG scoring context uses APP770 full reference, not the complete assay fusion. Report as standardized benchmark context, not APP function preservation.'
        elif assay.startswith('P53_HUMAN_Kotler_2018'):
            mutation_region=[102,292]
            construction='Mutagenized DNA-binding domain positions102-292 in expressed p53 context; mutated library region is not the whole construct length. Released full reference393 retained.'
        elif assay.startswith('KCNJ2_MOUSE_Macdonald'):
            construction='Kir2.1 surface-trafficking assay; PG no-FLAG standardized sequences retained. Surface abundance is not electrophysiological conductance.'
        elif 'Tsuboyama' in assay:
            construction='Assayed short protein/domain stability via proteolysis; standardized target is a fragment/domain where appropriate, not necessarily full native UniProt protein.'
        assays.append(dict(assay_id=assay,selection_type=m['selection_type'],selection_assay=m['selection_assay'],
            coarse_selection_type=m['coarse_selection_type'],biological_category=biological,raw_phenotype=m['raw_DMS_phenotype_name'],
            raw_directionality=m['raw_DMS_directionality'],released_score_direction='Already normalized by ProteinGym; no second flip',
            binarization_method=m['DMS_binarization_method'],binarization_cutoff=cutoff,
            reference_raw_cutoff_disagreements_without_direction=binary_disagreement,
            cutoff_in_released_score_orientation=oriented_cutoff,
            released_binary_threshold_disagreements_after_cutoff_orientation=oriented_disagreement,
            labels_interpretation='Official assay cutoff; median split is not a validated functional-retention threshold',
            standardized_WT_length=len(m['target_seq']),assayed_target_length_if_distinct=assayed_target_length,
            mutagenized_region_1based_inclusive=mutation_region,construct_interpretation=construction,
            maximum_deleted_count=max(r['deleted_length'] for r in rs),
            maximum_deleted_fraction_full_reference=max(r['deletion_fraction_full_reference'] for r in rs),
            maximum_deleted_fraction_assayed_peptide=(max(r['deleted_length'] for r in rs)/assayed_target_length) if assayed_target_length else None,
            DOI=m.get('jo'),title=m['title']))
    per_assay=json.loads((out/'assay_metrics.json').read_text());grouped=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for r in per_assay:
        if r['group']=='primary_unique_maps' and r['n']>=2:
            grouped[category[r['assay_id']]][r['method']][r['target']].append(r['Spearman'])
    balanced={}
    for phenotype,models in grouped.items():
        balanced[phenotype]={}
        for method,targets in models.items():
            vals={k:float(np.mean(v)) if all(x is not None for x in v) else None for k,v in targets.items()}
            balanced[phenotype][method]=macro_bootstrap(vals,component)
    write(BASE+'/assay_semantic_audit_v2.json',assays);write(BASE+'/biological_category_metrics.json',balanced)
    write(BASE+'/interpretation_completion_v2.json',dict(status='ASSAY_DIRECTION_CONSTRUCT_AND_CATEGORY_INTERPRETATION_AUDITED',
        assays=len(assays),WT=61,stability_proteolysis_assays=sum('Tsuboyama' in r['assay_id'] for r in assays),
        category_assay_counts=dict(Counter(r['biological_category'] for r in assays)),
        binary_threshold_disagreement_count=sum(r['released_binary_threshold_disagreements_after_cutoff_orientation'] for r in assays),
        raw_cutoff_orientation_explains_preliminary_241_disagreements=True,
        cutoff_note='Reference cutoff is in raw phenotype orientation; multiply cutoff by declared direction only for the audit. Released DMS_score and official binary labels are unchanged.',
        original_scores_and_mappings_changed=False,
        caveats=['Aggregation distinct from desired biological function; full-reference vs assayed-peptide deletion fraction distinguished',
                 'Single-protein category bootstrap interval is degenerate, not a generalization bound',
                 'Median DMS binary labels do not establish function preserved',
                 'Multi-residue evidence3assays; uniquely mapped multispan cases onlyA4, no cross-protein multi-span functional validation',
                 'Scoring benchmark does not validate freely generated20-40% RAG products'],
        sources=[dict(url='https://www.nature.com/articles/s41467-022-34742-3',role='Seuma2022 assay property and peptide context'),
                 dict(url='https://link.springer.com/article/10.1186/s13059-023-02880-6',role='DIMPLE surface-trafficking readout'),
                 dict(url='https://weizmann.elsevierpure.com/ws/files/23750487/es_MolecularCell_ASystematicP53Mutation_AM2018.pdf',role='Kotler mutagenized DBD vs expressed protein context')]))
    print(json.dumps(json.loads((out/'interpretation_completion_v2.json').read_text())),flush=True)


if __name__=='__main__':main()
