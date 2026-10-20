"""Generate descriptive manuscript assets from frozen data; no model inference."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

STRATA = ['K8 E1 (primary)', 'Earlier E1', 'Selected sensitivity']
COLORS = ['#27647c', '#64a0b0', '#abd0cf', '#e5c596', '#bd6b57']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def hierarchy(values, keys=('protein_id', 'budget', 'generator')):
    if not keys:
        return np.mean([v['value'] for v in values], axis=0)
    grouped = collections.defaultdict(list)
    for v in values:
        grouped[str(v[keys[0]])].append(v)
    return np.mean([hierarchy(g, keys[1:]) for g in grouped.values()], axis=0)


def stratum(row):
    if row['primary_eligible']:
        return STRATA[0]
    return STRATA[1] if row['source'] == 'e1_exact' else STRATA[2]


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def diagram(path):
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.set(xlim=(0, 6.5), ylim=(0, 3.0)); ax.axis('off')
    def box(x, y, w, h, text, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                     edgecolor='#53636a', facecolor=fill, linewidth=.9))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=11)
    def arrow(a, b, dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>', mutation_scale=11,
                     lw=1, color='#53636a', linestyle='--' if dashed else '-'))
    box(.08,1.78,1.3,.7,'WT + deletion\ncandidate', '#edf3f5')
    box(1.7,1.78,1.3,.7,'Frozen ESMC\naligned residues', '#dcecf1')
    box(3.32,1.78,1.25,.7,'Symmetric\n5-bin head', '#d8ece5')
    box(4.92,1.78,1.45,.7,'Pair preservation\nG and T = C G', '#eceddf')
    for a,b in [(1.38,1.7),(3.0,3.32),(4.57,4.92)]:arrow((a,2.13),(b,2.13))
    box(1.7,.7,1.3,.62,'WT reference\n+ deletion mask', '#edf3f5')
    arrow((3.0,1.0),(3.93,1.78))
    box(4.85,.45,1.52,.85,'Variant structure\nlabels: training\nand evaluation', '#f5e6df')
    arrow((5.6,1.30),(4.45,1.78),True)
    ax.text(.08,.95,'Use-time inputs',fontsize=10,color='#425762')
    ax.text(.08,.40,'Variant coordinates are not model inputs.',fontsize=10,color='#425762')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    fig.savefig(path, bbox_inches='tight', pad_inches=.04); plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.root).resolve();out=Path(a.output).resolve()
    if root not in out.parents or out.name!='paper':
        raise ValueError('paper assets must remain in the new experiment paper directory')
    seal=json.loads((root/'configs/execution_freeze.json').read_text())
    for name, expected in seal['input_hashes'].items():
        if sha(root/name)!=expected:raise ValueError('frozen input changed: '+name)
    rows=jsonl(root/'manifests/development.jsonl')
    labels={r['candidate_id']:r for r in jsonl(root/'reports/label_candidates.jsonl')}
    splits=json.loads((root/'manifests/splits.json').read_text())
    confirmation=jsonl(root/'manifests/confirmation_wt_frozen.jsonl')
    reserves=jsonl(root/'manifests/confirmation_wt_reserves_frozen.jsonl')
    generated=out/'generated';figures=out/'figures'
    generated.mkdir(parents=True,exist_ok=True);figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                         'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    summary=[];matrix=[];distributions=[]
    for name in STRATA:
        subset=[r for r in rows if stratum(r)==name]
        proteins={r['protein_id'] for r in subset}
        raw=np.sum([labels[r['candidate_id']]['class_counts'] for r in subset],axis=0)
        normalized=[]
        for r in subset:
            counts=np.array(labels[r['candidate_id']]['class_counts'],dtype=float)
            if counts.sum():normalized.append({**r,'value':counts/counts.sum()})
        weighted=hierarchy(normalized)
        distributions.append(weighted)
        summary.append({'stratum':name,'candidates':len(subset),'WT':len(proteins),
                        'train_WT':sum(splits['assignment'][p]=='train' for p in proteins),
                        'validation_WT':sum(splits['assignment'][p]=='validation' for p in proteins),
                        'unknown_pairs':sum(labels[r['candidate_id']]['unknown_pairs'] for r in subset),
                        'known_pairs':int(raw.sum()),
                        'class_counts':raw.tolist(),'protein_balanced_class_fractions':weighted.tolist()})
        for (budget,generator), group in sorted(_groups(subset).items()):
            spans=np.array([r['morphology']['num_spans'] for r in group])
            matrix.append({'stratum':name,'budget':budget,'generator':generator,'WT':len({r['protein_id'] for r in group}),
                           'candidates':len(group),'span_count_median':float(np.median(spans)),
                           'span_count_min':int(spans.min()),'span_count_max':int(spans.max()),
                           'terminal_fraction':float(np.mean([r['morphology']['terminal'] for r in group])),
                           'direct_coverage_mean':float(hierarchy([{**r,'value':r['direct_relationship_coverage']} for r in group])),
                           'contact_loss_mean':float(hierarchy([{**r,'value':r['direct_contact_loss']} for r in group]))})
    csv_write(generated/'source_budget_morphology.csv',matrix)
    csv_write(generated/'source_counts.csv',[{k:v for k,v in s.items() if k not in ['class_counts','protein_balanced_class_fractions']} for s in summary])
    unique_wt={r['protein_id']:r for r in rows}
    primary=[r for r in rows if r['primary_eligible']]
    duplicate=collections.defaultdict(list)
    for r in rows:duplicate[r['sequence_sha256']].append(r)
    duplicate_records=[{'variant_sequence_sha256':seq,'candidate_id':r['candidate_id'],'protein_id':r['protein_id'],'mask_sha256':r['mask_sha256']} for seq, group in duplicate.items() if len(group)>1 for r in group]
    if duplicate_records:csv_write(generated/'distinct_mappings_same_sequence.csv',duplicate_records)
    manifest_counts=seal['counts']
    report={'status':'PAPER_DATA_PREPARATION_COMPLETE','inference_performed':False,'new_structure_calls':0,
            'source_summary':summary,'source_budget_morphology':matrix,'counts':manifest_counts,
            'development_reference_scope':dict(collections.Counter(r['coordinate_scope'] for r in unique_wt.values())),
            'WT_reference_coverage_min':min(r['wt_ca_coverage'] for r in unique_wt.values()),
            'WT_reference_coverage_median':float(np.median([r['wt_ca_coverage'] for r in unique_wt.values()])),
            'confirmation_reference_kind':dict(collections.Counter(r['reference_kind'] for r in confirmation)),
            'confirmation_CA_coverage_min':min(r['wt_ca_coverage'] for r in confirmation),
            'confirmation_lengths':[len(r['wt_sequence']) for r in confirmation],
            'confirmation_WT':len(confirmation),'reserves':len(reserves),
            'input_hashes':seal['input_hashes'],'analysis_code_sha256':sha(__file__),
            'class_fraction_weighting':'protein -> budget -> generator -> candidate -> valid relationship',
            'interpretation':'Descriptive data and label audit, not learned-model performance.'}
    (generated/'data_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    macros={'DevRecords':len(rows),'UniqueVariants':len(duplicate),'AuditedWT':len(unique_wt),
            'PrimaryRecords':len(primary),'PrimaryWT':summary[0]['WT'],'PrimaryTrainWT':summary[0]['train_WT'],
            'PrimaryValWT':summary[0]['validation_WT'],'EarlierRecords':summary[1]['candidates'],
            'SelectedRecords':summary[2]['candidates'],'TrainWT':manifest_counts['train_wt'],
            'ValWT':manifest_counts['validation_wt'],'ConfirmWT':len(confirmation),'ReserveWT':len(reserves),
            'UniquePLMSequences':manifest_counts['unique_plm_sequences'],'KnownPairs':sum(s['known_pairs'] for s in summary),
            'UnknownPairs':sum(s['unknown_pairs'] for s in summary)}
    def number(v):return f'{v:,}'.replace(',',r'\,')
    (generated/'counts.tex').write_text(''.join((r'\newcommand{\%s}{%s}' % (k,number(v)))+'\n' for k,v in macros.items()))
    table=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Stratum & Candidates & WT & Train WT & Val. WT \\',r'\midrule']
    for s in summary:table.append(f"{s['stratum']} & {number(s['candidates'])} & {s['WT']} & {s['train_WT']} & {s['validation_WT']} " + r'\\')
    table += [r'\bottomrule',r'\end{tabular}']
    (generated/'source_table.tex').write_text('\n'.join(table)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.1),gridspec_kw={'width_ratios':[1,1.4]})
    counts=collections.Counter(r['protein_id'] for r in rows if r['source']=='e1_exact')
    xs=sorted(set(counts.values()));ys=[sum(v==x for v in counts.values()) for x in xs]
    axes[0].bar([str(x) for x in xs],ys,color='#27647c',width=.6)
    axes[0].set(xlabel='E1 candidates per WT',ylabel='Number of WT',title='Candidate imbalance')
    left=np.zeros(3)
    for i,label in enumerate(['<0.5','0.5-1','1-2','2-4','>=4']):
        values=np.array(distributions)[:,i]
        axes[1].barh(np.arange(3),values,left=left,color=COLORS[i],label=label,height=.55);left+=values
    axes[1].set_yticks(np.arange(3),['K8 E1','Earlier E1','Selected'])
    axes[1].set(xlabel='Protein-balanced label fraction',xlim=(0,1),title='Distance-error bins (\N{LATIN CAPITAL LETTER A WITH RING ABOVE})')
    axes[1].invert_yaxis();axes[1].legend(ncol=3,loc='upper center',bbox_to_anchor=(.42,-.3),frameon=False,columnspacing=.7,handlelength=1)
    fig.tight_layout();fig.subplots_adjust(bottom=.33)
    fig.savefig(figures/'data_audit.pdf',bbox_inches='tight',pad_inches=.04)
    fig.savefig(figures/'data_audit.png',dpi=180,bbox_inches='tight',pad_inches=.04);plt.close(fig)
    diagram(figures/'method_overview.pdf')
    print(json.dumps({'status':report['status'],'source_summary':summary,'figures':['data_audit.pdf','method_overview.pdf']}))


def _groups(rows):
    groups=collections.defaultdict(list)
    for row in rows:groups[(row['budget'],row['generator'])].append(row)
    return groups


if __name__=='__main__':main()
