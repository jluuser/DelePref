"""Read-only source/outline/cache-coverage audit for the requested paper-table plan."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
SHRINK=Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/evocompress')
RAY=Path('/public/nfs-data/keyuan.zhou/evocompress/results/formal_v9/external_raygun/runs/V9-S6-raygun_official-native_20260901T055959Z_attempt01')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    out=ROOT/'reports/recomb_narrative_plan_20261017';out.mkdir(parents=True,exist_ok=True)
    files=[Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/AAAI-Protein-Shrink/main.tex'),
           SHRINK/'paper_writing/README.md',SHRINK/'paper_writing/sections/experiments.tex',
           SHRINK/'docs/general240_reorganized_20260916/REPORT_ZH.md',
           SHRINK/'results/iclr_completion_20260910/E1_E4/MAIN_PROTOCOLS.md',
           SHRINK/'results/iclr_completion_20260910/E1_E4/Raygun_native/manifest.json',
           SHRINK/'results/iclr_completion_20260910/E1_E4/code/predict_k8.py',
           SHRINK/'data/stage2_three_method_scientific_validation_v1/protocol.json',
           RAY/'checkpoint_snapshot.json',RAY/'protocol_snapshot.json',RAY/'completion.json']
    ray_checkpoint=json.loads((RAY/'checkpoint_snapshot.json').read_text())
    asset_stats={}
    for key in ['input_encoder','raygun_checkpoint']:
        d=ray_checkpoint[key];p=Path(d['path'])
        asset_stats[key]=dict(d,exists=p.is_file(),actual_bytes=p.stat().st_size if p.is_file() else None,
                              checksum_scope='Recorded frozen snapshot SHA; large checkpoint bytes not rehashed during planning')
        assert p.is_file() and p.stat().st_size==d['bytes']
    current=json.loads((ROOT/'reports/delepref_v8_native_panel/main_panel.json').read_text())
    current_by_sha={x['sequence_sha256']:x['protein_id'] for x in current};overlap=[]
    group_root=SHRINK/'results/iclr_completion_20260910/E1_E4/Raygun_native/groups'
    for p in sorted(group_root.glob('E1_*.json')):
        group=json.loads(p.read_text());digest=hashlib.sha256(group['wt_sequence'].encode()).hexdigest()
        if digest in current_by_sha:
            overlap.append(dict(current_protein_id=current_by_sha[digest],legacy_protein_id=group['protein_id'],
                                WT_sha256=digest,group_path=str(p),group_sha256=sha(p),
                                existing_records=len(group['existing_records'])))
    main_rows=SHRINK/'results/iclr_completion_20260910/E1_E4/FINAL/per_candidate.jsonl'
    counts=Counter();paths=0
    with main_rows.open() as f:
        for line in f:
            row=json.loads(line)
            if row['experiment']=='E1':
                counts[(row['method'],str(row['budget']))]+=1
                paths+=int(Path(row['prediction_path']).is_file())
    aaa=files[0].read_text();outline=[dict(line=i,text=l) for i,l in enumerate(aaa.splitlines(),1)
        if any(t in l for t in ['\\section','\\subsection','\\caption'])]
    result=dict(status='READ_ONLY_NARRATIVE_AND_BASELINE_REUSE_AUDIT_COMPLETE_NO_NEW_EXPERIMENTS',
        sources=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files],
        AAAI_outline=outline,Raygun_assets=asset_stats,Raygun_protocol=json.loads((RAY/'protocol_snapshot.json').read_text()),
        Raygun_completion=json.loads((RAY/'completion.json').read_text()),
        General240_existing_records=[dict(method=k[0],budget=k[1],rows=v) for k,v in sorted(counts.items())],
        General240_prediction_paths_present=paths,current100_exact_WT_overlap_with_native_Raygun=len(overlap),overlap=overlap,
        reuse_limits=['Exact WT overlap supports candidate reuse, not automatic equality of predictor/cache/coordinate contracts',
                     'Legacy ESMFold seed20260908 versus current20261017; strict cache identities differ',
                     'Legacy global TM-align and current fixed-correspondence TM are different endpoints; one common computation is needed',
                     'Raygun500 native ranked outputs are not8 raw candidates; native-pipeline comparisons must disclose search cost',
                     'Raygun permits substitutions/insertions; do not force a pure-deletion mask or silently discard such outputs'],
        paper_modified=False,new_GPU_jobs=0,new_model_forwards=0,new_structure_predictions=0)
    (out/'asset_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','General240_prediction_paths_present','current100_exact_WT_overlap_with_native_Raygun','new_GPU_jobs']}))


if __name__=='__main__':main()
