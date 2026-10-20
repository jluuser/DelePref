"""Actual original-training exposure and WT-family components, without dropping targets."""
from collections import defaultdict,Counter
import json
import os
from pathlib import Path
import subprocess
import time
from recomb_expanded_external_prepare_20261017 import ROOT,BASE,sha,write
from delepref_v8_panel import MMSEQS,FORMAT,alignment,passes,Components


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    out=ROOT/BASE
    if (out/'homology_completion.json').exists():raise FileExistsError('exposure audit already completed')
    cohort=json.loads((out/'cohort_freeze.json').read_text())
    frozen_inputs={n:h for n,h in cohort['files'].items() if n in ('inputs.json','rows.jsonl','assay_metadata.json','protocol.json','wildtypes.fasta','retrieval_exclusion_sha256.json')}
    for n,h in frozen_inputs.items():assert sha(out/n)==h,n
    write(BASE+'/cohort_verified_manifest.json',dict(files=frozen_inputs,
        original_cohort_freeze_sha256=sha(out/'cohort_freeze.json'),
        note='Preparation stdout log is volatile and not a scientific input; only declared immutable data/protocol files are bound here. No data or cohort changed.'))
    reference=json.loads((ROOT/'reports/delepref_v8/train_reference_completion.json').read_text())
    assert reference['train_WT']==269915
    assert sha(reference['fasta_path'])==reference['fasta_sha256'] and sha(MMSEQS)==reference['mmseqs_binary_sha256']
    wildtypes=json.loads((out/'inputs.json').read_text())['wildtypes'];queries={k:dict(sequence=v) for k,v in wildtypes.items()}
    start=time.time();fasta=out/'wildtypes.fasta'
    options=['--threads','16','-s','7.5','-e','0.001','--min-seq-id','0.2','--max-seqs','512','--split-memory-limit','24G','--format-output',FORMAT]
    for label,target in [('train',reference['database_prefix']),('self',str(fasta))]:
        command=[str(MMSEQS),'easy-search',str(fasta),target,str(out/(label+'_hits.tsv')),str(out/('tmp_'+label))]+options
        run=subprocess.run(command,text=True,capture_output=True);(out/(label+'_search.log')).write_text(run.stdout+run.stderr)
        if run.returncode:raise RuntimeError('homology search failed: '+label)
    hits=defaultdict(list)
    for line in (out/'train_hits.tsv').read_text().splitlines():
        h=alignment(line.split('\t'),queries);hits[h['query']].append(h)
    components=Components(wildtypes)
    for line in (out/'self_hits.tsv').read_text().splitlines():
        h=alignment(line.split('\t'),queries)
        if passes(h,.3):components.union(h['query'],h['target'])
    exact_hashes=set()
    # Exact sequence overlap is independently verified from the original training FASTA.
    import hashlib
    seq=[]
    with Path(reference['fasta_path']).open() as f:
        for line in f:
            if line.startswith('>'):
                if seq:exact_hashes.add(hashlib.sha256(''.join(seq).encode()).hexdigest());seq=[]
            else:seq.append(line.strip())
        if seq:exact_hashes.add(hashlib.sha256(''.join(seq).encode()).hexdigest())
    rows=[]
    for target,seq in sorted(wildtypes.items()):
        hs=hits[target];covered=[h for h in hs if min(h['query_coverage'],h['target_coverage'])>=.8]
        nearest=max(hs,key=lambda x:(x['identity_shorter_full'],x['bits'],x['target'])) if hs else None
        rows.append(dict(target=target,WT_length=len(seq),WT_sha256=hashlib.sha256(seq.encode()).hexdigest(),
            exact_training_sequence_overlap=hashlib.sha256(seq.encode()).hexdigest() in exact_hashes,
            threshold40_both80=any(passes(h,.4) for h in hs),hits=len(hs),hit_cap_reached=len(hs)>=512,
            nearest_returned=nearest,nearest_both80=max(covered,key=lambda x:x['identity_shorter_full']) if covered else None,
            family_component=components.find(target)))
    write(BASE+'/training_exposure.json',rows)
    write(BASE+'/homology_completion.json',dict(status='EXPANDED_WT_ACTUAL_TRAINING_EXPOSURE_AUDITED',
        job_id=os.environ['SLURM_JOB_ID'],seconds=time.time()-start,WT=len(rows),training_WT=269915,
        exact_overlaps=sum(r['exact_training_sequence_overlap'] for r in rows),threshold_exposed=sum(r['threshold40_both80'] for r in rows),
        family_components=len({r['family_component'] for r in rows}),rows_removed=0,
        scope='Nearest returned sensitive MMseqs hits plus exact FASTA hashes; no-hit is not remote-homology absence',
        outputs={n:sha(out/n) for n in ['cohort_verified_manifest.json','training_exposure.json','train_hits.tsv','self_hits.tsv']}))
    print(json.dumps(json.loads((out/'homology_completion.json').read_text())),flush=True)


if __name__=='__main__':main()
