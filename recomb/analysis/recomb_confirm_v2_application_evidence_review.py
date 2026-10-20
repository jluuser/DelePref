"""Apply the unchanged reviewed-reference policy, reusing prior metadata caches."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
from recomb_confirm_v2_diagnostics import read,sha,write


def paths(root,cohort):
    return root/('retrieval/recomb_confirm_v2_'+cohort),root/('reports/recomb_confirm_v2_'+cohort+'_evidence_review')


def inventory(root,cohort):
    source,out=paths(root,cohort)
    if (out/'inventory.json').exists():raise FileExistsError('Reuse existing application evidence inventory')
    done=read(source/'completion.json');raw=source/'profiles_unreviewed_evidence.json'
    if sha(raw)!=done['profiles_sha256']:raise ValueError('Application unreviewed profile identity changed')
    profiles=read(raw);targets=defaultdict(set);queries={}
    if len(profiles)!=done['WT']:raise ValueError('Application query count differs')
    for pid,p in profiles.items():
        if p['evidence_status']!='UNREVIEWED_NOT_FOR_GENERATION':raise ValueError('Unexpected evidence release state')
        queries[pid]=[]
        for h in p['representatives']:
            if not h['complete_target_identity_checked']:raise ValueError('Full target identity not checked')
            targets[h['target']].add(h['target_sequence_sha256']);queries[pid].append(h['target'])
    write(out/'inventory.json',dict(status='APPLICATION_EVIDENCE_METADATA_INVENTORY_ONLY',cohort=cohort,
        profiles_sha256=sha(raw),targets={k:sorted(v) for k,v in sorted(targets.items())},query_targets=queries,
        raw_sequences_exported=False,new_model_calls=0,RAG_generation_allowed=False))
    print(dict(cohort=cohort,queries=len(queries),unique_representatives=len(targets)),flush=True)


def metadata(root,cohort):
    _,out=paths(root,cohort)
    if (out/'completion.json').exists():raise FileExistsError('Metadata already completed')
    claim=out/'metadata_execution_claim.json'
    if claim.exists():raise FileExistsError('Existing metadata process may be active; inspect before resubmitting')
    inv=read(out/'inventory.json');wanted=set(inv['targets'])
    write(claim,dict(process_id=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),cohort=cohort))
    copied_batches=copied_taxa=0
    for folder in [root/'reports/recomb_confirm_v2_General100_evidence_review']:
        for name in ['uniref_batches','taxonomy']:
            target=out/name;target.mkdir(parents=True,exist_ok=True)
            for p in (folder/name).glob('*.json'):
                d=read(p)
                if d.get('status')!='SUCCEEDED':continue
                if name=='uniref_batches' and not any(r.get('target') in wanted for r in d.get('rows',[])):continue
                dest=target/p.name
                if dest.exists():
                    if sha(dest)!=sha(p):raise ValueError('Conflicting successful metadata cache')
                else:
                    shutil.copyfile(p,dest)
                    if name=='uniref_batches':copied_batches+=1
                    else:copied_taxa+=1
    import recomb_confirm_v2_General100_evidence_metadata as parent
    parent.OUT=out
    write(out/'reuse_manifest.json',dict(cohort=cohort,source_policy='same numeric/source/function/full-sequence-identity policy as General100',
        copied_successful_uniref_batches=copied_batches,copied_taxonomy_records=copied_taxa,
        parent_source_sha256=sha(Path(parent.__file__)),wrapper_source_sha256=sha(Path(__file__)),
        inventory_sha256=sha(out/'inventory.json'),scope_approval_of_new_targets=False,new_model_calls=0))
    parent.main()
    d=read(out/'completion.json');d['parent_implementation_status']=d['status']
    d.update(status='APPLICATION_REFERENCE_METADATA_REVIEW_COMPLETE_PROFILE_BUILD_PENDING',cohort=cohort,
        reuse_manifest_sha256=sha(out/'reuse_manifest.json'))
    write(out/'completion.json',d)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inventory','metadata']);p.add_argument('--cohort',choices=['Binder60','Enzyme80'],required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    (inventory if a.mode=='inventory' else metadata)(a.root,a.cohort)
