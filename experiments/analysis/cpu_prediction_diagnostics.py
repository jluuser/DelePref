"""CPU audit of frozen first-seed prediction artifacts; no model imports or inference.

Run with --config configs/development_diagnostics_first_seed.json. All outputs are
restricted to reports/cpu_development_diagnostics/prediction. Unknown labels are
excluded from correctness metrics and retained explicitly in coverage accounting.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import development_review as core
import review_scalar_baselines as scalar

common = core.common
VALUES = np.array([1., .75, .5, .25, 0.])
THRESHOLDS = np.array([.5, 1., 2., 4.], dtype=np.float32)
READY = 'CPU_PREDICTION_DIAGNOSTICS_COMPLETE'
INTERPRETATION = ('Exploratory historical primary-validation development diagnostics specified after '
    'first-seed validation CE choices and before this pair-level audit. Learning-rate selection '
    'shares validation data. No independent confirmation, scientific pass threshold, or folding authorization.')


def close(actual, expected, name, tolerance=1e-6):
    core.require(np.isfinite(actual) and np.isfinite(expected) and abs(actual-expected)<=tolerance,
                 name+' differs')


def checked_hash(path, expected, name):
    actual = common.sha_file(path)
    core.equal(actual, expected, name+' hash')
    return actual


def validate_probabilities(probabilities, pair_ids, retained_ids):
    p = np.asarray(probabilities)
    core.require(p.shape==(len(retained_ids),5) and np.array_equal(pair_ids,retained_ids),
                 'prediction retained pair IDs/shape differ')
    core.require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(), 'invalid probabilities')
    core.require(np.all(np.abs(p.sum(axis=1,dtype=float)-1.)<=1e-5), 'probability row sums differ')


def integer_array(value, name):
    value=np.asarray(value)
    core.require(value.ndim==1 and value.dtype.kind in 'iu', name+' must be an integer vector')
    return value


def validate_labels(lab, row, summary, reference=None):
    """Check frozen identities/mappings only; never read variant structures."""
    cid=row['candidate_id']; length=len(row['wt_sequence'])
    mask=np.asarray(row['mask'])
    core.require(mask.shape==(length,) and np.isin(mask,[0,1]).all(),cid+' deletion mask invalid')
    kept=integer_array(lab['kept'],'kept')
    core.require(np.array_equal(kept,np.flatnonzero(mask==0)),cid+' kept deletion map differs')
    core.equal(''.join(row['wt_sequence'][i] for i in kept),row['sequence'],cid+' edited sequence')
    ri=integer_array(lab['reference_i'],'reference_i'); rj=integer_array(lab['reference_j'],'reference_j')
    core.require(ri.shape==rj.shape and ((ri>=0)&(rj<length)&(rj-ri>=3)).all(),cid+' reference indices invalid')
    core.require(len(np.unique(ri*length+rj))==len(ri),cid+' duplicate reference pairs')
    if reference is not None:
        core.require(np.array_equal(ri,reference[0]) and np.array_equal(rj,reference[1]),cid+' WT reference arrays differ')
    inverse=np.full(length,-1,dtype=int); inverse[kept]=np.arange(len(kept))
    retained=(inverse[ri]>=0)&(inverse[rj]>=0)
    core.require(np.asarray(lab['retained_mask']).dtype.kind=='b' and
                 np.array_equal(lab['retained_mask'],retained),cid+' retained mask differs')
    known=integer_array(lab['pair_ids'],'pair_ids'); unknown=integer_array(lab['unknown_pair_ids'],'unknown_pair_ids')
    for ids in (known,unknown):
        core.require(((ids>=0)&(ids<len(ri))).all() and (np.diff(ids)>0).all(),cid+' unordered/out-of-range pair IDs')
    core.require(not np.intersect1d(known,unknown).size and
                 np.array_equal(np.sort(np.r_[known,unknown]),np.flatnonzero(retained)),cid+' known/unknown partition differs')
    for key,expected in [('wt_i',ri[known]),('wt_j',rj[known]),
                         ('edited_i',inverse[ri[known]]),('edited_j',inverse[rj[known]])]:
        actual=integer_array(lab[key],key)
        core.require(np.array_equal(actual,expected),cid+' inverse mapping '+key+' differs')
    target=integer_array(lab['target'],'target'); error=np.asarray(lab['error'])
    core.require(error.shape==known.shape and np.isfinite(error).all() and (error>=0).all(),cid+' label error invalid')
    core.require(np.array_equal(target,np.searchsorted(THRESHOLDS,error,side='right')),cid+' threshold target differs')
    core.equal(np.bincount(target,minlength=5).tolist(),summary['class_counts'],cid+' class counts')
    n=len(ri); k=len(known); u=len(unknown); r=int(retained.sum()); contribution=float(VALUES[target].sum())
    for key,expected in [('reference_pairs',n),('retained_pairs',r),('known_pairs',k),('unknown_pairs',u)]:
        core.equal(summary[key],expected,cid+' '+key)
    for key,value in {'C':r/n if n else None,'G_lower':contribution/r if r else None,
                      'G_upper':(contribution+u)/r if r else None,
                      'T_lower':contribution/n if n else 0.,'T_upper':(contribution+u)/n if n else 0.}.items():
        core.equal(summary[key],value,cid+' '+key)
    return dict(reference_pairs=n,retained_pairs=r,known_pairs=k,unknown_pairs=u,
                class_counts=summary['class_counts'])


def bin_indices(probability):
    # searchsorted fixed decimal edges avoids 0.3*10 boundary rounding surprises.
    return np.minimum(np.searchsorted(np.arange(11)/10.,probability,side='right')-1,9)


def reliability(probability, observed):
    bins=bin_indices(probability); n=len(probability)
    return np.stack([np.bincount(bins,minlength=10),
                     np.bincount(bins,weights=probability,minlength=10),
                     np.bincount(bins,weights=observed,minlength=10)],axis=1)/n


def sufficient_statistics(probabilities, target):
    p=np.asarray(probabilities,dtype=float); y=np.asarray(target,dtype=int)
    if not len(y): return None
    core.require(p.shape==(len(y),5) and ((y>=0)&(y<5)).all(),'metric target shape/range differs')
    predicted=p.argmax(1); truth=np.eye(5)[y]; g=p@VALUES; true=VALUES[y]
    confusion=np.bincount(y*5+predicted,minlength=25).reshape(5,5)/len(y)
    return dict(ce=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-30)).mean()),
                multiclass_brier=float(np.square(p-truth).sum(1).mean()),
                expected_preservation_mae=float(np.abs(g-true).mean()),
                expected_preservation_bias=float((g-true).mean()),
                true_preservation=float(true.mean()),expected_preservation=float(g.mean()),
                normalized_confusion=confusion.tolist(),true_class_fraction=truth.mean(0).tolist(),
                mean_class_probability=p.mean(0).tolist(),predicted_argmax_fraction=(np.bincount(predicted,minlength=5)/len(y)).tolist(),
                top_reliability=reliability(p.max(1),(predicted==y).astype(float)).tolist(),
                class_reliability=np.stack([reliability(p[:,c],truth[:,c]) for c in range(5)]).tolist())


def finish_reliability(array):
    values=np.asarray(array)
    return {'ece':float(np.abs(values[:,2]-values[:,1]).sum()),'bins':[
        dict(lower=i/10,upper=(i+1)/10,weight=float(w),predicted_sum=float(p),observed_sum=float(o),
             mean_predicted=float(p/w) if w else None,mean_observed=float(o/w) if w else None)
        for i,(w,p,o) in enumerate(values)]}


def finish_statistics(statistics):
    result={k:v for k,v in statistics.items() if k not in ('top_reliability','class_reliability')}
    confusion=np.asarray(result['normalized_confusion'])
    result['confusion_recall']=[float(confusion[c,c]/confusion[c].sum()) if confusion[c].sum() else None for c in range(5)]
    result['confusion_precision']=[float(confusion[c,c]/confusion[:,c].sum()) if confusion[:,c].sum() else None for c in range(5)]
    result['top_label_reliability']=finish_reliability(statistics['top_reliability'])
    result['per_class_reliability']=[finish_reliability(a) for a in statistics['class_reliability']]
    return result


def hierarchy_statistics(rows, keys=('protein_id','budget','generator')):
    if keys:
        groups={}
        for row in rows: groups.setdefault(str(row[keys[0]]),[]).append(row)
        values=[hierarchy_statistics(part,keys[1:]) for part in groups.values()]
    else: values=[r['statistics'] for r in rows]
    return {key:np.mean([np.asarray(v[key]) for v in values],axis=0).tolist() for key in values[0]}


def aggregate(rows, expected_wts=None):
    core.require(bool(rows),'empty metric hierarchy')
    observed=sorted({r['protein_id'] for r in rows})
    if expected_wts is not None: core.equal(observed,sorted(expected_wts),'metric WT roster')
    result={'n_candidates':len(rows),'expected_wts':observed,'complete_case_filtering':False}
    counts={}
    for name in ('reference_pairs','retained_pairs','known_pairs','unknown_pairs','eligible_known_pairs',
                 'eligible_retained_pairs','eligible_unknown_pairs','eligible_reference_pairs'):
        if all(name in r.get('coverage',{}) for r in rows): counts[name]=sum(r['coverage'][name] for r in rows)
    if all('class_counts' in r.get('coverage',{}) for r in rows):
        counts['class_counts']=np.sum([r['coverage']['class_counts'] for r in rows],axis=0).tolist()
    result['coverage_counts']=counts
    absent=[dict(candidate_id=r['candidate_id'],protein_id=r['protein_id']) for r in rows if r['statistics'] is None]
    if absent:
        result.update(status='UNDEFINED_ZERO_ELIGIBLE_KNOWN_PAIRS',undefined_candidates=absent)
    else: result.update(status='OBSERVED',**finish_statistics(hierarchy_statistics(rows)))
    return result


def protected_output(root, output, paths):
    output=common.confined_path(output).resolve()
    for name in ('paper','code','configs','manifests','labels','checkpoints','provenance'):
        scalar.disjoint_output(output,root/name,name)
    scalar.reject_input_overlap(output,paths)
    return output


def load_context(config_path, kind):
    """Revalidate sealed cohort, ten-run inventory, selected artifacts and snapshot."""
    cfg=core.read_json(config_path); root=Path(cfg['root']).resolve()
    core.equal(root,common.AUTHORIZED_ROOT.resolve(),'authorized experiment root')
    core.equal(cfg['schema'],'first_seed_cpu_diagnostics_v1','diagnostic schema')
    core.equal(cfg['seed'],core.SEED,'diagnostic seed'); core.equal(cfg['arms'],list(core.ARMS),'diagnostic arms')
    definitions=cfg['prediction']
    for key,value in {'reliability_bins':10,'bin_edges':(np.arange(11)/10.).tolist(),
                      'rightmost_edge_inclusive':True,'preservation_values':VALUES.tolist(),
                      'probability_sum_absolute_tolerance':1e-5,
                      'recomputed_CE_and_Ghat_absolute_tolerance':1e-6}.items():
        core.equal(definitions[key],value,'prespecified prediction definition '+key)
    for key,value in {'bin_width':.05,'minimum_candidates':2,'range_maximum':.05}.items():
        core.equal(cfg['historical_static_matching'][key],value,'prespecified matching definition '+key)
    paths={k:([root/Path(v) for v in value] if isinstance(value,list) else root/Path(value)) for k,value in cfg['inputs'].items()}
    output=Path(root/cfg['outputs'][kind]).resolve()
    core.equal(output,root/'reports/cpu_development_diagnostics'/kind,'dedicated diagnostic output')
    flat=[p for v in paths.values() for p in (v if isinstance(v,list) else [v])]
    protected_output(root,output,[Path(config_path),*flat])
    seal=core.read_json(paths['freeze']); protocol=core.read_json(paths['protocol'])
    core.equal(Path(protocol['root']).resolve(),root,'protocol root')
    for key,value in {'ca_cutoff_angstrom':15.,'min_sequence_separation':3,'contact_cutoff_angstrom':8.,
                      'error_thresholds':THRESHOLDS.tolist(),'preservation_values':VALUES.tolist()}.items():
        core.equal(protocol['labels'][key],value,'frozen label definition '+key)
    identities={k:scalar.frozen_hash(root,paths[field],seal,k) for k,field in
                [('protocol','protocol'),('manifest','primary_manifest'),('splits','splits'),('label_summary_sha256','label_summary')]}
    learned=core.read_json(paths['core_report']); core.equal(learned['status'],core.READY,'core readiness')
    expected={(a,lr,core.SEED) for a in core.ARMS for lr in protocol['train']['learning_rates']}
    inventory=learned['runs']
    core.require(learned['expected_runs']==10 and learned['completed_runs']==10 and len(inventory)==10
                 and {(r['arm'],r['lr'],r['seed']) for r in inventory}==expected
                 and all(r['state']=='COMPLETE' for r in inventory),'all ten completed runs required')
    for key,value in identities.items(): core.equal(learned['identities'][key],value,'core input '+key)
    cache={str(p):common.sha_file(p) for p in paths['cache_indices']}
    core.equal(cache,learned['identities']['cache_index_sha256'],'cache index hashes')
    checked_hash(paths['freeze'],learned['identities']['execution_freeze_sha256'],'core execution seal')
    checked_hash(paths['dispatch'],learned['identities']['dispatch_sha256'],'core dispatch')
    checked_hash(core.__file__,learned['identities']['review_code_sha256'],'core implementation')
    dispatch=core.read_json(paths['dispatch']); snapshot=Path(dispatch['code_snapshot']).resolve()
    scalar.disjoint_output(output,snapshot,'frozen snapshot')
    hashes={p.relative_to(snapshot).as_posix():common.sha_file(p) for p in snapshot.rglob('*.py')}
    core.equal(hashes,dispatch['code_hashes'],'snapshot hashes'); core.equal(hashes,learned['identities']['code_hashes'],'core snapshot hashes')
    core.equal(str(snapshot),learned['identities']['code_snapshot'],'snapshot path')
    split=core.read_json(paths['splits']); core.equal(split['status'],'SPLIT_FROZEN','split status')
    core.equal(split['protocol_sha256'],identities['protocol'],'split protocol')
    primary=common.read_jsonl(paths['primary_manifest'])
    core.keyed(primary,lambda r:r['candidate_id'],'primary candidate')
    core.require(all(r['source']==protocol['primary_source'] and r['generator']!='mixed' for r in primary),'primary source differs')
    valid=[r for r in primary if split['assignment'][r['protein_id']]=='validation']
    roster={wt:split['component'][wt] for wt in sorted({r['protein_id'] for r in valid})}
    core.equal(roster,learned['validation_wt_components'],'primary validation roster')
    core.equal(len(roster),cfg['primary_validation_WT'],'primary validation WT count')
    core.equal(len(valid),cfg['primary_validation_candidates'],'primary validation candidates')
    core.equal(len(valid),learned['frozen_validation_candidates'],'core candidate count')
    summaries=core.keyed(common.read_jsonl(paths['label_summary']),lambda r:r['candidate_id'],'label summary')
    core.require(all(r['candidate_id'] in summaries for r in valid),'missing validation label summary')
    evaluations={}
    computational={k:v for k,v in hashes.items() if 'tests' not in Path(k).parts}
    for run in inventory:
        directory=Path(run['directory']); scalar.disjoint_output(output,directory,'run')
        validation=Path(run['validation_output'])
        # Resolve every nested input before hashing. A run directory can be
        # disjoint while an individual symlink points into our output subtree.
        scalar.reject_input_overlap(output,[*(directory/name for name in run['artifact_hashes']),
            Path(run['checkpoint_path']),validation,*(validation/name for name in run['output_hashes'])])
        core.require(set(run['artifact_hashes'])=={'run.json','status.json','train_trace.jsonl','validation_trace.jsonl'},'incomplete run artifact identities')
        for name,value in run['artifact_hashes'].items(): checked_hash(directory/name,value,'run '+name)
        provenance=core.read_json(directory/'run.json')
        for key,value in identities.items(): core.equal(provenance[key],value,'run input '+key)
        core.equal(provenance['cache_index_sha256'],cache,'run caches'); core.equal(provenance['code'],computational,'run frozen code')
        for key,value in {'arm':run['arm'],'lr':run['lr'],'seed':core.SEED,'technical':False,'steps':None}.items():
            core.equal(provenance['config'][key],value,'run config '+key)
        status=core.read_json(directory/'status.json'); core.equal(status['status'],'DEVELOPMENT_RUN_COMPLETE','run status')
        core.equal(status['arm'],run['arm'],'run status arm')
        trace=core.verify_trace(directory,status,protocol['train'])
        for key in ('best_ce','best_step','steps'): core.equal(run[key],trace[key],'run trace '+key)
        # The completed inventory includes both LRs: audit opaque checkpoint bytes
        # and evaluation identities even for runs that were not selected.
        core.equal(Path(run['checkpoint_path']).resolve(),(directory/'best.pt').resolve(),'checkpoint path')
        checked_hash(run['checkpoint_path'],run['checkpoint_sha256'],'checkpoint '+run['arm'])
        validation=Path(run['validation_output'])
        core.equal(validation.resolve(),(directory/f"validation_{trace['best_step']}").resolve(),'run validation output path')
        core.require(set(run['output_hashes'])=={'candidates.jsonl','proteins.jsonl','decisions.jsonl','report.json'},'incomplete run output hashes')
        for name,value in run['output_hashes'].items(): checked_hash(validation/name,value,'run validation output '+name)
    for arm,selected in learned['selected'].items():
        best=min([r for r in inventory if r['arm']==arm],key=lambda r:(r['best_ce'],r['lr']))
        core.equal(selected,best,'frozen selected run '+arm)
        validation=Path(selected['validation_output'])
        evaluations[arm]=core.verify_evaluation(validation,arm,selected['best_ce'],valid,summaries,roster)
    core.equal(set(evaluations),set(core.ARMS),'selected arm roster')
    identities.update(diagnostic_config_sha256=common.sha_file(config_path),core_review_sha256=common.sha_file(paths['core_report']),
                      dispatch_sha256=common.sha_file(paths['dispatch']),execution_freeze_sha256=common.sha_file(paths['freeze']),
                      frozen_code_hashes=hashes,cache_index_sha256=cache,
                      selected_artifacts={a:{key:selected[key] for key in ('lr','best_step','checkpoint_path','checkpoint_sha256',
                          'directory','artifact_hashes','validation_output','output_hashes')}
                          for a,selected in learned['selected'].items()})
    return dict(config=cfg,root=root,paths=paths,output=output,protocol=protocol,identities=identities,
                learned=learned,valid=valid,roster=roster,summaries=summaries,evaluations=evaluations,snapshot=snapshot)


def resolve_label_path(root,labels_directory,candidate_id,recorded_path):
    """Legacy build_pairs metadata was written with the frozen ROOT/code cwd."""
    recorded=Path(recorded_path)
    path=(recorded if recorded.is_absolute() else Path(root)/'code'/recorded).resolve()
    core.equal(path,(Path(labels_directory)/(candidate_id+'.npz')).resolve(),candidate_id+' label path')
    return path


def load_labels(context):
    labels={}; references={}; ledger=[]
    for row in context['valid']:
        cid=row['candidate_id']; wt=row['protein_id']; summary=context['summaries'][cid]
        path=resolve_label_path(context['root'],context['paths']['labels'],cid,summary['labels_path'])
        scalar.reject_input_overlap(context['output'],[path])
        checksum=checked_hash(path,summary['labels_sha256'],cid+' labels')
        with np.load(path,allow_pickle=False) as archive: lab={k:archive[k] for k in archive.files}
        checked_hash(path,checksum,cid+' labels after read')
        if wt not in references:
            native=Path(row['wt_native_json']); scalar.reject_input_overlap(context['output'],[native])
            native_hash=checked_hash(native,row['wt_native_json_sha256'],wt+' native WT coordinates')
            data=core.read_json(native); core.equal(data['wt_sequence_sha256'],common.sha_text(row['wt_sequence']),wt+' native sequence')
            checked_hash(native,native_hash,wt+' native WT coordinates after read')
            coords=np.full((len(row['wt_sequence']),3),np.nan)
            for index,value in data['coordinates'].items():
                i=int(index); core.require(0<=i<len(coords) and np.asarray(value).shape==(3,),wt+' native coordinate invalid')
                coords[i]=value
            ri,rj=np.triu_indices(len(coords),k=3); distances=np.linalg.norm(coords[ri]-coords[rj],axis=1)
            keep=np.isfinite(distances)&(distances<15.)
            references[wt]=(ri[keep],rj[keep],distances[keep])
            ledger.append(dict(kind='wt_native_json',protein_id=wt,path=str(native),sha256=native_hash))
        else:
            # Every candidate must point to the same sealed WT export.
            native_entry=next(r for r in ledger if r['kind']=='wt_native_json' and r['protein_id']==wt)
            core.equal(str(Path(row['wt_native_json'])),native_entry['path'],wt+' native export path')
            core.equal(row['wt_native_json_sha256'],native_entry['sha256'],wt+' native export hash')
        validate_labels(lab,row,summary,references[wt])
        labels[cid]=lab
        ledger.append(dict(kind='labels',candidate_id=cid,path=str(path),sha256=checksum))
    return labels,references,ledger


def write_artifacts(output,report,artifacts):
    for name,rows in artifacts.items():
        core.atomic_text(output/name,''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows))
    report['artifact_sha256']={name:common.sha_file(output/name) for name in artifacts}
    core.atomic_text(output/'report.json',json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')


def base_report(context):
    return dict(interpretation=INTERPRETATION,seed=core.SEED,output=str(context['output']),
                validation_wt_components=context['roster'],validation_candidates=len(context['valid']),
                identities=context['identities'],information_boundary=context['config']['information_boundary'],
                scientific_pass_decision=None,confirmation_folding_authorized=False,
                GPU_used=False,PLM_forward_calls=0,training_calls=0,structure_calls=0,
                variant_structure_files_read=0,implementation_sha256=common.sha_file(__file__))


def build_diagnostics(config):
    context=load_context(config,'prediction'); labels,references,ledger=load_labels(context)
    candidates=[]; per_wt=[]; summaries={}
    for arm in core.ARMS:
        expected=core.keyed(context['evaluations'][arm]['candidates'],lambda r:r['candidate_id'],'evaluated candidate')
        rows_by_stratum={s:[] for s in ('all','contact_lt8','noncontact_8_to15')}
        for row in context['valid']:
            cid=row['candidate_id']; lab=labels[cid]; ids=np.flatnonzero(lab['retained_mask'])
            path=Path(context['learned']['selected'][arm]['validation_output'])/(cid+'.npz')
            scalar.reject_input_overlap(context['output'],[path])
            checksum=common.sha_file(path)
            with np.load(path,allow_pickle=False) as archive:
                core.equal(set(archive.files),{'pair_ids','probabilities'},cid+' prediction arrays')
                p=archive['probabilities']; validate_probabilities(p,archive['pair_ids'],ids)
            core.equal(checksum,common.sha_file(path),cid+' prediction changed during read')
            ledger.append(dict(kind='prediction',arm=arm,candidate_id=cid,path=str(path),sha256=checksum))
            positions=np.searchsorted(ids,lab['pair_ids']); known=p[positions]; y=lab['target']
            statistics=sufficient_statistics(known,y)
            core.require(statistics is not None,cid+' zero known labels')
            close(statistics['ce'],expected[cid]['ce'],cid+' recomputed CE')
            close(float((np.asarray(p,dtype=float)@VALUES).mean()),expected[cid]['Ghat'],cid+' recomputed retained Ghat')
            distances=references[row['protein_id']][2]
            for stratum in rows_by_stratum:
                eligible=np.ones(len(distances),dtype=bool) if stratum=='all' else distances<8 if stratum=='contact_lt8' else distances>=8
                selected=eligible[lab['pair_ids']]
                coverage={k:context['summaries'][cid][k] for k in ('reference_pairs','retained_pairs','known_pairs','unknown_pairs')}
                coverage.update(eligible_reference_pairs=int(eligible.sum()),eligible_retained_pairs=int(eligible[ids].sum()),
                                eligible_known_pairs=int(selected.sum()),eligible_unknown_pairs=int(eligible[lab['unknown_pair_ids']].sum()),
                                class_counts=np.bincount(y[selected],minlength=5).tolist())
                result={k:row[k] for k in ('candidate_id','protein_id','budget','generator')}
                result.update(arm=arm,stratum=stratum,coverage=coverage,
                              statistics=statistics if stratum=='all' else sufficient_statistics(known[selected],y[selected]))
                rows_by_stratum[stratum].append(result); candidates.append(result)
        summaries[arm]={}
        for stratum,rows in rows_by_stratum.items():
            by_budget=[]
            for budget in sorted({str(r['budget']) for r in rows}):
                for generator in sorted({r['generator'] for r in rows if str(r['budget'])==budget}):
                    part=[r for r in rows if str(r['budget'])==budget and r['generator']==generator]
                    by_budget.append(dict(budget=budget,generator=generator,**aggregate(part,context['roster'])))
            summaries[arm][stratum]=dict(overall=aggregate(rows,context['roster']),by_budget_generator=by_budget,
                by_budget=[dict(budget=b,**aggregate([r for r in rows if str(r['budget'])==b],context['roster']))
                           for b in sorted({str(r['budget']) for r in rows})],
                by_generator=[dict(generator=g,**aggregate([r for r in rows if r['generator']==g],context['roster']))
                              for g in sorted({r['generator'] for r in rows})])
            for wt in context['roster']:
                per_wt.append(dict(arm=arm,stratum=stratum,protein_id=wt,
                                   **aggregate([r for r in rows if r['protein_id']==wt],[wt])))
    report=base_report(context)
    report.update(status=READY,metrics=summaries,prediction_definitions=context['config']['prediction'],
                  normalized_confusion_orientation='rows=true class, columns=predicted argmax class',
                  reliability_sum_interpretation='bin weights, predicted sums and observed sums are relation-normalized candidate masses before equal hierarchy averaging; ECE derives from aggregated masses',
                  coverage_count_interpretation='raw candidate relationship counts; repeated WT reference pairs are counted per candidate; metrics use equal hierarchy weights')
    write_artifacts(context['output'],report,{'candidates.jsonl':candidates,'per_wt.jsonl':per_wt,'artifact_ledger.jsonl':ledger})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',required=True)
    result=build_diagnostics(parser.parse_args().config)
    print(json.dumps({'status':result['status'],'output':result['output']}))


if __name__=='__main__': main()
