"""Small CPU fixtures for ranking translation and immutable artifact checks."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import selection_bottleneck as sb


def candidate(cid, true, predicted, coverage=1., wt='w', source='evocompress', ce=1.):
    return dict(candidate_id=cid, protein_id=wt,budget=.2,generator=source,
                Ghat=predicted,That=coverage*predicted,C=coverage,G_lower=true,G_upper=true,
                T_lower=coverage*true,T_upper=coverage*true,ce=ce,valid_pairs=3,known_pairs=3,
                pair_preservation_mae=.1,pair_preservation_brier=.02,multiclass_brier=.3)


class MetricTests(unittest.TestCase):
    def test_high_pooled_correlation_can_hide_reversed_within_pool_order(self):
        rows=[candidate('a',.1,.11,wt='low'),candidate('b',.11,.1,wt='low'),
              candidate('c',.8,.81,wt='high'),candidate('d',.81,.8,wt='high')]
        self.assertGreater(sb.error_metrics(rows,'G')['pooled_pearson'],.99)
        for wt in ('low','high'):
            part=[r for r in rows if r['protein_id']==wt]
            winner=max(part,key=lambda r:(r['That'],r['candidate_id']))['candidate_id']
            self.assertAlmostEqual(sb.pool_metrics(part,winner)['G_spearman'],-1.)

    def test_ce_can_decrease_without_changing_preservation_or_top1(self):
        # Move probability toward the observed middle class while keeping E[G]=.5.
        old=np.array([[.4,.05,.1,.05,.4]])
        new=np.array([[.2,.05,.5,.05,.2]])
        self.assertAlmostEqual(float((old@sb.pred.VALUES)[0]),float((new@sb.pred.VALUES)[0]))
        self.assertLess(-np.log(new[0,2]),-np.log(old[0,2]))
        rows=[candidate('a',.5,.5,ce=float(-np.log(old[0,2]))),candidate('b',.4,.4)]
        changed=[dict(rows[0],ce=float(-np.log(new[0,2]))),rows[1]]
        result=sb.pool_delta(changed,rows)
        self.assertEqual(result['ce_improves_top1_unchanged'],1)
        self.assertTrue(all(r['delta_Ghat']==0 for r in sb.delta_metrics({r['candidate_id']:r for r in changed},{r['candidate_id']:r for r in rows})))

    def test_truth_T_and_G_conflict_and_diagnostic_selectors(self):
        rows=[candidate('a',.9,.9,coverage=.4),candidate('b',.6,.6,coverage=.9)]
        result=sb.pool_metrics(rows,'b')
        self.assertEqual(result['oracle_candidate_ids'],['b'])
        self.assertEqual(result['G_diagnostic_selected'],'a')
        self.assertEqual(result['coverage_diagnostic_selected'],'b')
        self.assertAlmostEqual(result['Gtrue_vs_Ttrue_spearman'],-1.)

    def test_tied_ranks_oracle_ties_and_zero_headroom(self):
        np.testing.assert_array_equal(sb.ranks([2,1,2,4]),[2.5,1,2.5,4])
        self.assertIsNone(sb.correlation([1,1],[2,3],True))
        rows=[candidate('a',.5,.4),candidate('b',.5,.7)]
        result=sb.pool_metrics(rows,'b')
        self.assertEqual(result['oracle_hit'],1)
        self.assertEqual(set(result['oracle_candidate_ids']),{'a','b'})
        self.assertEqual(result['headroom'],0.)
        self.assertIsNone(result['fraction_coverage_headroom_captured'])
        self.assertIsNone(result['T_spearman'])
        self.assertEqual(result['oracle_sorted_true_top2_gap'],0.)

    def test_predicted_top2_realized_gap_differs_from_truth_sorted_top2_gap(self):
        rows=[candidate('a',.2,.9),candidate('b',.6,.8),candidate('c',.9,.1)]
        result=sb.pool_metrics(rows,'a')
        self.assertAlmostEqual(result['realized_T_gap_between_predicted_top2'],-.4)
        self.assertAlmostEqual(result['oracle_sorted_true_top2_gap'],.3)
        rows[1]['G_lower']=rows[1]['G_upper']=rows[1]['T_lower']=rows[1]['T_upper']=.9
        result=sb.pool_metrics(rows,'a')
        self.assertAlmostEqual(result['realized_T_gap_between_predicted_top2'],-.7)
        self.assertEqual(result['oracle_sorted_true_top2_gap'],0.)

    def test_D_lower_ce_can_have_worse_realized_choice(self):
        c=[candidate('a',.9,.8,ce=1.),candidate('b',.2,.3,ce=1.)]
        d=[candidate('a',.9,.3,ce=.8),candidate('b',.2,.8,ce=.8)]
        self.assertLess(sb.hierarchy(d,'ce'),sb.hierarchy(c,'ce'))
        self.assertGreater(sb.pool_metrics(d,'b')['regret'],sb.pool_metrics(c,'a')['regret'])
        self.assertEqual(sb.pool_delta(d,c)['same_top1'],0)

    def test_source_pool_dominance_and_full_WT_weighting(self):
        rows=[candidate('e0',.9,.8,wt='w0'),candidate('e1',.8,.7,wt='w0'),
              candidate('s0',.2,.3,wt='w0',source='scisor')]
        grouped=sb.pools(rows)
        mixed=sb.pool_metrics(grouped[('w0','0.2','mixed')],'e0')
        self.assertEqual(mixed['oracle_generators'],['evocompress'])
        self.assertEqual(mixed['winner_generator'],'evocompress')
        self.assertEqual(len(grouped),3)
        unequal=[dict(r,v=0.) for r in rows]+[dict(candidate('x',.5,.5,wt='w1'),v=1.)]
        self.assertEqual(sb.hierarchy(unequal,'v'),.5)

    def test_unknown_truth_retains_roster_and_disables_point_summary(self):
        rows=[candidate('a',.5,.7),candidate('b',.5,.4)]
        rows[0]['T_upper']=.8; rows[0]['G_upper']=.8
        result=sb.pool_metrics(rows,'a')
        self.assertEqual(result['point_status'],'UNDEFINED_POINT_TRUTH_BOUNDS')
        self.assertIsNone(result['oracle_hit'])
        self.assertIsNone(sb.error_metrics(rows,'T')['mae'])
        pools=[dict(result,protein_id='w',budget='0.2',generator='evocompress'),
               dict(sb.pool_metrics([candidate('c',.5,.5)],'c'),protein_id='w2',budget='0.2',generator='evocompress')]
        summary=sb.descriptive(pools,'oracle_hit')
        self.assertEqual(summary['total'],2)
        self.assertEqual(summary['unknown_truth_pools'],1)
        self.assertIsNone(summary['mean_defined'])


class SealedFixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve(); self.base=self.root/'reports/replication_review'
        self.output=self.base/'selection_bottleneck'
        self.addCleanup(patch.stopall); patch.object(sb.common,'AUTHORIZED_ROOT',self.root).start()
        self.valid=[dict(candidate_id=f'{wt}_{g}_{i}',protein_id=wt,budget=.2,generator=g)
                    for wt in ('w0','w1') for g in ('evocompress','scisor') for i in range(7)]
        self.labels={r['candidate_id']:dict(retained_mask=np.ones(3,dtype=bool),pair_ids=np.arange(3),target=np.array([0,2,4])) for r in self.valid}
        self.summaries={r['candidate_id']:dict(reference_pairs=3,known_pairs=3,C=1.,G_lower=.5,G_upper=.5,T_lower=.5,T_upper=.5,
                         protein_id=r['protein_id'],budget=.2) for r in self.valid}
        self.context=dict(root=self.root,output=self.output,identities={'freeze':'sealed'},valid=self.valid,
                          summaries=self.summaries,roster={'w0':'w0','w1':'w1'},snapshot=self.root/'frozen',protocol={'statistics':{'confidence':.95}})
        self.context['snapshot'].mkdir()
        shutil.copyfile(Path(sb.__file__).resolve().parents[1]/'code/paired_statistics.py',self.context['snapshot']/'paired_statistics.py')
        self.audit=dict(status=sb.review.READY,expected_runs=15,completed_runs=15,seeds=list(sb.review.SEEDS),arms=list(sb.ARMS),
            runs=[dict(arm=a,seed=s,state='COMPLETE',validation_output=str(self.root/f'checkpoints/{a}_{s}/validation_100')) for a in sb.ARMS for s in sb.review.SEEDS],
            validation_candidates=len(self.valid),validation_wt_components=self.context['roster'],identities=self.context['identities'],
            ensemble={'arms':{},'seeds':list(sb.review.SEEDS),'seed_weights':[1/3]*3})
        self.ledger=[]
        for arm in sb.ARMS:
            directory=self.base/'ensemble'/arm; directory.mkdir(parents=True)
            candidates=[]
            for row in self.valid:
                cid=row['candidate_id']; path=directory/(cid+'.npz')
                p=np.tile([.6,.1,.1,.1,.1],(3,1))
                np.savez(path,pair_ids=np.arange(3),probabilities=p)
                candidates.append(sb.review.candidate_metrics(row,self.summaries[cid],self.labels[cid],p))
                hashes=[]
                for seed in sb.review.SEEDS:
                    digest=sb.common.sha_text(f'{arm}:{seed}:{cid}'); hashes.append(digest)
                    self.ledger.append(dict(arm=arm,seed=seed,candidate_id=cid,path=str(self.root/f'checkpoints/{arm}_{seed}/validation_100'/(cid+'.npz')),sha256=digest))
                self.ledger.append(dict(arm=arm,candidate_id=cid,path=str(path),sha256=sb.common.sha_file(path),source_prediction_sha256=hashes))
            decisions=sb.core.expected_decisions(candidates); ce=sb.core.hierarchical(candidates)
            sb.review.write_rows(directory/'candidates.jsonl',candidates)
            sb.review.write_rows(directory/'decisions.jsonl',decisions)
            sb.review.write_rows(directory/'proteins.jsonl',[dict(protein_id=wt,ce=ce) for wt in self.context['roster']])
            sb.review.write_json(directory/'report.json',dict(status='DEVELOPMENT_EVALUATED',arm=arm,candidates=len(candidates),proteins=2,
                                 hierarchical_ce=ce,budget_decisions=sb.core.budget_decisions(decisions)))
            self.audit['ensemble']['arms'][arm]=dict(validation_output=str(directory),hierarchical_ce=ce,lr=.001,checkpoint_sha256='opaque',
                output_hashes={n:sb.common.sha_file(directory/n) for n in sb.review.EVALUATION_FILES})
        self.seal()

    def seal(self):
        sb.review.write_rows(self.base/'artifact_ledger.jsonl',self.ledger)
        self.audit['artifact_ledger_sha256']=sb.common.sha_file(self.base/'artifact_ledger.jsonl')
        sb.review.write_json(self.base/'report.json',self.audit)

    def read(self):
        return sb.sealed_ensemble(self.context,self.labels,[])

    def test_valid_full_ensemble_and_main_cpu_outputs(self):
        evaluations,candidates,_=self.read()
        self.assertEqual(set(candidates),set(sb.ARMS))
        self.assertTrue(all(len(v)==28 for v in candidates.values()))
        before=sb.common.sha_file(self.base/'report.json')
        with patch.object(sb.review,'read_context',return_value=self.context),patch.object(sb.pred,'load_labels',return_value=(self.labels,{},[])):
            report=sb.build_analysis(root=self.root,freeze=self.root/'freeze.json',output=self.output)
        self.assertEqual(report['status'],'CPU_SELECTION_BOTTLENECK_COMPLETE')
        self.assertFalse(report['GPU_used'])
        self.assertEqual(sb.common.sha_file(self.base/'report.json'),before)
        self.assertEqual(len(sb.common.read_jsonl(self.output/'per_wt_pools.jsonl')),30)
        self.assertEqual(set(report['paired_comparisons']),{'C-B','D-C','D-B'})
        result=report['paired_comparisons']['D-C']['T|0.2|mixed']
        self.assertEqual((result['n_wts'],result['replicates'],result['seed']),(2,10000,20261013))
        self.assertIsNone(report['summary'][0]['within_pool']['fraction_coverage_headroom_captured']['mean_defined'])

    def test_ledger_hash_roster_duplicate_path_and_label_tamper_rejected(self):
        original=copy.deepcopy(self.ledger)
        with (self.base/'artifact_ledger.jsonl').open('a') as handle: handle.write('\n')
        with self.assertRaisesRegex(ValueError,'ledger hash'): self.read()
        for mutation,pattern in [([], 'roster'),([original[-1]],'duplicate')]:
            self.ledger=(original[:-1] if not mutation else original+mutation); self.seal()
            with self.subTest(pattern=pattern),self.assertRaisesRegex(ValueError,pattern): self.read()
        self.ledger=copy.deepcopy(original); self.ledger[-1]['path']=str(self.root/'wrong.npz'); self.seal()
        with self.assertRaisesRegex(ValueError,'path'): self.read()
        self.ledger=original+[dict(kind='labels',candidate_id='wrong',path='wrong',sha256='a'*64)]; self.seal()
        with self.assertRaisesRegex(ValueError,'label/native WT ledger'): self.read()

    def test_changed_prediction_hash_missing_probabilities_and_pair_map_rejected(self):
        entry=next(r for r in self.ledger if 'seed' not in r); path=Path(entry['path'])
        original=path.read_bytes(); path.write_bytes(original+b'x')
        with self.assertRaisesRegex(ValueError,'prediction hash'): self.read()
        for arrays,pattern in [(dict(pair_ids=np.arange(3)),'prediction arrays'),
                               (dict(pair_ids=np.arange(2),probabilities=np.tile([.6,.1,.1,.1,.1],(2,1))),'pair IDs')]:
            np.savez(path,**arrays); entry['sha256']=sb.common.sha_file(path); self.seal()
            with self.subTest(pattern=pattern),self.assertRaisesRegex(ValueError,pattern): self.read()

    def test_changed_labels_fail_recomputed_ce_and_output_hash_fail(self):
        self.labels[self.valid[0]['candidate_id']]['target']=np.array([4,4,4])
        with self.assertRaisesRegex(ValueError,'recomputed ce'): self.read()
        self.labels[self.valid[0]['candidate_id']]['target']=np.array([0,2,4])
        path=self.base/'ensemble/S/candidates.jsonl'; path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'output candidates.jsonl hash'): self.read()

    def test_output_protection_and_no_complete_case_paired_inference(self):
        with self.assertRaisesRegex(ValueError,'dedicated bottleneck output'):
            sb.build_analysis(root=self.root,freeze=self.root/'freeze',output=self.base)
        with self.assertRaises(ValueError):
            sb.review.protected_output(self.root,self.output,[self.output/'protected.json'])
        evaluations,_,_=self.read()
        row=evaluations['D']['_evaluation']['decisions'][0]; row['G_lower']=None; row['G_upper']=None
        results=sb.paired_comparisons(evaluations,self.context)
        key='G|'+row['budget']+'|'+row['generator']
        self.assertEqual(results['D-C'][key]['status'],'UNDEFINED_ENDPOINT_BOUNDS')
        self.assertEqual(results['D-C'][key]['expected_wts'],self.context['roster'])
        self.assertFalse(results['D-C'][key]['complete_case_filtering'])


if __name__=='__main__':
    unittest.main()
