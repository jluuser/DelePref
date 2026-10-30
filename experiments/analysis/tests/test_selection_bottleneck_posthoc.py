"""Small effective tests for the read-only posthoc reproduction contract."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import selection_bottleneck_posthoc as post


def fixture_pools():
    candidates={}; rows=[]
    for arm in post.sb.ARMS:
        candidates[arm]={}
        for source,offset,spread in (('evocompress',.1,.01),('scisor',-.2,.02)):
            for i in range(7):
                error=offset+(i-3)*spread; cid=source+str(i)
                candidates[arm][cid]=dict(candidate_id=cid,protein_id='w',budget=.2,generator=source,
                    G_lower=.5,G_upper=.5,T_lower=.35,T_upper=.35,Ghat=.5+error,That=.7*(.5+error))
        for (_,budget,source),pool in post.sb.pools(list(candidates[arm].values())).items():
            row=dict(arm=arm,protein_id='w',budget=budget,generator=source,point_status='OBSERVED',pool_size=len(pool),
                     candidate_ids=[r['candidate_id'] for r in sorted(pool,key=lambda r:(r['That'],r['candidate_id']),reverse=True)])
            for ep in ('G','T'):
                x=np.array([r[ep+'hat'] for r in pool]); y=np.array([r[ep+'_lower'] for r in pool])
                row.update({ep+'_predicted_mean':float(x.mean()),ep+'_true_mean':float(y.mean()),ep+'_error_sd':float((x-y).std())})
            rows.append(row)
    return candidates,rows


class PoolTests(unittest.TestCase):
    def test_exact_MSE_identity_balances_source_pools_without_mixed_duplication(self):
        candidates,rows=fixture_pools()
        result=post.pool_decomposition(candidates,rows,{'w':'w'},['0.2'],['evocompress','scisor'])
        for arm in post.sb.ARMS:
            self.assertAlmostEqual(result[arm]['G']['pool_mean_error_MSE'],.025)
            self.assertAlmostEqual(result[arm]['G']['within_pool_centered_error_MSE'],.001)
            self.assertAlmostEqual(result[arm]['G']['total_MSE'],.026)
            self.assertLess(result[arm]['G']['identity_absolute_error'],1e-16)
            self.assertAlmostEqual(result[arm]['T']['total_MSE'],.026*.7**2)

    def test_missing_duplicate_pool_unknown_labels_and_changed_stored_metric_rejected(self):
        for mutation,pattern in [('missing','roster'),('duplicate','duplicate'),('unknown','unknown'),('metric','reproduction differs')]:
            candidates,rows=fixture_pools()
            if mutation=='missing': rows.pop()
            if mutation=='duplicate': rows.append(rows[0])
            if mutation=='unknown': candidates['S']['evocompress0']['G_upper']=.8
            if mutation=='metric': rows[0]['G_error_sd']+=.1
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,pattern):
                post.pool_decomposition(candidates,rows,{'w':'w'},['0.2'],['evocompress','scisor'])

    def test_reproduction_comparison_rejects_changed_fields_or_nonfinite_numbers(self):
        expected={'rows':[{'step':100,'T':.5}]}
        self.assertEqual(post.reproduce_equal(expected,expected,'test'),0.)
        for actual in ({'rows':[]},{'rows':[{'step':100,'T':.51}]},{'rows':[{'step':100,'T':float('nan')}]}):
            with self.assertRaises(ValueError): post.reproduce_equal(actual,expected,'test')


class CurveTests(unittest.TestCase):
    def test_hindsight_keeps_original_CE_step_and_uses_earliest_best_T_tie(self):
        curve=[dict(step=s,ce=ce,mixed_T_three_budget_mean=t) for s,ce,t in ((100,1.1,.8),(200,1.,.5),(300,1.2,.8))]
        result=post.summarize_curve(curve,200)
        self.assertEqual(result['CE_selected_step'],200)
        self.assertEqual(result['hindsight_best_T_step'],100)
        self.assertAlmostEqual(result['hindsight_T_gap'],.3)
        self.assertEqual(curve[1]['step'],200)

    def test_missing_selected_duplicate_step_and_unknown_curve_truth_rejected(self):
        for curve,selected in [([],100),([dict(step=100,mixed_T_three_budget_mean=.5)],200),
                                ([dict(step=100,mixed_T_three_budget_mean=.5)]*2,100),
                                ([dict(step=100,mixed_T_three_budget_mean=None)],100)]:
            with self.assertRaises(ValueError): post.summarize_curve(curve,selected)

    def test_checkpoint_source_hash_and_exact_report_roster(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); directory=root/'checkpoints/C_s20261013_lr0.001'; output=root/post.BASE/post.POSTHOC
            with patch.object(post.common,'AUTHORIZED_ROOT',root):
                post.review.write_json(directory/'run.json',{})
                post.review.write_json(directory/'status.json',dict(status='DEVELOPMENT_RUN_COMPLETE',arm='C',steps=200,best_ce=1.,stopped_by_patience=False))
                post.review.write_rows(directory/'train_trace.jsonl',[dict(step=i,ce=1.) for i in range(1,201)])
                post.review.write_rows(directory/'validation_trace.jsonl',[dict(step=100,ce=1.,improved=True),dict(step=200,ce=1.1,improved=False)])
                run=dict(arm='C',seed=20261013,directory=str(directory),best_step=100,best_ce=1.,steps=200,
                         validation_output=str(directory/'validation_100'),artifact_hashes={n:post.common.sha_file(directory/n) for n in post.review.ARTIFACTS})
                context=dict(root=root,output=output,protocol={'train':{'max_steps':200,'eval_every':100,'patience':5}})
                with self.assertRaisesRegex(ValueError,'report roster'):
                    post.checkpoint_curves(context,{'runs':[run]},{'report_ledger':[]},{})
                (directory/'validation_trace.jsonl').write_text('tampered')
                with self.assertRaisesRegex(ValueError,'curve validation_trace.jsonl hash'):
                    post.checkpoint_curves(context,{'runs':[run]},{'report_ledger':[]},{})


class IsolationTests(unittest.TestCase):
    def test_output_cannot_replace_old_reports_and_reference_sha_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); base=root/post.BASE; output=base/post.POSTHOC
            with self.assertRaisesRegex(ValueError,'dedicated posthoc output'):
                post.build_posthoc(root=root,freeze=root/'freeze',output=base,decomposition_sha256='x',curve_sha256='y')
            base.mkdir(parents=True); (base/'pool_error_decomposition.json').write_text('{}')
            with patch.object(post.review,'read_context',return_value={'root':root,'output':output}),self.assertRaisesRegex(ValueError,'pinned posthoc reference'):
                post.build_posthoc(root=root,freeze=root/'freeze',output=output,decomposition_sha256='0'*64,curve_sha256='1'*64)
            self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
