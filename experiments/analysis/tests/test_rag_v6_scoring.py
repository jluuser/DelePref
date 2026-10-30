"""Small semantic fixtures; no real outcome files or protein search is used."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rag_v6_features as f
import rag_v6_scoring as s


def matrix_profile(patterns):
    blocks = [(5,8),(18,21),(10,13),(24,27)]
    result = []
    for pattern in patterns:
        row = list("A"*30)
        for delete,(a,b) in zip(pattern,blocks):
            if delete: row[a:b] = ["-"]*(b-a)
        result.append({"projected":"".join(row)})
    return {"representatives":result}


class SemanticTests(unittest.TestCase):
    def test_joint_information_beyond_identical_marginals_and_gap_burden(self):
        mask = np.zeros(30,dtype=int); mask[5:8]=1; mask[18:21]=1
        coupled=matrix_profile([[1,1,0,0]]*4+[[0,0,1,1]]*4)
        separate=matrix_profile([[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1]]*2)
        a=f.plan_features("A"*30,mask,coupled); b=f.plan_features("A"*30,mask,separate)
        self.assertTrue(a["evidence"] and b["evidence"])
        self.assertTrue(a["joint_valid"] and b["joint_valid"])
        np.testing.assert_allclose(a["values"][:-1],b["values"][:-1])
        self.assertEqual(a["values"][-1],.5); self.assertEqual(b["values"][-1],0.)

    def test_terminal_is_unknown_and_zero_observation_not_negative_label(self):
        mask=np.r_[np.ones(6,dtype=int),np.zeros(24,dtype=int)]
        a=f.plan_features("A"*30,mask,matrix_profile([[0,0,0,0]]*8))
        self.assertFalse(a["evidence"]); self.assertEqual(a["fallback_reason"],"terminal_no_double_anchor")

    def test_standardization_masks_q1_joint_after_centering(self):
        x=np.array([[2.,0.]])
        z=s.standardize(x,np.array([[True,False]]),np.array([1.,.5]),np.array([1.,.2]))
        self.assertEqual(z[0,1],0.)

    def test_unknown_alignment_and_internal_gap_are_distinct(self):
        fields="q t 0.4 6 2 7 8 1 4 4 1e-9 30 .75 1 CDEFGH CD--GH".split()
        h=f.project_hit(fields,"ACDEFGHI")
        self.assertEqual(h["projected"],".CD--GH.")
        fields[-1]="--EF--"; fields[8]="2"
        self.assertEqual(f.project_hit(fields,"ACDEFGHI")["projected"],"...EF...")

    def test_mismatched_query_rejected(self):
        fields="q t 0.4 4 1 4 4 1 4 4 1e-9 30 1 1 ACDE ACDE".split()
        with self.assertRaises(ValueError): f.project_hit(fields,"CDEF")

    def test_whitelist_never_includes_structure_or_teacher_metadata(self):
        row=dict(candidate_id="c",protein_id="p",budget=.2,wt_sequence="ACDEF",sequence="ADEF",mask=[0,1,0,0,0],
                 reference_path="must_not_open",direct_contact_loss=999,legacy_metrics={"T":999})
        out=f.pure_candidate(row)
        self.assertNotIn("reference_path",out); self.assertNotIn("legacy_metrics",out)

    def test_mixed_pool_fallback_and_full_supported_pool(self):
        rows=[dict(protein_id=p,budget=.2) for p in ["a","a","b","b"]]
        e=np.array([1,0,1,1],dtype=bool)
        np.testing.assert_array_equal(s.final_gate(e,rows,False),[0,0,1,1])
        np.testing.assert_array_equal(s.final_gate(e,rows,True),e)

    def test_all_pairs_and_equal_hierarchy_weights(self):
        rows=[dict(protein_id=p,budget=b) for p,b,n in [("a",.2,2),("a",.3,3),("b",.2,4)] for _ in range(n)]
        z=np.arange(len(rows)*2,dtype=float).reshape(-1,2)
        d,pairs,w=s.pair_design(z,np.zeros(len(rows)),rows)
        self.assertEqual(len(pairs),1+3+6); self.assertAlmostEqual(float(w.sum()),1.)
        self.assertTrue(np.all(d==0)); self.assertAlmostEqual(float(s.balanced_weights(rows).sum()),1.)

    def test_fallback_scores_equal_original_U_exactly(self):
        u=np.array([.75123,.805]); z=np.ones((2,len(f.NAMES)))
        models={"arm":{"theta":np.ones(len(f.NAMES)).tolist()}}
        np.testing.assert_array_equal(s.predict(models,z,np.zeros(2),u,.1)["arm"],u)

    def test_zero_joint_signal_cannot_create_nested_model_gain(self):
        rng=np.random.default_rng(3); z=rng.normal(size=(12,len(f.NAMES))); z[:,-1]=0.
        rows=[dict(protein_id=f"p{i//3}",budget=.2) for i in range(12)]
        models=s.fit_arms(z,np.ones(12),np.linspace(.3,.8,12),np.linspace(.1,.6,12),rows,.2,.2)
        scores=s.predict(models,z,np.ones(12),np.linspace(.3,.8,12),.2)
        np.testing.assert_array_equal(scores["U_S_M"],scores["U_S_M_J"])

    def test_mixed_calibration_needs_several_WTs_and_pools(self):
        rows=[dict(protein_id=f"p{p}",budget=b) for p in range(6) for b in [.2,.3] for _ in range(2)]
        e=np.tile([True,False],12); z=np.zeros((24,len(f.NAMES))); z[:,0]=1.
        a=s.calibration_audit(z,e,rows)
        self.assertTrue(a["allow_mixed_residual"])
        a=s.calibration_audit(z,np.ones(24,dtype=bool),rows)
        self.assertFalse(a["allow_mixed_residual"])

    def test_clustering_does_not_treat_discordant_gaps_as_matching_AA(self):
        self.assertLess(f.aligned_identity("AAAA----AAAA","AAAAAAAAAAAA"),.9)


if __name__=="__main__":
    unittest.main()
