import unittest
import numpy as np
import torch
from model import PairResponseHead
from features import pair_features


class Controls(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = PairResponseHead().eval()
        self.w = torch.randn(10, 1152, requires_grad=True)
        self.m = torch.randn(7, 1152, requires_grad=True)
        self.kept = torch.tensor([0, 1, 2, 6, 7, 8, 9])
        self.i, self.j = torch.tensor([0, 2, 3]), torch.tensor([4, 5, 6])
        self.g = torch.randn(3, 60); self.h = torch.randn(3, 60)

    def run_arm(self, arm, w=None, m=None):
        return self.model(self.w if w is None else w, self.m if m is None else m, self.kept, self.i, self.j, self.g, self.h, arm)

    def test_parameters_and_symmetry(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 133701)
        for arm in "SABCD":
            a = self.run_arm(arm)
            b = self.model(self.w, self.m, self.kept, self.j, self.i, self.h, self.g, arm)
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_disabled_inputs_and_gradients(self):
        for arm, disabled in [("S", "wm"), ("A", "m"), ("D", "w")]:
            original = self.run_arm(arm)
            for field in disabled:
                perturbed = self.run_arm(arm, w=self.w + 17 if field == "w" else None, m=self.m * 3 if field == "m" else None)
                torch.testing.assert_close(original, perturbed, rtol=0, atol=0)
            gradients = torch.autograd.grad(original.sum(), (self.w, self.m), allow_unused=True)
            for field, gradient in zip("wm", gradients):
                if field in disabled:
                    self.assertTrue(gradient is None or torch.count_nonzero(gradient) == 0)
        for arm in "BC":
            gradient = torch.autograd.grad(self.run_arm(arm).sum(), self.w)[0]
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_internal_deletion_features(self):
        seq = "ACDEFGHIKL"; xyz = np.column_stack((np.arange(10), np.zeros((10, 2))))
        g, h = pair_features(seq, xyz, self.kept.numpy(), self.i.numpy(), self.j.numpy())
        self.assertEqual(g.shape, (3, 60))
        self.assertAlmostEqual(g[0, 4], 7/9, places=6)
        self.assertAlmostEqual(g[0, 6], 4/6, places=6)
        np.testing.assert_array_equal(g[:, 20:40], h[:, 40:60])
        np.testing.assert_array_equal(g[:, 3], h[:, 4])
        identity, _ = pair_features(seq, xyz, np.arange(10), np.array([0]), np.array([4]))
        np.testing.assert_array_equal(identity[:, 7:11], np.ones((1, 4)))
        np.testing.assert_array_equal(identity[:, 15:18], np.zeros((1, 3)))

    def test_only_allowlisted_inputs(self):
        with self.assertRaises(TypeError):
            self.model(self.w, self.m, self.kept, self.i, self.j, self.g, self.h, labels=torch.zeros(3))

    def test_control_slots(self):
        captured = []
        hook = self.model.head[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        self.run_arm("B"); b = captured[0]; captured.clear()
        self.run_arm("C"); c = captured[0]
        hook.remove()
        torch.testing.assert_close(b[:, :256], c[:, :256], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(b[:, 256:384]), 0)
        self.assertGreater(torch.count_nonzero(c[:, 256:384]), 0)
        projected = self.model.project(self.w)[self.kept]
        torch.testing.assert_close(b[:, :64], projected[self.i])

    def test_response_permutation_preserves_pool_changes_local(self):
        class First64(torch.nn.Module):
            def forward(self, values):
                return values[:, :64]
        self.model.project = First64()
        w_kept = self.w.detach()[self.kept, :64]
        response = torch.arange(7 * 64, dtype=self.w.dtype).reshape(7, 64) / 64
        permutation = torch.tensor([6, 5, 4, 3, 2, 1, 0])
        edited = self.m.detach().clone()
        shuffled = edited.clone()
        edited[:, :64] = w_kept + response
        shuffled[:, :64] = w_kept + response[permutation]
        captured = []
        hook = self.model.head[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        b_original = self.run_arm("B", m=edited)
        b_slots = captured[0]; captured.clear()
        b_shuffled = self.run_arm("B", m=shuffled)
        b_shuffled_slots = captured[0]; captured.clear()
        torch.testing.assert_close(b_original, b_shuffled, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(b_slots, b_shuffled_slots, rtol=1e-5, atol=1e-6)
        self.run_arm("C", m=edited)
        c_slots = captured[0]; captured.clear()
        self.run_arm("C", m=shuffled)
        c_shuffled_slots = captured[0]
        hook.remove()
        torch.testing.assert_close(c_slots[:, :256], c_shuffled_slots[:, :256], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(c_slots[:, 384:], c_shuffled_slots[:, 384:], rtol=0, atol=0)
        torch.testing.assert_close(c_shuffled_slots[:, 256:320], response[permutation][self.i])
        torch.testing.assert_close(c_shuffled_slots[:, 320:384], response[permutation][self.j])
        self.assertGreater((c_slots[:, 256:384] - c_shuffled_slots[:, 256:384]).abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
