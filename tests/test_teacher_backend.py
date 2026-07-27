import importlib.util
import pathlib
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "teacher_scoring.py"


def load_scorer_module():
    spec = importlib.util.spec_from_file_location("teacher_scoring", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ESMCTeacherBackendTests(unittest.TestCase):
    def test_supported_backends_include_esm2_and_esmc(self):
        scorer = load_scorer_module()

        self.assertIn("esm2_transformers", scorer.SUPPORTED_TEACHER_BACKENDS)
        self.assertIn("esmc_official_safetensors", scorer.SUPPORTED_TEACHER_BACKENDS)

    def test_esmc_state_key_mapping_matches_local_safetensors_layout(self):
        scorer = load_scorer_module()

        self.assertEqual(
            scorer.normalize_esmc_state_key("esmc.transformer.blocks.0.attn.layernorm_qkv.layer_norm_weight"),
            "transformer.blocks.0.attn.layernorm_qkv.0.weight",
        )
        self.assertEqual(
            scorer.normalize_esmc_state_key("esmc.transformer.blocks.0.ffn.fc2_weight"),
            "transformer.blocks.0.ffn.3.weight",
        )
        self.assertEqual(
            scorer.normalize_esmc_state_key("lm_head.weight"),
            "sequence_head.weight",
        )
        self.assertIsNone(scorer.normalize_esmc_state_key("esmc.transformer.blocks.0.attn._extra_state"))

    def test_multi_interval_junction_combines_mean_and_min(self):
        scorer = load_scorer_module()

        self.assertAlmostEqual(scorer.combine_junction_values([0.8, 0.6, 0.4]), 0.5)
        self.assertAlmostEqual(scorer.combine_junction_values([0.8]), 0.8)
        self.assertAlmostEqual(scorer.combine_junction_values([]), 0.0)


if __name__ == "__main__":
    unittest.main()
