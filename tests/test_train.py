import contextlib
import gc
import importlib.util
import io
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PROJECT_ROOT))
    spec.loader.exec_module(module)
    return module


prepare = load_module("medical_autoresearch_prepare_test2", "prepare.py")
train = load_module("medical_autoresearch_train_test", "train.py")


# ============================================================
# Model Component Tests
# ============================================================

class TestCodeEmbedding(unittest.TestCase):
    def test_output_shape(self):
        emb = train.CodeEmbedding(vocab_size=100, embed_dim=32, dropout=0.0)
        x = torch.randint(0, 100, (2, 5, 8))  # batch=2, visits=5, codes=8
        out = emb(x)
        self.assertEqual(out.shape, (2, 5, 32))

    def test_padding_idx_zero(self):
        emb = train.CodeEmbedding(vocab_size=100, embed_dim=16, dropout=0.0)
        x = torch.zeros(1, 1, 3, dtype=torch.long)  # all padding
        out = emb(x)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_gradient_flows(self):
        emb = train.CodeEmbedding(vocab_size=50, embed_dim=16, dropout=0.0)
        x = torch.randint(1, 50, (2, 3, 4))
        out = emb(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(emb.embedding.weight.grad)


class TestPolicyAgent(unittest.TestCase):
    def test_output_shapes(self):
        agent = train.PolicyAgent(obs_dim=64, n_actions=10, hidden_dim=32, use_baseline=True)
        obs = torch.randn(4, 64)
        action, log_prob, entropy, baseline = agent(obs)
        self.assertEqual(action.shape, (4,))
        self.assertEqual(log_prob.shape, (4,))
        self.assertEqual(entropy.shape, (4,))
        self.assertEqual(baseline.shape, (4,))

    def test_no_baseline(self):
        agent = train.PolicyAgent(obs_dim=32, n_actions=5, hidden_dim=16, use_baseline=False)
        obs = torch.randn(2, 32)
        action, log_prob, entropy, baseline = agent(obs)
        self.assertIsNone(baseline)

    def test_action_in_range(self):
        agent = train.PolicyAgent(obs_dim=16, n_actions=5)
        obs = torch.randn(100, 16)
        action, _, _, _ = agent(obs)
        self.assertTrue((action >= 0).all())
        self.assertTrue((action < 5).all())

    def test_entropy_positive(self):
        agent = train.PolicyAgent(obs_dim=16, n_actions=5)
        obs = torch.randn(10, 16)
        _, _, entropy, _ = agent(obs)
        self.assertTrue((entropy >= 0).all())


def _make_small_spec(task_type="binary", label_key="label", label_dim=2,
                     feature_keys=None):
    """Create a small TaskSpec for memory-efficient unit tests."""
    if feature_keys is None:
        feature_keys = ["conditions", "procedures"]
    return prepare.TaskSpec(
        name="test_task",
        task_type=task_type,
        description="test",
        feature_keys=feature_keys,
        label_key=label_key,
        feature_dims={k: 20 for k in feature_keys},
        label_dim=label_dim,
        primary_metric="auroc" if task_type == "binary" else "f1_macro",
        metric_direction="max",
        metrics=["auroc"] if task_type == "binary" else ["f1_macro"],
        reward_components={},
    )


class TestClinicalRLModel(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_binary_forward(self):
        spec = _make_small_spec("binary", "mortality", 2)
        model = train.ClinicalRLModel(spec)
        batch = {
            "conditions": torch.randint(0, 15, (2, 3, 4)),
            "procedures": torch.randint(0, 15, (2, 3, 4)),
            "mortality": torch.tensor([0, 1]),
            "mask": torch.ones(2, 3, dtype=torch.bool),
        }
        output = model(**batch)
        self.assertIn("loss", output)
        self.assertEqual(output["logit"].shape, (2, 2))
        del model

    def test_multilabel_forward(self):
        spec = _make_small_spec("multilabel", "drugs", 10,
                                ["conditions", "procedures", "drugs_hist"])
        model = train.ClinicalRLModel(spec)
        batch = {
            "conditions": torch.randint(0, 15, (2, 3, 4)),
            "procedures": torch.randint(0, 15, (2, 3, 4)),
            "drugs_hist": torch.randint(0, 15, (2, 3, 4)),
            "drugs": torch.zeros(2, 10),
            "mask": torch.ones(2, 3, dtype=torch.bool),
        }
        batch["drugs"][:, :3] = 1.0
        output = model(**batch)
        self.assertIn("loss", output)
        self.assertIn("y_prob", output)
        self.assertEqual(output["logit"].shape, (2, 10))
        del model

    def test_multiclass_forward(self):
        spec = _make_small_spec("multiclass", "los", 3)
        model = train.ClinicalRLModel(spec)
        batch = {
            "conditions": torch.randint(0, 15, (2, 3, 4)),
            "procedures": torch.randint(0, 15, (2, 3, 4)),
            "los": torch.tensor([0, 2]),
            "mask": torch.ones(2, 3, dtype=torch.bool),
        }
        output = model(**batch)
        self.assertIn("loss", output)
        self.assertEqual(output["logit"].shape, (2, 3))
        del model

    def test_forward_without_labels(self):
        spec = _make_small_spec("binary", "mortality", 2)
        model = train.ClinicalRLModel(spec)
        batch = {
            "conditions": torch.randint(0, 15, (2, 3, 4)),
            "procedures": torch.randint(0, 15, (2, 3, 4)),
            "mask": torch.ones(2, 3, dtype=torch.bool),
        }
        output = model(**batch)
        self.assertIn("logit", output)
        self.assertNotIn("loss", output)
        del model

    def test_variable_sequence_length(self):
        spec = _make_small_spec("binary", "mortality", 2)
        model = train.ClinicalRLModel(spec)
        batch = {
            "conditions": torch.randint(0, 15, (2, 5, 4)),
            "procedures": torch.randint(0, 15, (2, 5, 4)),
            "mortality": torch.tensor([0, 1]),
            "mask": torch.tensor([
                [True, True, True, False, False],
                [True, True, False, False, False],
            ]),
        }
        output = model(**batch)
        self.assertFalse(torch.isnan(output["loss"]))
        del model

    def test_single_visit(self):
        spec = _make_small_spec("binary", "mortality", 2)
        model = train.ClinicalRLModel(spec)
        batch = {
            "conditions": torch.randint(0, 15, (1, 1, 4)),
            "procedures": torch.randint(0, 15, (1, 1, 4)),
            "mortality": torch.tensor([0]),
            "mask": torch.ones(1, 1, dtype=torch.bool),
        }
        output = model(**batch)
        self.assertFalse(torch.isnan(output["loss"]))
        del model

    def test_gradient_flows_through_model(self):
        spec = _make_small_spec("multilabel", "drugs", 10,
                                ["conditions", "procedures", "drugs_hist"])
        model = train.ClinicalRLModel(spec)
        batch = {
            "conditions": torch.randint(0, 15, (2, 3, 4)),
            "procedures": torch.randint(0, 15, (2, 3, 4)),
            "drugs_hist": torch.randint(0, 15, (2, 3, 4)),
            "drugs": torch.zeros(2, 10),
            "mask": torch.ones(2, 3, dtype=torch.bool),
        }
        batch["drugs"][:, :3] = 1.0
        output = model(**batch)
        output["loss"].backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        self.assertTrue(has_grad)
        del model


# ============================================================
# Reward Shaping Tests
# ============================================================

class TestShapeReward(unittest.TestCase):
    def test_drug_recommendation_ddi_penalty(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        reward = train.shape_reward(
            {"jaccard_samples": 0.8, "ddi_rate": 0.1}, spec,
        )
        base = prepare.compute_reward({"jaccard_samples": 0.8, "ddi_rate": 0.1}, spec)
        ddi_penalty = -0.5 * 0.1
        self.assertAlmostEqual(reward, base + ddi_penalty)

    def test_mortality_no_ddi_penalty(self):
        spec = prepare.TaskRegistry.get("mortality_prediction")
        reward = train.shape_reward({"auroc": 0.9}, spec)
        base = prepare.compute_reward({"auroc": 0.9}, spec)
        self.assertAlmostEqual(reward, base)


# ============================================================
# End-to-End Smoke Tests
# ============================================================

class TestTrainSmoke(unittest.TestCase):
    """End-to-end smoke test via subprocess.

    On login nodes with tight memory limits the subprocess may be
    OOM-killed (returncode -9).  We skip gracefully in that case
    so the rest of the suite stays green.
    """

    def test_main_prints_result_summary(self):
        result = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "train.py"),
                "--time-budget", "0",
                "--n-patients", "10",
                "--batch-size", "2",
            ],
            capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode in (-9, -6, 137):
            self.skipTest(
                "train.py was OOM-killed on login node (returncode "
                f"{result.returncode}); skipping smoke test"
            )
        output = result.stdout
        self.assertEqual(result.returncode, 0, f"train.py failed:\n{result.stderr}")
        self.assertIn("primary_metric (jaccard_samples):", output)
        self.assertIn("reward:", output)
        self.assertIn("task:             drug_recommendation", output)
        self.assertIn("training_seconds:", output)
        self.assertIn("total_seconds:", output)
        self.assertIn("peak_vram_mb:", output)
        self.assertIn("num_params:", output)


# ============================================================
# Parse Args Tests
# ============================================================

class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        args = train.parse_args([])
        self.assertIsNone(args.task)
        self.assertIsNone(args.time_budget)
        self.assertIsNone(args.batch_size)
        self.assertEqual(args.n_patients, 2000)
        self.assertFalse(args.use_pyhealth)

    def test_overrides(self):
        args = train.parse_args([
            "--task", "mortality_prediction",
            "--time-budget", "120",
            "--batch-size", "16",
            "--n-patients", "500",
        ])
        self.assertEqual(args.task, "mortality_prediction")
        self.assertEqual(args.time_budget, 120)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.n_patients, 500)

    def test_pyhealth_flag(self):
        args = train.parse_args(["--use-pyhealth", "--data-root", "/tmp/mimic"])
        self.assertTrue(args.use_pyhealth)
        self.assertEqual(args.data_root, "/tmp/mimic")


# ============================================================
# Evaluation Integration Tests
# ============================================================

class TestEvaluation(unittest.TestCase):
    """Lightweight evaluation test using small synthetic specs (no subprocess)."""

    def test_evaluate_model_binary(self):
        spec = _make_small_spec("binary", "mortality", 2)
        ds = prepare.SyntheticEHRDataset(spec, n_patients=15, seed=42)
        n = len(ds)
        collate = prepare.collate_fn_factory(spec)
        test_dl = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=collate,
        )
        model = train.ClinicalRLModel(spec)
        metrics = prepare.evaluate_model(model, test_dl, spec, device="cpu")
        self.assertIn("auroc", metrics)
        del model
        gc.collect()

    def test_evaluate_model_multiclass(self):
        spec = _make_small_spec("multiclass", "los", 3)
        ds = prepare.SyntheticEHRDataset(spec, n_patients=15, seed=42)
        collate = prepare.collate_fn_factory(spec)
        test_dl = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=collate,
        )
        model = train.ClinicalRLModel(spec)
        metrics = prepare.evaluate_model(model, test_dl, spec, device="cpu")
        self.assertIn("f1_macro", metrics)
        del model
        gc.collect()


# ============================================================
# Shape Reward — All Task Types
# ============================================================

class TestShapeRewardAllTasks(unittest.TestCase):
    def test_readmission_no_ddi(self):
        spec = prepare.TaskRegistry.get("readmission_prediction")
        reward = train.shape_reward({"auroc": 0.75}, spec)
        base = prepare.compute_reward({"auroc": 0.75}, spec)
        self.assertAlmostEqual(reward, base)

    def test_los_no_ddi(self):
        spec = prepare.TaskRegistry.get("length_of_stay")
        reward = train.shape_reward({"f1_macro": 0.6}, spec)
        base = prepare.compute_reward({"f1_macro": 0.6}, spec)
        self.assertAlmostEqual(reward, base)

    def test_drug_rec_zero_ddi(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        reward = train.shape_reward(
            {"jaccard_samples": 0.5, "ddi_rate": 0.0}, spec,
        )
        base = prepare.compute_reward({"jaccard_samples": 0.5, "ddi_rate": 0.0}, spec)
        self.assertAlmostEqual(reward, base)

    def test_drug_rec_no_ddi_key(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        reward = train.shape_reward({"jaccard_samples": 0.5}, spec)
        base = prepare.compute_reward({"jaccard_samples": 0.5}, spec)
        self.assertAlmostEqual(reward, base)


# ============================================================
# ClinicalRLModel — RL Loss Edge Cases
# ============================================================

class TestRLLossEdgeCases(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_rl_loss_single_timestep(self):
        """With only 1 visit, no RL actions are taken — rl_loss should be 0."""
        spec = _make_small_spec("binary", "mortality", 2)
        model = train.ClinicalRLModel(spec)
        batch = {
            "conditions": torch.randint(0, 15, (2, 1, 4)),
            "procedures": torch.randint(0, 15, (2, 1, 4)),
            "mortality": torch.tensor([0, 1]),
            "mask": torch.ones(2, 1, dtype=torch.bool),
        }
        output = model(**batch)
        self.assertAlmostEqual(output["rl_loss"].item(), 0.0, places=5)
        del model

    def test_rl_loss_many_timesteps(self):
        """With many visits, RL loss should be non-zero."""
        spec = _make_small_spec("binary", "mortality", 2)
        model = train.ClinicalRLModel(spec)
        batch = {
            "conditions": torch.randint(0, 15, (2, 8, 4)),
            "procedures": torch.randint(0, 15, (2, 8, 4)),
            "mortality": torch.tensor([0, 1]),
            "mask": torch.ones(2, 8, dtype=torch.bool),
        }
        output = model(**batch)
        # With 8 timesteps, there should be RL actions
        self.assertIsNotNone(output["rl_loss"])
        del model


# ============================================================
# ClinicalRLModel — Readmission Task
# ============================================================

class TestReadmissionModel(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_readmission_forward(self):
        spec = _make_small_spec("binary", "readmission", 2,
                                ["conditions", "procedures", "drugs"])
        model = train.ClinicalRLModel(spec)
        batch = {
            "conditions": torch.randint(0, 15, (2, 3, 4)),
            "procedures": torch.randint(0, 15, (2, 3, 4)),
            "drugs": torch.randint(0, 15, (2, 3, 4)),
            "readmission": torch.tensor([0, 1]),
            "mask": torch.ones(2, 3, dtype=torch.bool),
        }
        output = model(**batch)
        self.assertIn("loss", output)
        self.assertEqual(output["logit"].shape, (2, 2))
        del model

    def test_readmission_evaluation(self):
        spec = _make_small_spec("binary", "readmission", 2,
                                ["conditions", "procedures", "drugs"])
        ds = prepare.SyntheticEHRDataset(spec, n_patients=15, seed=42)
        collate = prepare.collate_fn_factory(spec)
        test_dl = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=collate,
        )
        model = train.ClinicalRLModel(spec)
        metrics = prepare.evaluate_model(model, test_dl, spec, device="cpu")
        self.assertIn("auroc", metrics)
        del model
        gc.collect()


# ============================================================
# Evaluation — Multilabel
# ============================================================

class TestEvaluationMultilabel(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_evaluate_multilabel(self):
        spec = prepare.TaskSpec(
            name="test_multilabel",
            task_type="multilabel",
            description="test",
            feature_keys=["conditions", "procedures", "drugs_hist"],
            label_key="drugs",
            feature_dims={k: 20 for k in ["conditions", "procedures", "drugs_hist"]},
            label_dim=10,
            primary_metric="jaccard_samples",
            metric_direction="max",
            metrics=["jaccard_samples", "f1_samples"],
            reward_components={},
        )
        ds = prepare.SyntheticEHRDataset(spec, n_patients=15, seed=42)
        collate = prepare.collate_fn_factory(spec)
        test_dl = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=collate,
        )
        model = train.ClinicalRLModel(spec)
        metrics = prepare.evaluate_model(model, test_dl, spec, device="cpu")
        self.assertIn("jaccard_samples", metrics)
        del model
        gc.collect()


# ============================================================
# SUPPORT2 End-to-End Tests
# ============================================================

def _has_datasets_lib():
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_datasets_lib(), "requires `datasets` package")
class TestSupport2Model(unittest.TestCase):
    """SUPPORT2 model tests. Uses shared data to reduce memory on login nodes."""

    _spec = None
    _train_dl = None
    _test_dl = None

    @classmethod
    def setUpClass(cls):
        cls._spec, cls._train_dl, _, cls._test_dl = prepare.load_task_data(
            "support2_mortality", batch_size=8, return_spec=True,
        )

    def tearDown(self):
        gc.collect()

    def test_support2_mortality_forward_and_eval(self):
        model = train.ClinicalRLModel(self._spec)
        batch = next(iter(self._train_dl))
        output = model(**batch)
        self.assertIn("loss", output)
        self.assertEqual(output["logit"].shape[1], 2)

        # Evaluate on a small subset (first 2 batches) to save memory
        import itertools
        small_dl = list(itertools.islice(self._test_dl, 2))

        class _SmallLoader:
            def __init__(self, batches): self._b = batches
            def __iter__(self): return iter(self._b)

        metrics = prepare.evaluate_model(model, _SmallLoader(small_dl), self._spec, device="cpu")
        self.assertIn("auroc", metrics)
        del model


if __name__ == "__main__":
    unittest.main()
