import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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


prepare = load_module("medical_autoresearch_prepare_test", "prepare.py")


# ============================================================
# Fake PyHealth fixtures
# ============================================================

class FakeSampleDataset:
    def __init__(self, samples):
        self.samples = list(samples)
        self.patient_to_index = {}
        for idx, sample in enumerate(self.samples):
            self.patient_to_index.setdefault(sample["patient_id"], []).append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def subset(self, indices):
        return FakeSampleDataset([self.samples[idx] for idx in indices])


def build_fake_pyhealth_samples():
    samples = []
    for idx in range(10):
        samples.append(
            {
                "patient_id": f"patient-{idx}",
                "visit_id": f"visit-{idx}",
                "conditions": [[f"cond-{idx}", "shared-cond"], [f"cond-next-{idx}"]],
                "procedures": [[f"proc-{idx}"], [f"proc-next-{idx}"]],
                "drugs_hist": [[f"hist-{idx}"], [f"hist-next-{idx}"]],
                "drugs": [f"drug-{idx}", "shared-drug"],
            }
        )
    return samples


def fake_split_by_patient(dataset, ratios, seed=None):
    patient_ids = list(dataset.patient_to_index.keys())
    n_patients = len(patient_ids)
    n_train = int(n_patients * ratios[0])
    n_val = int(n_patients * ratios[1])
    train_patients = patient_ids[:n_train]
    val_patients = patient_ids[n_train:n_train + n_val]
    test_patients = patient_ids[n_train + n_val:]

    def collect(names):
        indices = []
        for name in names:
            indices.extend(dataset.patient_to_index[name])
        return dataset.subset(indices)

    return collect(train_patients), collect(val_patients), collect(test_patients)


class FakeMIMIC3Dataset:
    def __init__(self, root, tables):
        self.root = root
        self.tables = tables

    def set_task(self, task):
        return FakeSampleDataset(build_fake_pyhealth_samples())


class DrugRecommendationMIMIC3:
    pass


# ============================================================
# TaskRegistry Tests
# ============================================================

class TestTaskRegistry(unittest.TestCase):
    def test_all_tasks_registered(self):
        names = prepare.TaskRegistry.list_tasks()
        self.assertIn("drug_recommendation", names)
        self.assertIn("mortality_prediction", names)
        self.assertIn("readmission_prediction", names)
        self.assertIn("length_of_stay", names)
        self.assertIn("support2_mortality", names)
        self.assertIn("support2_dzclass", names)
        self.assertIn("support2_survival", names)
        self.assertIn("mimic4_mortality", names)
        self.assertIn("mimic4_readmission", names)
        self.assertIn("mimic4_los", names)
        self.assertIn("mimic4_drugrec", names)
        self.assertEqual(len(names), 11)

    def test_get_returns_correct_spec(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        self.assertEqual(spec.task_type, "multilabel")
        self.assertEqual(spec.primary_metric, "jaccard_samples")
        self.assertEqual(spec.label_key, "drugs")

    def test_get_unknown_task_raises(self):
        with self.assertRaises(KeyError):
            prepare.TaskRegistry.get("nonexistent_task")

    def test_select_tasks_all(self):
        specs = prepare.TaskRegistry.select_tasks()
        self.assertEqual(len(specs), 11)

    def test_select_tasks_subset(self):
        specs = prepare.TaskRegistry.select_tasks(["mortality_prediction", "length_of_stay"])
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].name, "mortality_prediction")
        self.assertEqual(specs[1].name, "length_of_stay")

    def test_mortality_spec(self):
        spec = prepare.TaskRegistry.get("mortality_prediction")
        self.assertEqual(spec.task_type, "binary")
        self.assertEqual(spec.primary_metric, "auroc")
        self.assertEqual(spec.label_key, "mortality")

    def test_readmission_spec(self):
        spec = prepare.TaskRegistry.get("readmission_prediction")
        self.assertEqual(spec.task_type, "binary")
        self.assertEqual(spec.primary_metric, "auroc")
        self.assertEqual(spec.label_key, "readmission")

    def test_los_spec(self):
        spec = prepare.TaskRegistry.get("length_of_stay")
        self.assertEqual(spec.task_type, "multiclass")
        self.assertEqual(spec.primary_metric, "f1_macro")
        self.assertEqual(spec.label_key, "los")
        self.assertEqual(spec.label_dim, 3)


# ============================================================
# Reward & Metric Tests
# ============================================================

class TestComputeReward(unittest.TestCase):
    def test_drug_recommendation_reward(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        reward = prepare.compute_reward(
            {"jaccard_samples": 0.6, "ddi_rate": 0.2},
            spec,
        )
        self.assertAlmostEqual(reward, 0.5)

    def test_mortality_reward(self):
        spec = prepare.TaskRegistry.get("mortality_prediction")
        reward = prepare.compute_reward({"auroc": 0.85}, spec)
        self.assertAlmostEqual(reward, 0.85)

    def test_los_reward(self):
        spec = prepare.TaskRegistry.get("length_of_stay")
        reward = prepare.compute_reward({"f1_macro": 0.7}, spec)
        self.assertAlmostEqual(reward, 0.7)

    def test_reward_with_missing_metrics(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        reward = prepare.compute_reward({}, spec)
        self.assertAlmostEqual(reward, 0.0)

    def test_reward_with_partial_metrics(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        reward = prepare.compute_reward({"jaccard_samples": 0.8}, spec)
        self.assertAlmostEqual(reward, 0.8)


class TestComputeMetric(unittest.TestCase):
    def test_pr_auc_is_non_negative(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
        y_prob = np.array([[0.9, 0.2], [0.3, 0.8], [0.7, 0.6], [0.1, 0.1]], dtype=float)
        y_pred = (y_prob >= 0.5).astype(float)
        score = prepare._compute_metric("pr_auc_samples", y_true, y_prob, y_pred, spec)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_jaccard_samples(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        y_true = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
        y_pred = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
        y_prob = y_pred
        score = prepare._compute_metric("jaccard_samples", y_true, y_prob, y_pred, spec)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_auroc_binary(self):
        spec = prepare.TaskRegistry.get("mortality_prediction")
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        y_pred = (y_prob >= 0.5).astype(float)
        score = prepare._compute_metric("auroc", y_true, y_prob, y_pred, spec)
        self.assertGreater(score, 0.5)

    def test_auroc_single_class_returns_half(self):
        spec = prepare.TaskRegistry.get("mortality_prediction")
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])
        y_pred = np.array([0, 0, 0, 0], dtype=float)
        score = prepare._compute_metric("auroc", y_true, y_prob, y_pred, spec)
        self.assertAlmostEqual(score, 0.5)

    def test_f1_macro(self):
        spec = prepare.TaskRegistry.get("length_of_stay")
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.eye(3)[y_true]
        y_pred = y_true
        score = prepare._compute_metric("f1_macro", y_true, y_prob, y_pred, spec)
        self.assertAlmostEqual(score, 1.0)

    def test_accuracy(self):
        spec = prepare.TaskRegistry.get("length_of_stay")
        y_true = np.array([0, 1, 2, 0])
        y_prob = np.eye(3)[y_true]
        y_pred = np.array([0, 1, 2, 1])
        score = prepare._compute_metric("accuracy", y_true, y_prob, y_pred, spec)
        self.assertAlmostEqual(score, 0.75)

    def test_auprc_non_negative(self):
        spec = prepare.TaskRegistry.get("mortality_prediction")
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.7, 0.8])
        y_pred = (y_prob >= 0.5).astype(float)
        score = prepare._compute_metric("auprc", y_true, y_prob, y_pred, spec)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_unknown_metric_returns_zero(self):
        spec = prepare.TaskRegistry.get("mortality_prediction")
        score = prepare._compute_metric("nonexistent", np.array([0, 1]), np.array([0.1, 0.9]), np.array([0, 1]), spec)
        self.assertEqual(score, 0.0)


# ============================================================
# DDI Rate Tests
# ============================================================

class TestDDIRate(unittest.TestCase):
    def test_no_interactions(self):
        ddi_matrix = np.zeros((5, 5), dtype=np.float32)
        y_pred = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0]], dtype=float)
        rate = prepare.compute_ddi_rate(y_pred, ddi_matrix)
        self.assertAlmostEqual(rate, 0.0)

    def test_all_interactions(self):
        ddi_matrix = np.ones((3, 3), dtype=np.float32)
        np.fill_diagonal(ddi_matrix, 0)
        y_pred = np.array([[1, 1, 1]], dtype=float)
        rate = prepare.compute_ddi_rate(y_pred, ddi_matrix)
        self.assertAlmostEqual(rate, 1.0)

    def test_none_matrix(self):
        rate = prepare.compute_ddi_rate(np.array([[1, 0]]), None)
        self.assertAlmostEqual(rate, 0.0)

    def test_1d_input(self):
        rate = prepare.compute_ddi_rate(np.array([1, 0, 1]), np.ones((3, 3)))
        self.assertAlmostEqual(rate, 0.0)

    def test_get_ddi_matrix_returns_array(self):
        matrix = prepare.get_ddi_matrix("drug_recommendation")
        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.shape[0], matrix.shape[1])

    def test_get_ddi_matrix_non_multilabel_returns_none(self):
        matrix = prepare.get_ddi_matrix("mortality_prediction")
        self.assertIsNone(matrix)


# ============================================================
# Synthetic EHR Data Tests
# ============================================================

class TestSyntheticEHRDataset(unittest.TestCase):
    def test_drug_recommendation_data(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        ds = prepare.SyntheticEHRDataset(spec, n_patients=50, seed=42)
        self.assertGreater(len(ds), 0)
        sample = ds[0]
        self.assertIn("conditions", sample)
        self.assertIn("procedures", sample)
        self.assertIn("drugs_hist", sample)
        self.assertIn("drugs", sample)
        self.assertEqual(len(sample["drugs"]), spec.label_dim)

    def test_mortality_data(self):
        spec = prepare.TaskRegistry.get("mortality_prediction")
        ds = prepare.SyntheticEHRDataset(spec, n_patients=50, seed=42)
        self.assertGreater(len(ds), 0)
        sample = ds[0]
        self.assertIn("conditions", sample)
        self.assertIn("procedures", sample)
        self.assertIn("mortality", sample)
        self.assertIn(sample["mortality"], [0, 1])

    def test_readmission_data(self):
        spec = prepare.TaskRegistry.get("readmission_prediction")
        ds = prepare.SyntheticEHRDataset(spec, n_patients=50, seed=42)
        self.assertGreater(len(ds), 0)
        sample = ds[0]
        self.assertIn("readmission", sample)
        self.assertIn(sample["readmission"], [0, 1])

    def test_los_data(self):
        spec = prepare.TaskRegistry.get("length_of_stay")
        ds = prepare.SyntheticEHRDataset(spec, n_patients=50, seed=42)
        self.assertGreater(len(ds), 0)
        sample = ds[0]
        self.assertIn("los", sample)
        self.assertIn(sample["los"], [0, 1, 2])

    def test_ddi_matrix_created_for_multilabel(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        ds = prepare.SyntheticEHRDataset(spec, n_patients=20, seed=42)
        self.assertIsNotNone(ds.ddi_matrix)
        self.assertEqual(ds.ddi_matrix.shape, (spec.label_dim, spec.label_dim))
        self.assertTrue(np.allclose(ds.ddi_matrix, ds.ddi_matrix.T))

    def test_no_ddi_matrix_for_binary(self):
        spec = prepare.TaskRegistry.get("mortality_prediction")
        ds = prepare.SyntheticEHRDataset(spec, n_patients=20, seed=42)
        self.assertIsNone(ds.ddi_matrix)

    def test_reproducibility(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        ds1 = prepare.SyntheticEHRDataset(spec, n_patients=50, seed=123)
        ds2 = prepare.SyntheticEHRDataset(spec, n_patients=50, seed=123)
        self.assertEqual(len(ds1), len(ds2))
        s1 = ds1[0]
        s2 = ds2[0]
        self.assertEqual(s1["conditions"], s2["conditions"])


# ============================================================
# Collate & Padding Tests
# ============================================================

class TestCollateAndPadding(unittest.TestCase):
    def test_pad_nested_sequence(self):
        seqs = [[[1, 2, 3], [4, 5]], [[6]]]
        result = prepare._pad_nested_sequence(seqs, vocab_size=10)
        self.assertEqual(result.shape, (2, 2, 3))
        self.assertEqual(result[0, 0, 0].item(), 1)
        self.assertEqual(result[1, 0, 0].item(), 6)
        self.assertEqual(result[1, 1, 0].item(), 0)  # padding

    def test_pad_clamps_to_vocab(self):
        seqs = [[[999]]]
        result = prepare._pad_nested_sequence(seqs, vocab_size=10)
        self.assertEqual(result[0, 0, 0].item(), 9)  # clamped to vocab-1

    def test_build_visit_mask(self):
        seqs = [[[1, 2], [3]], [[4]]]
        mask = prepare._build_visit_mask(seqs, max_visits=3)
        self.assertEqual(mask.shape, (2, 3))
        self.assertTrue(mask[0, 0].item())
        self.assertTrue(mask[0, 1].item())
        self.assertFalse(mask[0, 2].item())
        self.assertTrue(mask[1, 0].item())
        self.assertFalse(mask[1, 1].item())

    def test_collate_fn_drug_recommendation(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        collate = prepare.collate_fn_factory(spec)
        samples = [
            {
                "conditions": [[1, 2], [3]],
                "procedures": [[10]],
                "drugs_hist": [[5, 6]],
                "drugs": np.zeros(spec.label_dim, dtype=np.float32),
            },
            {
                "conditions": [[4]],
                "procedures": [[11, 12]],
                "drugs_hist": [[7]],
                "drugs": np.ones(spec.label_dim, dtype=np.float32),
            },
        ]
        batch = collate(samples)
        self.assertEqual(batch["conditions"].ndim, 3)
        self.assertEqual(batch["procedures"].ndim, 3)
        self.assertEqual(batch["drugs_hist"].ndim, 3)
        self.assertEqual(batch["drugs"].shape, (2, spec.label_dim))
        self.assertEqual(batch["mask"].shape[0], 2)

    def test_collate_fn_binary_task(self):
        spec = prepare.TaskRegistry.get("mortality_prediction")
        collate = prepare.collate_fn_factory(spec)
        samples = [
            {"conditions": [[1, 2]], "procedures": [[10]], "mortality": 0},
            {"conditions": [[3]], "procedures": [[11]], "mortality": 1},
        ]
        batch = collate(samples)
        self.assertEqual(batch["mortality"].dtype, torch.long)
        self.assertEqual(batch["mortality"].tolist(), [0, 1])


# ============================================================
# Data Loading Tests
# ============================================================

class TestLoadTaskData(unittest.TestCase):
    def test_load_synthetic_drug_recommendation(self):
        spec, train_dl, val_dl, test_dl = prepare.load_task_data(
            "drug_recommendation", batch_size=4, n_synthetic_patients=50,
            return_spec=True,
        )
        self.assertGreater(len(train_dl.dataset), 0)
        self.assertGreater(len(test_dl.dataset), 0)
        batch = next(iter(train_dl))
        self.assertIn("conditions", batch)
        self.assertIn("drugs", batch)

    def test_load_synthetic_mortality(self):
        spec, train_dl, val_dl, test_dl = prepare.load_task_data(
            "mortality_prediction", batch_size=4, n_synthetic_patients=50,
            return_spec=True,
        )
        batch = next(iter(train_dl))
        self.assertIn("mortality", batch)
        self.assertEqual(batch["mortality"].dtype, torch.long)

    def test_load_synthetic_los(self):
        spec, train_dl, val_dl, test_dl = prepare.load_task_data(
            "length_of_stay", batch_size=4, n_synthetic_patients=50,
            return_spec=True,
        )
        batch = next(iter(train_dl))
        self.assertIn("los", batch)
        self.assertEqual(batch["los"].dtype, torch.long)

    def test_load_without_return_spec(self):
        result = prepare.load_task_data(
            "drug_recommendation", batch_size=4, n_synthetic_patients=50,
        )
        self.assertEqual(len(result), 3)

    def test_pyhealth_normalization(self):
        pyhealth_module = types.ModuleType("pyhealth")
        datasets_module = types.ModuleType("pyhealth.datasets")
        splitter_module = types.ModuleType("pyhealth.datasets.splitter")
        tasks_module = types.ModuleType("pyhealth.tasks")
        drug_module = types.ModuleType("pyhealth.tasks.drug_recommendation")

        datasets_module.MIMIC3Dataset = FakeMIMIC3Dataset
        splitter_module.split_by_patient = fake_split_by_patient
        drug_module.DrugRecommendationMIMIC3 = DrugRecommendationMIMIC3

        pyhealth_module.datasets = datasets_module
        pyhealth_module.tasks = tasks_module
        datasets_module.splitter = splitter_module

        injected_modules = {
            "pyhealth": pyhealth_module,
            "pyhealth.datasets": datasets_module,
            "pyhealth.datasets.splitter": splitter_module,
            "pyhealth.tasks": tasks_module,
            "pyhealth.tasks.drug_recommendation": drug_module,
        }

        with patch.dict(sys.modules, injected_modules, clear=False):
            spec, train_loader, val_loader, test_loader = prepare.load_task_data(
                "drug_recommendation",
                batch_size=2,
                use_pyhealth=True,
                data_root="/tmp/fake-mimic",
                return_spec=True,
            )

        self.assertGreater(len(train_loader.dataset), 0)
        self.assertGreater(len(test_loader.dataset), 0)
        self.assertGreaterEqual(len(val_loader.dataset), 0)

        batch = next(iter(train_loader))
        self.assertEqual(batch["conditions"].ndim, 3)
        self.assertEqual(batch["procedures"].ndim, 3)
        self.assertEqual(batch["drugs_hist"].ndim, 3)
        self.assertEqual(batch["drugs"].shape[-1], spec.label_dim)
        self.assertGreater(spec.feature_dims["conditions"], 1)
        self.assertLess(spec.feature_dims["conditions"], 500)
        self.assertEqual(spec.label_dim, 11)


# ============================================================
# MultiTaskLoader Tests
# ============================================================

class TestMultiTaskLoader(unittest.TestCase):
    def setUp(self):
        self.specs = prepare.TaskRegistry.select_tasks(
            ["drug_recommendation", "mortality_prediction"],
        )
        loaders = {}
        for spec in self.specs:
            train_dl, _, _ = prepare.load_task_data(
                spec.name, batch_size=4, n_synthetic_patients=30,
            )
            loaders[spec.name] = train_dl
        self.loaders = loaders

    def test_round_robin(self):
        loader = prepare.MultiTaskLoader(self.loaders, self.specs, strategy="round_robin")
        names = []
        for _ in range(4):
            name, batch, spec = loader.sample()
            names.append(name)
            self.assertIn("conditions", batch)
        self.assertEqual(names[0], names[2])
        self.assertEqual(names[1], names[3])
        self.assertNotEqual(names[0], names[1])

    def test_bandit(self):
        loader = prepare.MultiTaskLoader(self.loaders, self.specs, strategy="bandit")
        for _ in range(4):
            name, batch, spec = loader.sample()
            self.assertIn(name, ["drug_recommendation", "mortality_prediction"])
            loader.update_reward(name, 0.5)

    def test_proportional(self):
        loader = prepare.MultiTaskLoader(self.loaders, self.specs, strategy="proportional")
        for _ in range(4):
            name, batch, spec = loader.sample()
            self.assertIn(name, ["drug_recommendation", "mortality_prediction"])

    def test_invalid_strategy(self):
        loader = prepare.MultiTaskLoader(self.loaders, self.specs, strategy="invalid")
        with self.assertRaises(ValueError):
            loader.sample()

    def test_update_reward(self):
        loader = prepare.MultiTaskLoader(self.loaders, self.specs, strategy="bandit")
        loader.update_reward("drug_recommendation", 1.0)
        self.assertEqual(loader._counts["drug_recommendation"], 1)
        self.assertAlmostEqual(loader._rewards["drug_recommendation"], 1.0)


# ============================================================
# Helper Function Tests
# ============================================================

class TestHelperFunctions(unittest.TestCase):
    def test_ensure_list_from_numpy(self):
        result = prepare._ensure_list(np.array([1, 2, 3]))
        self.assertEqual(result, [1, 2, 3])

    def test_ensure_list_from_tensor(self):
        result = prepare._ensure_list(torch.tensor([1, 2]))
        self.assertEqual(result, [1, 2])

    def test_ensure_list_from_none(self):
        self.assertEqual(prepare._ensure_list(None), [])

    def test_ensure_list_from_scalar(self):
        self.assertEqual(prepare._ensure_list(42), [42])

    def test_ensure_nested_sequence_flat(self):
        result = prepare._ensure_nested_sequence([1, 2, 3])
        self.assertEqual(result, [[1, 2, 3]])

    def test_ensure_nested_sequence_nested(self):
        result = prepare._ensure_nested_sequence([[1, 2], [3]])
        self.assertEqual(result, [[1, 2], [3]])

    def test_ensure_nested_sequence_empty(self):
        result = prepare._ensure_nested_sequence([])
        self.assertEqual(result, [[]])

    def test_coerce_scalar_tensor(self):
        self.assertEqual(prepare._coerce_scalar(torch.tensor(5)), 5)

    def test_coerce_scalar_numpy(self):
        self.assertEqual(prepare._coerce_scalar(np.int64(3)), 3)

    def test_coerce_scalar_int(self):
        self.assertEqual(prepare._coerce_scalar(7), 7)

    def test_set_seed_deterministic(self):
        prepare.set_seed(99)
        a = torch.randn(5)
        prepare.set_seed(99)
        b = torch.randn(5)
        self.assertTrue(torch.allclose(a, b))

    def test_count_parameters(self):
        model = torch.nn.Linear(10, 5)
        self.assertEqual(prepare.count_parameters(model), 55)  # 10*5 + 5 bias

    def test_get_peak_vram_returns_float(self):
        vram = prepare.get_peak_vram_mb()
        self.assertIsInstance(vram, float)
        self.assertGreaterEqual(vram, 0.0)


# ============================================================
# Print Results Tests
# ============================================================

class TestPrintResults(unittest.TestCase):
    def test_print_results_format(self):
        import io
        import contextlib

        spec = prepare.TaskRegistry.get("drug_recommendation")
        metrics = {"jaccard_samples": 0.5, "f1_samples": 0.6, "pr_auc_samples": 0.4}

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            prepare.print_results(
                metrics=metrics,
                task_spec=spec,
                training_seconds=10.0,
                total_seconds=12.0,
                peak_vram_mb=100.0,
                num_params=1000,
            )
        output = stdout.getvalue()
        self.assertIn("primary_metric (jaccard_samples): 0.500000", output)
        self.assertIn("reward:", output)
        self.assertIn("task:             drug_recommendation", output)
        self.assertIn("num_params:       1000", output)
        self.assertIn("---", output)


# ============================================================
# InMemoryEHRDataset Tests
# ============================================================

class TestInMemoryEHRDataset(unittest.TestCase):
    def test_len_and_getitem(self):
        samples = [{"a": 1}, {"a": 2}, {"a": 3}]
        ds = prepare.InMemoryEHRDataset(samples)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], {"a": 1})
        self.assertEqual(ds[2], {"a": 3})

    def test_empty(self):
        ds = prepare.InMemoryEHRDataset([])
        self.assertEqual(len(ds), 0)


# ============================================================
# _resolve_label_key Tests
# ============================================================

class TestResolveLabelKey(unittest.TestCase):
    def test_exact_match(self):
        spec = prepare.TaskRegistry.get("drug_recommendation")
        sample = {"conditions": [], "procedures": [], "drugs_hist": [], "drugs": []}
        self.assertEqual(prepare._resolve_label_key(spec, sample), "drugs")

    def test_los_alias(self):
        spec = prepare.TaskSpec(
            name="length_of_stay",
            task_type="multiclass",
            description="test",
            feature_keys=["conditions", "procedures"],
            label_key="los_category",
            feature_dims={"conditions": 20, "procedures": 20},
            label_dim=3,
            primary_metric="f1_macro",
            metric_direction="max",
            metrics=["f1_macro"],
            reward_components={},
        )
        sample = {"conditions": [], "procedures": [], "los": 1}
        self.assertEqual(prepare._resolve_label_key(spec, sample), "los")

    def test_single_candidate_fallback(self):
        spec = prepare.TaskSpec(
            name="test_task",
            task_type="binary",
            description="test",
            feature_keys=["conditions"],
            label_key="nonexistent",
            feature_dims={"conditions": 20},
            label_dim=2,
            primary_metric="auroc",
            metric_direction="max",
            metrics=["auroc"],
            reward_components={},
        )
        sample = {"conditions": [], "my_label": 0}
        self.assertEqual(prepare._resolve_label_key(spec, sample), "my_label")

    def test_ambiguous_raises(self):
        spec = prepare.TaskSpec(
            name="test_task",
            task_type="binary",
            description="test",
            feature_keys=["conditions"],
            label_key="nonexistent",
            feature_dims={"conditions": 20},
            label_dim=2,
            primary_metric="auroc",
            metric_direction="max",
            metrics=["auroc"],
            reward_components={},
        )
        sample = {"conditions": [], "label_a": 0, "label_b": 1}
        with self.assertRaises(KeyError):
            prepare._resolve_label_key(spec, sample)


# ============================================================
# _build_token_mapping Tests
# ============================================================

class TestBuildTokenMapping(unittest.TestCase):
    def test_basic(self):
        mapping = prepare._build_token_mapping(["a", "c", "b"])
        self.assertEqual(len(mapping), 3)
        # Tokens sorted alphabetically, indices start at 1
        self.assertEqual(mapping["a"], 1)
        self.assertEqual(mapping["b"], 2)
        self.assertEqual(mapping["c"], 3)

    def test_deduplication(self):
        mapping = prepare._build_token_mapping(["x", "x", "y", "y"])
        self.assertEqual(len(mapping), 2)

    def test_none_excluded(self):
        mapping = prepare._build_token_mapping([None, "a", None, "b"])
        self.assertEqual(len(mapping), 2)
        self.assertNotIn(None, mapping)

    def test_empty(self):
        mapping = prepare._build_token_mapping([])
        self.assertEqual(mapping, {})

    def test_numeric_values(self):
        mapping = prepare._build_token_mapping([3, 1, 2, 1])
        self.assertEqual(len(mapping), 3)
        # Sorted by str representation
        self.assertIn(1, mapping)
        self.assertIn(2, mapping)
        self.assertIn(3, mapping)


# ============================================================
# _iter_codes Tests
# ============================================================

class TestIterCodes(unittest.TestCase):
    def test_basic(self):
        result = list(prepare._iter_codes([[1, 2], [3]]))
        self.assertEqual(result, [1, 2, 3])

    def test_empty(self):
        result = list(prepare._iter_codes([[]]))
        self.assertEqual(result, [])

    def test_nested_empty(self):
        result = list(prepare._iter_codes([]))
        self.assertEqual(result, [])


# ============================================================
# Readmission Task Data Tests
# ============================================================

class TestReadmissionData(unittest.TestCase):
    def test_load_readmission(self):
        spec, train_dl, val_dl, test_dl = prepare.load_task_data(
            "readmission_prediction", batch_size=4, n_synthetic_patients=50,
            return_spec=True,
        )
        batch = next(iter(train_dl))
        self.assertIn("conditions", batch)
        self.assertIn("readmission", batch)
        self.assertEqual(batch["readmission"].dtype, torch.long)
        self.assertEqual(spec.task_type, "binary")

    def test_readmission_has_drugs_feature(self):
        spec = prepare.TaskRegistry.get("readmission_prediction")
        self.assertIn("drugs", spec.feature_keys)


# ============================================================
# compute_reward Edge Cases
# ============================================================

class TestComputeRewardEdgeCases(unittest.TestCase):
    def test_reward_with_aliased_components(self):
        """Test that old alias 'jaccard' maps to 'jaccard_samples'."""
        spec = prepare.TaskSpec(
            name="test_task",
            task_type="multilabel",
            description="test",
            feature_keys=["conditions"],
            label_key="drugs",
            feature_dims={"conditions": 20},
            label_dim=10,
            primary_metric="jaccard_samples",
            metric_direction="max",
            metrics=["jaccard_samples"],
            reward_components={"jaccard": 1.0},
        )
        reward = prepare.compute_reward({"jaccard_samples": 0.7}, spec)
        self.assertAlmostEqual(reward, 0.7)

    def test_reward_empty_components(self):
        spec = prepare.TaskSpec(
            name="test_task",
            task_type="binary",
            description="test",
            feature_keys=["conditions"],
            label_key="label",
            feature_dims={"conditions": 20},
            label_dim=2,
            primary_metric="auroc",
            metric_direction="max",
            metrics=["auroc"],
            reward_components={},
        )
        reward = prepare.compute_reward({"auroc": 0.9}, spec)
        self.assertAlmostEqual(reward, 0.0)


# ============================================================
# SUPPORT2 Dataset Tests
# ============================================================

# SUPPORT2 tests download from HuggingFace on first run (~2s).
# They are skipped if the `datasets` package is not installed.

def _has_datasets_lib():
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_datasets_lib(), "requires `datasets` package")
class TestSupport2Tasks(unittest.TestCase):
    def test_three_tasks_registered(self):
        names = prepare.TaskRegistry.list_tasks()
        self.assertIn("support2_mortality", names)
        self.assertIn("support2_dzclass", names)
        self.assertIn("support2_survival", names)

    def test_mortality_spec(self):
        spec = prepare.TaskRegistry.get("support2_mortality")
        self.assertEqual(spec.task_type, "binary")
        self.assertEqual(spec.label_key, "hospdead")
        self.assertEqual(spec.primary_metric, "auroc")

    def test_dzclass_spec(self):
        spec = prepare.TaskRegistry.get("support2_dzclass")
        self.assertEqual(spec.task_type, "multiclass")
        self.assertEqual(spec.label_key, "dzclass")
        self.assertEqual(spec.label_dim, 4)

    def test_survival_spec(self):
        spec = prepare.TaskRegistry.get("support2_survival")
        self.assertEqual(spec.task_type, "binary")
        self.assertEqual(spec.label_key, "survival_2m")


@unittest.skipUnless(_has_datasets_lib(), "requires `datasets` package")
class TestSupport2DataLoading(unittest.TestCase):
    """Test SUPPORT2 data loading — downloads data on first run."""

    @classmethod
    def setUpClass(cls):
        """Load once, share across tests."""
        cls.spec, cls.train_dl, cls.val_dl, cls.test_dl = prepare.load_task_data(
            "support2_mortality", batch_size=8, return_spec=True,
        )

    def test_split_sizes(self):
        total = (len(self.train_dl.dataset) + len(self.val_dl.dataset)
                 + len(self.test_dl.dataset))
        self.assertEqual(total, 9105)
        self.assertGreater(len(self.train_dl.dataset), 7000)

    def test_vocab_size_updated(self):
        self.assertGreater(self.spec.feature_dims["conditions"], 100)
        self.assertEqual(
            self.spec.feature_dims["conditions"],
            self.spec.feature_dims["procedures"],
        )

    def test_batch_shapes(self):
        batch = next(iter(self.train_dl))
        self.assertEqual(batch["conditions"].ndim, 3)
        self.assertEqual(batch["procedures"].ndim, 3)
        self.assertEqual(batch["mask"].ndim, 2)
        self.assertIn(batch["hospdead"].dtype, [torch.long, torch.int64])

    def test_labels_binary(self):
        batch = next(iter(self.train_dl))
        labels = batch["hospdead"]
        self.assertTrue((labels >= 0).all())
        self.assertTrue((labels <= 1).all())

    def test_dzclass_loading(self):
        spec = prepare.TaskRegistry.get("support2_dzclass")
        # Reuse already-downloaded data by building dataset directly
        ds = prepare.Support2Dataset(spec, seed=42)
        self.assertGreater(len(ds), 9000)
        sample = ds[0]
        self.assertIn("dzclass", sample)
        self.assertIn(sample["dzclass"], [0, 1, 2, 3])

    def test_survival_loading(self):
        spec = prepare.TaskRegistry.get("support2_survival")
        ds = prepare.Support2Dataset(spec, seed=42)
        sample = ds[0]
        self.assertIn("survival_2m", sample)
        self.assertIn(sample["survival_2m"], [0, 1])


def _has_mimic4_data():
    """Check if MIMIC-IV data is available locally."""
    import os
    root = "/home/sw2572/project_pi_yz875/sw2572/data/physionet.org/files/mimiciv/3.1"
    return os.path.isdir(os.path.join(root, "hosp"))


class TestMIMIC4Tasks(unittest.TestCase):
    """Test MIMIC-IV task registration (no data needed)."""

    def test_four_tasks_registered(self):
        names = prepare.TaskRegistry.list_tasks()
        self.assertIn("mimic4_mortality", names)
        self.assertIn("mimic4_readmission", names)
        self.assertIn("mimic4_los", names)
        self.assertIn("mimic4_drugrec", names)

    def test_mortality_spec(self):
        spec = prepare.TaskRegistry.get("mimic4_mortality")
        self.assertEqual(spec.task_type, "binary")
        self.assertEqual(spec.label_key, "mortality")
        self.assertEqual(spec.primary_metric, "auroc")
        self.assertEqual(spec.label_dim, 2)

    def test_readmission_spec(self):
        spec = prepare.TaskRegistry.get("mimic4_readmission")
        self.assertEqual(spec.task_type, "binary")
        self.assertEqual(spec.label_key, "readmission")

    def test_los_spec(self):
        spec = prepare.TaskRegistry.get("mimic4_los")
        self.assertEqual(spec.task_type, "multiclass")
        self.assertEqual(spec.label_key, "los_bucket")
        self.assertEqual(spec.label_dim, 4)

    def test_drugrec_spec(self):
        spec = prepare.TaskRegistry.get("mimic4_drugrec")
        self.assertEqual(spec.task_type, "multilabel")
        self.assertEqual(spec.label_key, "drugs")
        self.assertIn("drugs_hist", spec.feature_keys)


@unittest.skipUnless(_has_mimic4_data(), "requires local MIMIC-IV data")
class TestMIMIC4DataLoading(unittest.TestCase):
    """Test MIMIC-IV data loading with tiny dev subset (200 patients)."""

    @classmethod
    def setUpClass(cls):
        # Use max_patients=200 directly for minimal memory footprint
        spec = prepare.TaskRegistry.get("mimic4_mortality")
        ds = prepare.MIMIC4Dataset(spec, seed=42, max_patients=200)
        cls.dataset = ds
        resolved = prepare.replace(
            spec,
            feature_dims={"conditions": ds.diag_vocab_size, "procedures": ds.proc_vocab_size},
        )
        cls.spec = resolved
        # Manual split
        n = len(ds)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        n_test = n - n_train - n_val
        import torch
        gen = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            ds, [n_train, n_val, n_test], generator=gen,
        )
        collate = prepare.collate_fn_factory(resolved)
        cls.train_dl = torch.utils.data.DataLoader(train_ds, batch_size=8, collate_fn=collate)
        cls.val_dl = torch.utils.data.DataLoader(val_ds, batch_size=8, collate_fn=collate)
        cls.test_dl = torch.utils.data.DataLoader(test_ds, batch_size=8, collate_fn=collate)

    def test_split_sizes(self):
        total = (len(self.train_dl.dataset) + len(self.val_dl.dataset)
                 + len(self.test_dl.dataset))
        self.assertGreater(total, 100)

    def test_vocab_size_updated(self):
        self.assertGreater(self.spec.feature_dims["conditions"], 10)
        self.assertGreater(self.spec.feature_dims["procedures"], 10)

    def test_batch_shapes(self):
        batch = next(iter(self.train_dl))
        self.assertEqual(batch["conditions"].ndim, 3)
        self.assertEqual(batch["procedures"].ndim, 3)
        self.assertEqual(batch["mask"].ndim, 2)
        self.assertIn(batch["mortality"].dtype, [torch.long, torch.int64])

    def test_labels_binary(self):
        batch = next(iter(self.train_dl))
        labels = batch["mortality"]
        self.assertTrue((labels >= 0).all())
        self.assertTrue((labels <= 1).all())

    def test_readmission_loading(self):
        spec = prepare.TaskRegistry.get("mimic4_readmission")
        ds = prepare.MIMIC4Dataset(spec, seed=42, max_patients=200)
        sample = ds[0]
        self.assertIn("readmission", sample)
        self.assertIn(sample["readmission"], [0, 1])

    def test_los_loading(self):
        spec = prepare.TaskRegistry.get("mimic4_los")
        ds = prepare.MIMIC4Dataset(spec, seed=42, max_patients=200)
        sample = ds[0]
        self.assertIn("los_bucket", sample)
        self.assertIn(sample["los_bucket"], [0, 1, 2, 3])

    def test_drugrec_loading(self):
        spec = prepare.TaskRegistry.get("mimic4_drugrec")
        ds = prepare.MIMIC4Dataset(spec, seed=42, max_patients=200)
        self.assertGreater(len(ds), 0)
        sample = ds[0]
        self.assertIn("drugs", sample)
        self.assertEqual(sample["drugs"].ndim, 1)
        self.assertIn("drugs_hist", sample)


if __name__ == "__main__":
    unittest.main()
