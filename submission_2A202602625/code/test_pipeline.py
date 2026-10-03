"""Kiểm thử CPU: metric, clipping, resume, divergence, optimizer và workbook."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import f1_score
from model import MLP, count_params, EXPECTED_PARAMS
from optimizer import clip_gradients
from train import run_experiment, evaluate, set_seed
from results_table import write_xlsx, to_row, load_results

ROOT = Path(__file__).resolve().parents[2]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        set_seed(17)
        cls.data = dict(X_tr=torch.randn(128, 54), y_tr=torch.arange(128) % 7,
                        X_val=torch.randn(73, 54), y_val=torch.arange(73) % 7,
                        signature="synthetic-test-v1")
        cls.cfg = dict(exp_id="test", optimizer="adam", lr=0.01, epochs=3, batch=32,
                       train_eval_size=128, eval_batch=23, verbose=False)

    def test_architecture_and_weight_initialization(self):
        for hidden, count in EXPECTED_PARAMS.items():
            m = MLP(hidden=hidden)
            self.assertEqual(count_params(m), count)
            self.assertEqual(m(torch.randn(8, 54)).shape, (8, 7))
        m = MLP(init="he")
        for l in m.modules():
            if isinstance(l, torch.nn.Linear):
                self.assertTrue(torch.all(l.bias == 0))
                self.assertAlmostEqual(l.weight.var().item() / (2 / l.in_features), 1.0, delta=.15)

    def test_weighted_metrics_ce_and_mse(self):
        m = MLP()
        x, y = self.data["X_val"], self.data["y_val"]
        with torch.no_grad():
            logits = m(x)
        for loss_name in ("ce", "mse"):
            r = evaluate(m, x, y, loss_name, batch_size=23)
            expected = F.cross_entropy(logits, y) if loss_name == "ce" else F.mse_loss(logits, F.one_hot(y, 7).float())
            self.assertAlmostEqual(r["loss"], expected.item(), places=6)
            f1 = f1_score(y, logits.argmax(1), average="macro", labels=range(7), zero_division=0)
            self.assertAlmostEqual(r["macro_f1"], f1, places=12)

    def test_clipping_pre_norm(self):
        p = torch.nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor([3., 4.])
        self.assertEqual(clip_gradients([p], 1), 5)
        self.assertAlmostEqual(p.grad.norm().item(), 1, places=5)

    def test_resume_and_completed_cache(self):
        reference = run_experiment(self.cfg, self.data)
        self.assertLess(reference["history"]["train_loss"][-1], reference["history"]["train_loss"][0])
        with tempfile.TemporaryDirectory() as folder:
            cfg = {**self.cfg, "out_dir": folder}
            with patch("results_table.save_result", side_effect=InterruptedError("simulate disconnect")):
                with self.assertRaises(InterruptedError):
                    run_experiment(cfg, self.data)
            resumed = run_experiment(cfg, self.data)
            self.assertEqual(reference["history"]["val_loss"], resumed["history"]["val_loss"])
            for name in reference["best_state"]:
                self.assertTrue(torch.equal(reference["best_state"][name], resumed["best_state"][name]))
            cached = run_experiment(cfg, self.data)
            self.assertEqual(cached["history"]["epoch_time_s"], resumed["history"]["epoch_time_s"])
            with self.assertRaises(ValueError):
                run_experiment({**cfg, "lr": .02}, self.data)

    def test_adam_adamw_zero_decay(self):
        a = run_experiment({**self.cfg, "optimizer": "adam"}, self.data)
        b = run_experiment({**self.cfg, "optimizer": "adamw"}, self.data)
        for key in a["best_state"]:
            self.assertTrue(torch.equal(a["best_state"][key], b["best_state"][key]))

    def test_divergence_and_unsupported_amp(self):
        r = run_experiment({**self.cfg, "optimizer": "sgd", "lr": 1e30}, self.data)
        self.assertTrue(r["summary"]["diverged"])
        for precision in ("fp16", "bf16"):
            r = run_experiment({**self.cfg, "precision": precision}, self.data)
            self.assertEqual(r["summary"]["status"], "skipped")

    def test_workbook_extension_and_health_schema(self):
        import openpyxl
        r = run_experiment(self.cfg, self.data)
        rows = [{**to_row(r), "exp_id": f"row{i}", "group": "hparam"} for i in range(65)]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "table.xlsx"
            write_xlsx(rows, str(ROOT / "templates/experiment_table_template.xlsx"), str(out))
            w = openpyxl.load_workbook(out)
            self.assertEqual(w.sheetnames, ["Legend", "Experiments", "Seeds", "Summary"])
            self.assertEqual(w["Experiments"]["A66"].value, "row64")
            self.assertIn("P66", w["Experiments"]["AD66"].value)
            self.assertIn("$66", w["Seeds"]["B2"].value)
            Path(folder, "health.json").write_text('{"loss_step0": 2.3}')
            self.assertEqual(load_results(folder), [])

    def test_official_eval_csv_and_cache(self):
        from reporting import official_evaluation
        with np.load(ROOT / "data/processed/eval.npz") as raw:
            evaluation = dict(X_eval=torch.from_numpy(raw["X"]), eval_row_id=raw["row_id"])
        # Model test ngắn; điểm ở temp chỉ kiểm tra script/CSV, không dùng chọn cấu hình.
        r = run_experiment({**self.cfg, "epochs": 1}, self.data)
        with tempfile.TemporaryDirectory() as folder:
            scores = official_evaluation(r, evaluation, ROOT, folder)
            self.assertEqual(scores["n_eval"], 116203)
            self.assertEqual(len(scores["per_class"]), 7)
            cached = official_evaluation(r, evaluation, ROOT, folder)
            self.assertEqual(scores, cached)


if __name__ == "__main__":
    unittest.main()
