import json
import tempfile
import time
import unittest
from pathlib import Path

import sympy as sp

from scientific_intelligent_modelling.benchmarks import runner


def _write_dataset(dataset_dir: Path):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "metadata.yaml").write_text(
        """
dataset:
  description: progress snapshot demo
  target:
    name: target
    description: regression target
  features:
    - name: x0
      description: feature 0
    - name: x1
      description: feature 1
""".strip(),
        encoding="utf-8",
    )
    rows = {
        "train.csv": "x0,x1,target\n1,2,9\n2,1,8\n3,4,19\n",
        "valid.csv": "x0,x1,target\n4,5,24\n",
        "id_test.csv": "x0,x1,target\n5,6,29\n",
        "ood_test.csv": "x0,x1,target\n6,7,34\n",
    }
    for name, content in rows.items():
        (dataset_dir / name).write_text(content, encoding="utf-8")


def _equation_function() -> str:
    return (
        "def equation(x0: float, x1: float, params):\n"
        "    \"\"\"demo\"\"\"\n"
        "    return (\n"
        "        params[0]\n"
        "        + params[1] * x0\n"
        "        + params[2] * x1\n"
        "    )\n"
    )


class BenchmarkProgressSnapshotsTest(unittest.TestCase):
    def test_build_periodic_snapshot_payload_for_llmsr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            exp_dir = root / "exp"
            (exp_dir / "samples").mkdir(parents=True, exist_ok=True)
            _write_dataset(dataset_dir)

            (exp_dir / "samples" / "top01_demo.json").write_text(
                json.dumps(
                    {
                        "iteration": 3,
                        "sample_order": 12,
                        "nmse": 0.01,
                        "mse": 0.01,
                        "function": _equation_function(),
                        "params": [1.0, 2.0, 3.0],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._build_periodic_snapshot_payload(
                tool_name="llmsr",
                dataset=dataset,
                params={"niterations": 100},
                seed=1314,
                started_at=time.time() - 600,
                experiment_dir=exp_dir,
                checkpoint_index=1,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["record_type"], "periodic_best")
            self.assertEqual(payload["tool"], "llmsr")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["checkpoint_index"], 1)
            self.assertEqual(payload["source_iteration"], 3)
            self.assertEqual(payload["source_sample_order"], 12)
            self.assertIsNotNone(payload["canonical_artifact"])
            expr = sp.sympify(payload["canonical_artifact"]["instantiated_expression"])
            self.assertEqual(
                str(sp.expand(expr)),
                str(sp.expand(sp.sympify("2.0*x0 + 3.0*x1 + 1.0"))),
            )
            self.assertAlmostEqual(payload["valid"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["id_test"]["rmse"], 0.0, places=10)
            self.assertAlmostEqual(payload["ood_test"]["rmse"], 0.0, places=10)

    def test_write_progress_payload_writes_outer_and_experiment_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outer"
            exp_dir = root / "exp"
            payload = {
                "record_type": "periodic_best",
                "tool": "drsr",
                "dataset": "demo",
                "status": "ok",
                "checkpoint_index": 2,
                "elapsed_seconds": 600.0,
                "elapsed_minutes": 10,
            }

            written = runner._write_progress_payload(
                payload,
                primary_dir=out_dir / "progress",
                experiment_dir=exp_dir,
            )

            self.assertEqual(len(written), 2)
            for path in written:
                self.assertEqual(path.name, "minute_0010.json")
                self.assertIn("progress", str(path))
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["checkpoint_index"], 2)


if __name__ == "__main__":
    unittest.main()
