import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scientific_intelligent_modelling" / "algorithms" / "drsr_wrapper" / "drsr"))
sys.path.insert(0, str(ROOT / "scientific_intelligent_modelling" / "algorithms" / "llmsr_wrapper" / "llmsr"))

from drsr_420 import profile as drsr_profile
from llmsr import profile as llmsr_profile


class _DummyFunction:
    def __init__(self, sample_order: int, score: float):
        self.global_sample_nums = sample_order
        self.score = score
        self.sample_time = 0.1
        self.evaluate_time = 0.2
        self.params = [1.0, 2.0]
        self.optimized_params = [1.0, 2.0]

    def __str__(self):
        return "def equation(x0, params):\n    return params[0] * x0 + params[1]\n"


class SamplePersistenceSwitchTest(unittest.TestCase):
    def test_llmsr_default_only_keeps_topk_history_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler = llmsr_profile.Profiler(log_dir=tmp, samples_per_iteration=2)
            profiler.register_function(_DummyFunction(sample_order=1, score=-0.5))
            samples_dir = Path(tmp) / "samples"
            self.assertFalse((samples_dir / "samples_1.json").exists())
            self.assertTrue(any(p.name.startswith("top") for p in samples_dir.glob("*.json")))
            self.assertTrue((Path(tmp) / "best_history" / "best_sample_1.json").exists())
            self.assertTrue((Path(tmp) / "progress.json").exists())

    def test_llmsr_can_persist_all_samples_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler = llmsr_profile.Profiler(
                log_dir=tmp,
                samples_per_iteration=2,
                persist_all_samples=True,
            )
            profiler.register_function(_DummyFunction(sample_order=1, score=-0.5))
            self.assertTrue((Path(tmp) / "samples" / "samples_1.json").exists())

    def test_drsr_default_only_keeps_topk_history_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler = drsr_profile.Profiler(results_root=tmp, samples_per_iteration=2)
            profiler.register_function(_DummyFunction(sample_order=1, score=0.5))
            samples_dir = Path(tmp) / "samples"
            self.assertFalse((samples_dir / "samples_1.json").exists())
            self.assertTrue(any(p.name.startswith("top") for p in samples_dir.glob("*.json")))
            self.assertTrue((Path(tmp) / "best_history" / "best_sample_1.json").exists())
            self.assertTrue((Path(tmp) / "progress.json").exists())

    def test_drsr_can_persist_all_samples_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler = drsr_profile.Profiler(
                results_root=tmp,
                samples_per_iteration=2,
                persist_all_samples=True,
            )
            profiler.register_function(_DummyFunction(sample_order=1, score=0.5))
            self.assertTrue((Path(tmp) / "samples" / "samples_1.json").exists())


if __name__ == "__main__":
    unittest.main()
