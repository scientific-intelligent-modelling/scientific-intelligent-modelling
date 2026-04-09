import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scientific_intelligent_modelling.benchmarks.judges import (
    LLMSymbolicJudge,
    llm_srbench_symbolic_accuracy,
)


class _FakeClient:
    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return {
            "content": json.dumps(
                {
                    "equivalent": True,
                    "reasoning": "same structure",
                    "confidence": 0.95,
                }
            )
        }


class BenchmarkJudgeTest(unittest.TestCase):
    def test_judge_uses_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "judge_cache.json"
            judge = LLMSymbolicJudge(
                config={
                    "model": "openai/gpt-4o",
                    "api_key": "dummy",
                },
                cache_path=cache_path,
            )
            fake_client = _FakeClient()
            with patch.object(judge, "_get_client", return_value=fake_client):
                r1 = judge.judge(gold_equation="x + 1", predicted_equation="x + 1")
                r2 = judge.judge(gold_equation="x + 1", predicted_equation="x + 1")
            self.assertTrue(r1["equivalent"])
            self.assertTrue(r2["equivalent"])
            self.assertFalse(r1["cached"])
            self.assertTrue(r2["cached"])
            self.assertEqual(fake_client.calls, 1)

    def test_llm_srbench_symbolic_accuracy_wrapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            judge = LLMSymbolicJudge(
                config={"model": "openai/gpt-4o", "api_key": "dummy"},
                cache_path=Path(tmpdir) / "judge_cache.json",
            )
            with patch.object(
                judge,
                "judge",
                return_value={"equivalent": True, "reasoning": "ok", "confidence": 0.9},
            ):
                result = llm_srbench_symbolic_accuracy(
                    "x+1",
                    "x+1",
                    judge=judge,
                )
            self.assertEqual(result["symbolic_accuracy"], 1.0)
            self.assertTrue(result["judge_result"]["equivalent"])


if __name__ == "__main__":
    unittest.main()
