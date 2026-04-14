import json
import tempfile
import unittest
from pathlib import Path

from scientific_intelligent_modelling.benchmarks.result_archive import write_result_payload


class ResultArchiveTest(unittest.TestCase):
    def test_write_result_payload_writes_primary_and_experiment_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "outer" / "result.json"
            exp_dir = root / "experiments" / "demo_exp"
            payload = {"tool": "pysr", "status": "ok"}

            written = write_result_payload(
                payload,
                primary_path=primary,
                experiment_dir=exp_dir,
            )

            self.assertEqual(len(written), 2)
            self.assertEqual(json.loads(primary.read_text(encoding="utf-8")), payload)
            self.assertEqual(
                json.loads((exp_dir / "result.json").read_text(encoding="utf-8")),
                payload,
            )

    def test_write_result_payload_deduplicates_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "experiments" / "demo_exp" / "result.json"
            payload = {"tool": "drsr", "status": "ok"}

            written = write_result_payload(
                payload,
                primary_path=primary,
                experiment_dir=primary.parent,
            )

            self.assertEqual(len(written), 1)
            self.assertEqual(json.loads(primary.read_text(encoding="utf-8")), payload)


if __name__ == "__main__":
    unittest.main()
