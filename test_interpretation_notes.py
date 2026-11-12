"""Behaviour the README's interpretation notes describe. The code they test is frozen (evaluate.py by the v0.0.2
manifest; rag_eval.py since its first receipts), so these tests pin the behaviour the notes explain. If a later
instrument version changes it, a test fails and the README note must change with it."""
import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate
import rag_eval

ROOT = Path(__file__).parent
UNKNOWN_FAMILIES = {"missing_information", "transfer_unknown"}


def completed(body, status=200):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=body + b"\n" + str(status).encode(), stderr=b"")


class InterpretationNoteTests(unittest.TestCase):
    def test_a_constant_no_passes_every_unknown_information_case(self):
        for name in ("suite.json", "suite_release_v002.json"):
            cases = [c for c in json.loads((ROOT / name).read_text())["cases"] if c["family"] in UNKNOWN_FAMILIES]
            self.assertTrue(cases)
            self.assertTrue(all(evaluate.score(c, "No.") for c in cases), name)

    def test_summary_counts_identical_greedy_repeats_as_separate_checks(self):
        rows = [{"case_id": "a", "family": "f", "repeat": r, "complete": True, "turns": [], "replies": ["x"],
                 "exact_pass": False} for r in (0, 1)]
        self.assertEqual(evaluate.summarise(rows)["exact_checks"]["f"]["n"], 2)

    def test_raw_abstention_is_filed_under_false_refusal(self):
        manual, _ = rag_eval.load()
        power = next(c for c in manual["cases"] if c["id"] == "power_volts")
        self.assertTrue(rag_eval.score_reply(power, "unknown")["false_refusal"])

    def test_a_non_json_success_body_is_an_operational_error_without_its_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "model.gguf").write_bytes(b"fixture")
            case = {"id": "a", "family": "qa", "split": "dev", "users": ["Hi"],
                    "check": {"kind": "exact", "answers": ["Hello"]}}
            (root / "suite.json").write_text(json.dumps({"cases": [case]}))
            health = json.dumps({"model": "/x/model.gguf"}).encode()
            replies = [completed(health), completed(json.dumps({"data": [{"id": "m"}]}).encode()),
                       completed(b"<html>proxy page</html>"), completed(health)]
            args = argparse.Namespace(url="http://127.0.0.1:8081", suite=str(root / "suite.json"), split="dev",
                                      allow_holdout=False, repeats=1, weights=str(root / "model.gguf"),
                                      out=str(root / "out"), candidate="c", server_identity="s")
            with patch("evaluate.subprocess.run", side_effect=replies), self.assertRaises(SystemExit):
                evaluate.run(args)
            text = (root / "out" / "report.json").read_text()
            report = json.loads(text)
            self.assertEqual(report["summary"]["operational_errors"], 1)
            self.assertIsNone(report["episodes"][0]["exact_pass"])
            self.assertTrue(report["episodes"][0]["error"].startswith("JSONDecodeError"))
            self.assertNotIn("proxy page", text)


if __name__ == "__main__":
    unittest.main()
