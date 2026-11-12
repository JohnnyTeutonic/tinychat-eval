"""The accepted v0.0.2 gate and the calibration record must bind the bytes a checkout holds.

GATE_ACCEPTANCE_V002.json and calibration/human_final_reply_ratings_v002_sfull.json record SHA-256 hashes of frozen
files. Those hashes were taken from Windows (CRLF) checkouts, so each bound file must check out with CRLF on every
platform (.gitattributes), or the binding cannot be verified from a Linux or WSL clone.
"""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parent


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


class AcceptanceBindingTests(unittest.TestCase):
    def test_gate_acceptance_binds_the_checked_out_gate_and_manifest(self):
        acceptance = json.loads((ROOT / "GATE_ACCEPTANCE_V002.json").read_text(encoding="utf-8"))
        self.assertEqual(sha("release_gate_v002.json"), acceptance["gate_spec_sha256"])
        self.assertEqual(sha("evaluation_manifest_v002_final.json"), acceptance["evaluation_manifest_sha256"])
        gate = json.loads((ROOT / "release_gate_v002.json").read_text(encoding="utf-8"))
        self.assertEqual(acceptance["accepted_criteria"], gate)

    def test_calibration_record_binds_the_checked_out_rubric_and_suite(self):
        record = json.loads((ROOT / "calibration/human_final_reply_ratings_v002_sfull.json").read_text(encoding="utf-8"))
        self.assertEqual(sha("integrations.py"), record["rubric_source_sha256"])
        self.assertEqual(sha(record["suite"]), record["suite_sha256"])


if __name__ == "__main__":
    unittest.main()
