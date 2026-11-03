import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import context_sweep
from context_sweep import FILLER, PROBES, SCHEDULE, mark_context, messages_for

ROOT = Path(__file__).parent


def fake_server(cap=1000, base=20, per_filler=18, malformed_at=None):
    """curl_json stand-in: /health and /v1/models, then chat completions whose prompt grows by per_filler tokens per
    filler exchange until it reaches cap, where the oldest exchanges are dropped (the prompt stops growing). The reply
    is the planted answer while the fact is in context, and 'I do not know.' after it was dropped."""
    def call(url, payload=None, timeout=45):
        if url.endswith("/health"):
            return {"model": "/x/model.gguf", "context": 1024, "status": "ok"}, 0
        if url.endswith("/v1/models"):
            return {"data": [{"id": "model.gguf"}]}, 0
        msgs = payload["messages"]
        fillers = (len(msgs) - 3) // 2
        if malformed_at is not None and fillers == malformed_at:
            return {"choices": None}, 0
        tokens = base + per_filler * fillers
        fact_kept = tokens <= cap
        tokens = min(tokens, cap)
        probe = next(p for p in PROBES if p["question"] == msgs[-1]["content"])
        text = probe["answers"][0] if fact_kept else "I do not know."
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": tokens, "completion_tokens": 3}}, 1.0
    return call


class ContextSweepTests(unittest.TestCase):
    def args(self, root):
        (root / "model.gguf").write_bytes(b"fixture")
        return argparse.Namespace(url="http://127.0.0.1:8081", weights=str(root / "model.gguf"),
                                  server_identity="fixture", candidate="fixture", out=str(root / "out"))

    def test_marks_the_onset_of_dropped_history_and_records_every_point(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch("evaluate.curl_json", side_effect=fake_server()):
                context_sweep.run(self.args(root))
            report = json.loads((root / "out" / "report.json").read_text())
            self.assertEqual(report["operational_errors"], 0)
            self.assertTrue(report["server_and_weights_stable"])
            lines = (root / "out" / "points.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), len(FILLER) + 1 + len(PROBES) * len(SCHEDULE))
            self.assertEqual(report["filler_costs"], [18] * len(FILLER))
            self.assertEqual(report["length_mismatches"], 0)
            for r in report["probes"]:
                # 20 + 18 F <= 1000 holds up to F = 54: F = 48 is the last in-context point, F = 56 the first dropped
                self.assertEqual(r["last_in_context"], {"fillers": 48, "prompt_tokens": 20 + 18 * 48})
                self.assertEqual(r["first_history_dropped"], {"fillers": 56, "prompt_tokens": 1000})
                self.assertEqual(r["exact_in_context"], "11 of 11")
                after = [p for p in r["points"] if p["in_context"] is False]
                self.assertEqual([p["fillers"] for p in after], [56, 64])
                self.assertTrue(all(p["exact_pass"] is False for p in after))

    def test_a_malformed_reply_is_kept_and_fails_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch("evaluate.curl_json", side_effect=fake_server(malformed_at=12)), self.assertRaises(SystemExit):
                context_sweep.run(self.args(root))
            report = json.loads((root / "out" / "report.json").read_text())
            self.assertEqual(report["operational_errors"], len(PROBES))
            bad = [json.loads(x) for x in (root / "out" / "points.jsonl").read_text().splitlines()
                   if not json.loads(x)["complete"]]
            self.assertTrue(all(p["response"] == {"choices": None} for p in bad))

    def test_an_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "out").mkdir()
            with patch("evaluate.curl_json", side_effect=fake_server()), self.assertRaises(FileExistsError):
                context_sweep.run(self.args(root))

    def test_a_failed_calibration_stops_before_the_sweep(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch("evaluate.curl_json", side_effect=fake_server(malformed_at=4)), self.assertRaises(SystemExit):
                context_sweep.run(self.args(root))
            report = json.loads((root / "out" / "report.json").read_text())
            self.assertIn("calibration incomplete", report["calibration_error"])
            lines = (root / "out" / "points.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), len(FILLER) + 1)

    def test_mark_context_needs_the_exact_predicted_length(self):
        costs = [10, 12]
        pts = [{"fillers": 0, "prompt_tokens": 20}, {"fillers": 3, "prompt_tokens": 52},
               {"fillers": 4, "prompt_tokens": 60}, {"fillers": 6, "prompt_tokens": 86}]
        mark_context(pts, costs)
        # predicted 20, 20+10+12+10 = 52, 64, 86: F = 4 reports 60 < 64, so it and every later point are out
        self.assertEqual([p["predicted_tokens"] for p in pts], [20, 52, 64, 86])
        self.assertEqual([p["in_context"] for p in pts], [True, True, False, False])

    def test_a_longer_than_predicted_prompt_is_a_mismatch_not_context(self):
        pts = [{"fillers": 0, "prompt_tokens": 20}, {"fillers": 2, "prompt_tokens": 45}]
        mark_context(pts, [10, 12])
        self.assertIsNone(pts[1]["in_context"])
        self.assertTrue(pts[1]["length_mismatch"])

    def test_probes_are_not_in_the_evaluation_suites(self):
        texts = set()
        for name in ("suite.json", "suite_release_v002.json"):
            for c in json.loads((ROOT / name).read_text())["cases"]:
                texts.update(u.casefold() for u in c["users"])
        for p in PROBES:
            self.assertNotIn(p["fact"].casefold(), texts)
            self.assertNotIn(p["question"].casefold(), texts)
        m = messages_for(PROBES[0], 3)
        self.assertEqual([x["role"] for x in m], ["user", "assistant"] * 4 + ["user"])
        self.assertTrue(all(x["content"] == "Noted." for x in m if x["role"] == "assistant"))


if __name__ == "__main__":
    unittest.main()
