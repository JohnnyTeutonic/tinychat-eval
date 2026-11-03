import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rag_eval

ROOT = Path(__file__).parent
MANUAL, _ = rag_eval.load()
CASES = {c["question"]: c for c in MANUAL["cases"]}


def fake_server():
    """A reader that is perfect given its excerpts: it answers with the first gold answer and cites the gold sections
    when every one of them is in the prompt, and says 'unknown' otherwise (including the raw condition)."""
    def call(url, payload=None, timeout=45):
        if url.endswith("/health"):
            return {"model": "/x/model.gguf", "context": 1024, "status": "ok"}, 0
        if url.endswith("/v1/models"):
            return {"data": [{"id": "model.gguf"}]}, 0
        text = payload["messages"][-1]["content"]
        question = text.split("Question: ", 1)[1].split("\n", 1)[0] if "Question: " in text else \
            text.split(": ", 1)[1].split("\n", 1)[0]
        case = CASES[question]
        if case["evidence"] and all(f"[{s}]" in text for s in case["evidence"]):
            reply = case["answers"][0] + " " + " ".join(f"[{s}]" for s in case["evidence"])
        else:
            reply = "unknown"
        return {"choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100}}, 1.0
    return call


class RagEvalTests(unittest.TestCase):
    def args(self, root, split="dev", allow=False):
        (root / "model.gguf").write_bytes(b"fixture")
        return argparse.Namespace(url="http://127.0.0.1:8081", weights=str(root / "model.gguf"), split=split,
                                  allow_holdout=allow, k=2, server_identity="fixture", candidate="fixture",
                                  out=str(root / "out"))

    def test_conditions_are_scored_separately(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch("evaluate.curl_json", side_effect=fake_server()):
                rag_eval.run(self.args(root))
            report = json.loads((root / "out" / "report.json").read_text())
            s = report["summary"]
            self.assertTrue(report["server_and_weights_stable"])
            self.assertEqual(s["raw"]["answer_exact"]["text"], "0 of 11")
            self.assertEqual(s["raw"]["false_refusal"]["text"], "11 of 11")
            self.assertEqual(s["raw"]["correct_refusal_when_absent"]["text"], "4 of 4")
            self.assertEqual(s["oracle"]["supported_answer"]["text"], "11 of 11")
            self.assertEqual(s["oracle"]["correct_refusal_when_absent"]["text"], "4 of 4")
            # the frozen retriever misses power_volts at k = 2, so even a perfect reader refuses there
            self.assertEqual(s["retrieved"]["supported_answer"]["text"], "10 of 11")
            self.assertEqual(s["retrieved"]["false_refusal"]["text"], "1 of 11")
            self.assertEqual(s["oracle"]["excerpt_echo"]["text"], "0 of 15")
            lines = (root / "out" / "episodes.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 15 * len(rag_eval.CONDITIONS))

    def test_scoring_separates_answer_citation_and_refusal(self):
        power = next(c for c in MANUAL["cases"] if c["id"] == "power_volts")
        absent = next(c for c in MANUAL["cases"] if c["id"] == "absent_weight")
        good = rag_eval.score_reply(power, "24 volts [S2]")
        self.assertTrue(good["supported_answer"])
        wrong_cite = rag_eval.score_reply(power, "24 volts [S18]")
        self.assertTrue(wrong_cite["answer_exact"])
        self.assertFalse(wrong_cite["citation_supported"])
        uncited = rag_eval.score_reply(power, "24 volts")
        self.assertTrue(uncited["answer_exact"])
        self.assertFalse(uncited["supported_answer"])
        wordy = rag_eval.score_reply(power, "The BR-40 runs on 24 volts. [S2]")
        self.assertFalse(wordy["answer_exact"])
        self.assertTrue(wordy["semantic_content"]["answer_mentioned"])
        echo = rag_eval.score_reply(power, "[S2] Power: The BR-40 runs on 24 volts DC.")
        self.assertTrue(echo["excerpt_echo"])
        self.assertFalse(echo["citation_supported"])
        self.assertFalse(rag_eval.score_reply(power, "[S2] 24 volts")["excerpt_echo"])
        self.assertTrue(rag_eval.score_reply(power, "unknown")["false_refusal"])
        self.assertTrue(rag_eval.score_reply(absent, "Unknown.")["correct_refusal"])
        hedge = rag_eval.score_reply(absent, "The excerpts do not say.")
        self.assertFalse(hedge["correct_refusal"])
        self.assertTrue(hedge["lenient_refusal"])

    def test_frozen_retrieval_recall(self):
        rep = rag_eval.retrieval_report(MANUAL, "all", 2)
        self.assertEqual(rep["recall"], "18 of 19")
        self.assertEqual([c["case_id"] for c in rep["cases"] if not c["recalled"]], ["power_volts"])

    def test_holdout_needs_the_flag_and_outputs_are_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch("evaluate.curl_json", side_effect=fake_server()):
                with self.assertRaises(ValueError):
                    rag_eval.run(self.args(root, split="holdout"))
                (root / "out").mkdir()
                with self.assertRaises(FileExistsError):
                    rag_eval.run(self.args(root))

    def test_manual_is_consistent_and_absent_from_the_suites(self):
        ids = {s["id"] for s in MANUAL["sections"]}
        self.assertEqual(len(ids), 20)
        absent = [c for c in MANUAL["cases"] if not c["evidence"]]
        self.assertTrue(all(c["near"] and c["answers"] == ["unknown"] for c in absent))
        self.assertEqual(sorted(collections_count(MANUAL["cases"]).items()), [("dev", 15), ("holdout", 11)])
        texts = " ".join(u.casefold() for name in ("suite.json", "suite_release_v002.json")
                         for c in json.loads((ROOT / name).read_text())["cases"] for u in c["users"])
        self.assertNotIn("br-40", texts)
        self.assertNotIn("brindle", texts)


def collections_count(cases):
    out = {}
    for c in cases:
        out[c["split"]] = out.get(c["split"], 0) + 1
    return out


if __name__ == "__main__":
    unittest.main()
