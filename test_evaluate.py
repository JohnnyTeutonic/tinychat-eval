import copy
import unittest
import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from evaluate import run
from evaluate import score, summarise, compare
from integrations import validate_judgment

class EvaluationTests(unittest.TestCase):
    def test_malformed_reply_and_final_health_failure_preserve_receipts(self):
        for malformed in ({"choices":None}, {"choices":[None]}, {"choices":[{"message":None}]}):
            with self.subTest(reply=malformed), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                (root/"model.gguf").write_bytes(b"fixture")
                (root/"suite.json").write_text(json.dumps({"cases":[{"id":"a","family":"qa","split":"dev","users":["Hi"],"check":{"kind":"exact","answers":["Hello"]}}]}))
                args=argparse.Namespace(url="http://127.0.0.1:8081",suite=str(root/"suite.json"),split="dev",allow_holdout=False,repeats=1,weights=str(root/"model.gguf"),out=str(root/"out"),candidate="fixture",server_identity="fixture")
                replies=[({"model":"model.gguf"},0),({"data":[{"id":"fixture"}]},0),(malformed,1),RuntimeError("health unavailable")]
                with patch("evaluate.curl_json",side_effect=replies), self.assertRaises(SystemExit):run(args)
                report=json.loads((root/"out/report.json").read_text())
                self.assertEqual(report["summary"]["operational_errors"],1)
                self.assertFalse(report["server_and_weights_stable"])
                self.assertIn("final_observation_error",report)
                row=json.loads((root/"out/episodes.jsonl").read_text())
                self.assertEqual(row["turns"][0]["response"],malformed)
                self.assertIsNone(row["exact_pass"])

    def test_exact_recall_cannot_be_gamed_by_listing_alternatives(self):
        case={"check":{"kind":"exact","answers":["Thursday"]}}
        self.assertTrue(score(case,"Thursday."))
        self.assertFalse(score(case,"Tuesday or Thursday"))
        self.assertFalse(score(case,"not Thursday"))
        self.assertFalse(score(case,""))

    def test_errors_and_unjudged_are_not_quality_passes(self):
        rows=[{"case_id":"a","family":"recall","complete":False,"turns":[],"repeat":0,"replies":[],"exact_pass":None}]
        result=summarise(rows)
        self.assertEqual(result["operational_errors"],1)
        self.assertEqual(result["exact_checks"],{})
        self.assertEqual(result["fluency"],"UNJUDGED")
        self.assertEqual(result["repeat_identical"]["n"],0)

    def test_repeat_disagreement_is_observed_even_with_same_final_answer(self):
        base={"case_id":"a","family":"recall","complete":True,"turns":[],"exact_pass":True}
        rows=[dict(base,repeat=0,replies=["Hello","Nora"]),dict(base,repeat=1,replies=["Hi","Nora"])]
        self.assertEqual(summarise(rows)["repeat_identical"],{"passed":0,"n":1})

    def test_judge_boolean_or_outside_rubric_refused(self):
        value={"fluency":3,"coherence":1,"relevance":0,"evidence":"Yes does not answer which person."}
        self.assertEqual(validate_judgment(value),value)
        for bad in (True,5,-1):
            with self.assertRaises(ValueError):validate_judgment(dict(value,fluency=bad))

    def test_changed_suite_or_missing_episodes_refuses_comparison(self):
        meta={k:1 for k in ("suite_sha256","harness_sha256","settings","split","repeats")}
        a={"metadata":meta,"server_and_weights_stable":True,"summary":{"operational_errors":0},"episodes":[]}
        b=copy.deepcopy(a);b["metadata"]["suite_sha256"]=2
        with self.assertRaises(ValueError):compare(a,b)
        b=copy.deepcopy(a);b["episodes"]=[{"case_id":"other","repeat":0}]
        with self.assertRaises(ValueError):compare(a,b)

if __name__=="__main__":unittest.main()
