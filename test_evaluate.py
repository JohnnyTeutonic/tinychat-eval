import copy
import unittest
from evaluate import score, summarise, compare
from integrations import validate_judgment

class EvaluationTests(unittest.TestCase):
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
