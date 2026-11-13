"""The CANDIDATE v0.0.3 suites and their scorer (score_v003.py). Nothing here is frozen or read by a gate."""
import copy
import hashlib
import json
import os
import unittest
from pathlib import Path

import evaluate
import score_v003
from score_v003 import content_correct, validate_suite

ROOT = Path(__file__).parent
DEV = json.loads((ROOT / "suite_v003_candidate_dev.json").read_text(encoding="utf-8"))
CUE = json.loads((ROOT / "suite_v003_candidate_cue.json").read_text(encoding="utf-8"))
V002 = json.loads((ROOT / "suite_release_v002.json").read_text(encoding="utf-8"))
SEAL = json.loads((ROOT / "suite_v003_candidate_holdout.seal.json").read_text(encoding="utf-8"))
CASES = {c["id"]: c for c in DEV["cases"] + CUE["cases"]}


def case(answers, **rule):
    return {"id": "x", "family": "f", "split": "dev", "users": ["u"],
            "check": {"kind": "exact", "answers": answers, "content": rule}}


def report(rows, suite=DEV):
    episodes = []
    for cid, texts in rows.items():
        c = next(x for x in suite["cases"] if x["id"] == cid)
        for r, t in enumerate(texts):
            episodes.append({"case_id": cid, "family": c["family"], "repeat": r, "complete": t is not None,
                             "replies": [] if t is None else ["earlier", t]})
    return {"metadata": {}, "episodes": episodes}


class ContentRuleTests(unittest.TestCase):
    def test_value_rule_accepts_paraphrase_and_rejects_absence(self):
        c = case(["drawer"], require_any=["drawer"])
        for text in ("drawer", "The drawer.", "You left the keys in the drawer.", "  DRAWER!! ", "in the\ndrawer"):
            self.assertTrue(content_correct(c, text), text)
        for text in ("", "   ", "The cupboard.", "drawers", "I do not know."):
            self.assertFalse(content_correct(c, text), text)
        self.assertFalse(content_correct(c, None))

    def test_forbid_rejects_contradiction_unless_negated(self):
        c = CASES["meeting_corrected"]
        for text in ("Thursday", "The meeting is on Thursday, not Tuesday.", "Thursday, no longer Tuesday.",
                     "Thursday instead of Tuesday"):
            self.assertTrue(content_correct(c, text), text)
        for text in ("Tuesday or Thursday", "Tuesday", "Thursday. Or maybe Tuesday.", "It was Tuesday, then Thursday"):
            self.assertFalse(content_correct(c, text), text)

    def test_role_reversal_is_rejected(self):
        self.assertFalse(content_correct(CASES["name_now"], "My name is Nora."))
        self.assertTrue(content_correct(CASES["name_now"], "Your name is Nora."))
        self.assertTrue(content_correct(CASES["name_now"], "Nora"))
        self.assertFalse(content_correct(CASES["imm_rabbit"], "My rabbit is called Pickle."))

    def test_negative_polarity(self):
        c = CASES["unknown_name"]
        for text in ("No.", "no", "Nope", "I have not told you.", "You haven't told me your name.",
                     "You haven’t told me.", "Not yet.", "You never said."):
            self.assertTrue(content_correct(c, text), text)
        for text in ("Yes.", "Yes, you have not.", "I bought a notebook.", "I have a cat, but I like cats.", ""):
            self.assertFalse(content_correct(c, text), text)

    def test_affirmative_polarity(self):
        c = CASES["told_pet"]
        for text in ("Yes.", "yes", "Yes, Juniper.", "Your cat is called Juniper.", "YES"):
            self.assertTrue(content_correct(c, text), text)
        for text in ("No.", "Nope, Juniper", "You haven't told me.", "I do not know.", "Maybe.", "Juniper? I don't know."):
            self.assertFalse(content_correct(c, text), text)

    def test_numbers_words_and_punctuation(self):
        c = CASES["cor_locker"]
        for text in ("41", "41.", "Forty-one", "forty one", "It is 41, not 14."):
            self.assertTrue(content_correct(c, text), text)
        for text in ("14", "14 or 41", "4 1", "141"):
            self.assertFalse(content_correct(c, text), text)

    def test_strict_scoring_is_evaluate_py_unchanged(self):
        c = CASES["imm_sister"]
        self.assertTrue(evaluate.score(c, "Tenby."))
        self.assertFalse(evaluate.score(c, "Your sister lives in Tenby."))
        self.assertTrue(content_correct(c, "Your sister lives in Tenby."))
        self.assertTrue(evaluate.score(CASES["ins_digits"], "20"))
        self.assertFalse(evaluate.score(CASES["ins_digits"], "twenty"))
        self.assertTrue(content_correct(CASES["ins_digits"], "twenty"))


class SuiteValidationTests(unittest.TestCase):
    def test_candidate_suites_validate_with_the_stated_counts(self):
        dev, cue = validate_suite(DEV), validate_suite(CUE)
        self.assertEqual({k: v["scored"] for k, v in dev["families"].items()},
                         {"basic_qa": 6, "immediate_recall": 6, "delayed_recall": 6, "correction": 6, "told_or_not": 8,
                          "instruction": 6, "open_conversation": 0})
        self.assertEqual((dev["cases"], dev["twin_pairs"]), (40, 4))
        self.assertEqual((cue["cases"], cue["cue_triplets"]), (12, 4))
        self.assertTrue(all(c.get("diagnostic") is True for c in CUE["cases"]))

    def test_v002_cases_keep_their_wording_and_gold_and_its_holdout_is_left_out(self):
        v002 = {c["id"]: c for c in V002["cases"]}
        for c in DEV["cases"]:
            if c["origin"] == "v002":
                self.assertEqual((c["users"], c["check"]["kind"], c["check"].get("answers")),
                                 (v002[c["id"]]["users"], v002[c["id"]]["check"]["kind"],
                                  v002[c["id"]]["check"].get("answers")), c["id"])
                self.assertEqual(v002[c["id"]]["split"], "dev")
        holdout_texts = {u for c in V002["cases"] if c["split"] == "holdout" for u in c["users"]}
        for c in DEV["cases"] + CUE["cases"]:
            self.assertNotIn(c["id"], {x["id"] for x in V002["cases"] if x["split"] == "holdout"})
            self.assertFalse(set(c["users"]) & holdout_texts, c["id"])

    def test_a_constant_answer_passes_at_most_half_of_told_or_not(self):
        told = [c for c in DEV["cases"] if c["family"] == "told_or_not"]
        for text in ("yes", "no", "Yes.", "No.", "I don't know.", "Maybe"):
            strict = sum(evaluate.score(c, text) for c in told)
            content = sum(content_correct(c, text) for c in told)
            self.assertLessEqual(strict, len(told) // 2, text)
            self.assertLessEqual(content, len(told) // 2, text)

    def test_malformed_suites_are_refused(self):
        def broken(mutate, suite=DEV):
            s = copy.deepcopy(suite)
            mutate(s)
            with self.assertRaises(ValueError):
                validate_suite(s)
        by = lambda s, i: next(c for c in s["cases"] if c["id"] == i)  # noqa: E731
        broken(lambda s: s["cases"].append(copy.deepcopy(s["cases"][0])))
        broken(lambda s: by(s, "qa_sum")["check"].update(answers="5"))
        broken(lambda s: by(s, "qa_sum")["check"].update(answers=["fünf"]))
        broken(lambda s: by(s, "qa_sum")["check"].update(answers=["!!"]))
        broken(lambda s: by(s, "qa_sum")["check"].pop("content"))
        broken(lambda s: by(s, "qa_sum")["check"]["content"].update(require_any=["six"]))
        broken(lambda s: by(s, "qa_sum")["check"]["content"].update(forbid=["  "]))
        broken(lambda s: by(s, "qa_sum")["check"]["content"].update(polarity="maybe"))
        broken(lambda s: by(s, "qa_sum")["check"]["content"].update(require=["5"]))
        broken(lambda s: by(s, "told_pet")["users"].__setitem__(1, "Did I say my cat's name? Reply yes or no."))
        broken(lambda s: by(s, "told_pet")["check"].update(answers=["no"], content={"polarity": "negative"}))
        broken(lambda s: by(s, "told_pet")["twin"].update(told=False))
        broken(lambda s: by(s, "cue_tins_N")["check"].update(answers=["seeds"], content={"require_any": ["seeds"]}), CUE)
        broken(lambda s: by(s, "cue_tins_B")["check"].update(answers=["stamps"], content={"require_any": ["stamps"]}), CUE)
        broken(lambda s: by(s, "cue_tins_B")["users"].__setitem__(1, "Which tin holds seeds?"), CUE)
        broken(lambda s: by(s, "cue_tins_A").pop("diagnostic"), CUE)
        broken(lambda s: s["cases"].remove(by(s, "cue_tins_N")), CUE)

    def test_cue_triplets_change_gold_on_the_decisive_edit_only(self):
        for name in ("tins", "shops", "kites", "blanket"):
            a, b, n = (CASES[f"cue_{name}_{r}"] for r in "ABN")
            self.assertNotEqual(a["check"]["answers"], b["check"]["answers"])
            self.assertEqual(a["check"]["answers"], n["check"]["answers"])


class ScoreReportTests(unittest.TestCase):
    def test_distinct_case_unit_strict_and_content_separately(self):
        rows = {"name_now": ["Nora", "Nora"], "imm_sister": ["Your sister lives in Tenby."] * 2,
                "imm_floor": ["4", "4"], "imm_rabbit": ["Pickle", "Bun"], "object_now": ["My bicycle is blue."] * 2,
                "imm_drink": [None, None]}
        f = score_v003.score_report(report(rows), DEV)["families"]["immediate_recall"]
        self.assertEqual((f["n"], f["strict"], f["content"]), (6, 2, 3))
        self.assertEqual(f["content_not_strict"], ["imm_sister"])
        cases = score_v003.score_report(report(rows), DEV)["cases"]
        self.assertFalse(cases["imm_rabbit"]["strict"])          # repeats disagree: not a pass
        self.assertFalse(cases["imm_rabbit"]["repeats_agree"])
        self.assertFalse(cases["imm_drink"]["complete"])

    def test_twin_pairs_and_cue_triplets(self):
        rows = {"unknown_name": ["No."], "told_name": ["Yes."], "unknown_pet": ["No."], "told_pet": ["No."]}
        t = score_v003.score_report(report(rows), DEV)["twin_pairs"]
        self.assertEqual((t["pairs"], t["both_strict"], t["both_content"]), (2, 1, 1))
        rows = {"cue_tins_A": ["stamps"], "cue_tins_B": ["Seeds."], "cue_tins_N": ["stamps"],
                "cue_kites_A": ["grey"], "cue_kites_B": ["grey"], "cue_kites_N": ["grey"]}
        r = score_v003.score_report(report(rows, CUE), CUE)
        self.assertEqual(r["cue_summary"]["selective"], 1)
        self.assertEqual(r["cue_triplets"]["kites"]["B_followed_decisive_edit"], False)
        self.assertTrue(r["families"]["cue_sensitivity"]["diagnostic"])

    def test_the_baseline_replies_read_as_the_audit_describes(self):
        """S-full's recorded v002 replies (runs/v002-baseline-generated-20261103) under the content rules."""
        expect = {"name_now": ("My name is Nora.", False), "unknown_name": ("I have not told you.", True),
                  "unknown_pet": ("I have a cat, but I like cats.", False), "pet_delayed": ("Its name is Ted.", False),
                  "meeting_corrected": ("I'm sorry. I'm not sure.", False), "format_only": (
                      "Lanterns are the same, I'm not sure if they are the same", False)}
        for cid, (text, want) in expect.items():
            self.assertEqual(content_correct(CASES[cid], text), want, cid)
            self.assertFalse(evaluate.score(CASES[cid], text))

    def test_the_frozen_runner_runs_the_candidate_suite_and_the_scorer_reads_its_report(self):
        import argparse
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "model.gguf").write_bytes(b"fixture")
            health = ({"model": "/x/model.gguf"}, 0)
            turns = sum(len(c["users"]) for c in DEV["cases"])
            replies = [health, ({"data": [{"id": "m"}]}, 0)] + \
                [({"choices": [{"message": {"content": "No."}}]}, 1)] * turns + [health]
            args = argparse.Namespace(url="http://127.0.0.1:8081", suite=str(ROOT / "suite_v003_candidate_dev.json"),
                                      split="dev", allow_holdout=False, repeats=1, weights=str(root / "model.gguf"),
                                      out=str(root / "out"), candidate="c", server_identity="s")
            with patch("evaluate.curl_json", side_effect=replies):
                evaluate.run(args)
            rep = json.loads((root / "out" / "report.json").read_text())
            result = score_v003.score_report(rep, DEV)
            told = result["families"]["told_or_not"]
            self.assertEqual((told["strict"], told["content"], told["n"]), (4, 4, 8))   # a constant "No." gets half
            self.assertEqual(result["twin_pairs"]["both_strict"], 0)
            self.assertEqual(score_v003.main([str(root / "out" / "report.json"), "--suite",
                                              str(ROOT / "suite_v003_candidate_dev.json")]), 0)
            with self.assertRaises(SystemExit):
                score_v003.main([str(root / "out" / "report.json"), "--suite", str(ROOT / "suite_v003_candidate_cue.json")])

    def test_a_report_case_outside_the_suite_is_refused(self):
        bad = report({"qa_sum": ["5"]})
        bad["episodes"][0]["case_id"] = "not_a_case"
        with self.assertRaises(ValueError):
            score_v003.score_report(bad, DEV)


class SealTests(unittest.TestCase):
    def test_the_seal_records_a_hash_and_counts_only(self):
        self.assertRegex(SEAL["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(sum(SEAL["families"].values()), SEAL["cases"])
        self.assertIn("CANDIDATE", SEAL["status"])
        self.assertFalse(list(ROOT.glob("*holdout*.json")) != [ROOT / "suite_v003_candidate_holdout.seal.json"])

    @unittest.skipUnless(os.environ.get("TCE_SEALED_HOLDOUT"), "opt-in: TCE_SEALED_HOLDOUT=path to the sealed holdout")
    def test_the_sealed_holdout_matches_its_seal(self):
        data = Path(os.environ["TCE_SEALED_HOLDOUT"]).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), SEAL["sha256"])
        self.assertEqual(len(data), SEAL["bytes"])
        held = json.loads(data)
        summary = validate_suite(held)
        self.assertEqual({k: v["scored"] for k, v in summary["families"].items()}, SEAL["families"])
        self.assertEqual(summary["twin_pairs"], SEAL["twin_pairs"])
        self.assertTrue(all(c["split"] == "holdout" for c in held["cases"]))
        others = {c["id"] for c in DEV["cases"] + CUE["cases"] + V002["cases"]}
        self.assertFalse({c["id"] for c in held["cases"]} & others)
        texts = {u for c in DEV["cases"] + CUE["cases"] + V002["cases"] for u in c["users"]}
        self.assertFalse({u for c in held["cases"] for u in c["users"]} & texts)


if __name__ == "__main__":
    unittest.main()
