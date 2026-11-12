"""The v0.0.3 release specification: its binding, its evaluator, and the three decisions it records."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import evaluate
import freeze_v003
import gate_v003

ROOT = Path(__file__).parent
GATE = gate_v003.load_gate()
SUITE = json.loads((ROOT / "suite_release_v002.json").read_text(encoding="utf-8"))
WEIGHTS = "ab" * 32


def health(verified):
    h = {"model": "/x/model.gguf", "status": "ok"}
    if verified:
        h["provenance"] = {"schema": "ember.provenance/1", "weights": {"status": "sha256", "sha256": WEIGHTS},
                           "build": {"git_commit": "fixture"}, "executable": {"sha256": "cd" * 32},
                           "runtime": {"ember_env": []}, "instance_id": "fixture"}
    return h


def write_run(root, name, split, passes, verified=True, repeats=2):
    episodes = []
    for case in (c for c in SUITE["cases"] if c["split"] == split):
        for r in range(repeats):
            scored = case["check"]["kind"] == "exact"
            episodes.append({"case_id": case["id"], "family": case["family"], "repeat": r, "complete": True,
                             "turns": [], "replies": ["fixture"],
                             "exact_pass": (case["id"] in passes) if scored else None})
    meta = {"suite_sha256": GATE["instrument"]["suite_sha256"], "harness_sha256": GATE["instrument"]["harness_sha256"],
            "settings": evaluate.SETTINGS, "split": split, "repeats": repeats, "history_mode": "generated",
            "weights_sha256": WEIGHTS, "health": health(verified), "server_identity": name}
    report = {"metadata": meta, "server_and_weights_stable": True, "episodes": episodes,
              "summary": evaluate.summarise(episodes)}
    d = Path(root) / name
    d.mkdir()
    (d / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return str(d)


GOOD_DEV = {"name_now", "pet_delayed", "meeting_corrected"}          # 3 of 11 scored, one per recall family
GOOD_HOLDOUT = {"holdout_container", "holdout_owner", "transfer_choice", "transfer_update"}  # 4 of 8


def runs(root, cand_verified=True, base_verified=True, dev=GOOD_DEV, base_dev=(), holdout=True):
    r = {"candidate_dev": write_run(root, "cd", "dev", dev, cand_verified),
         "baseline_dev": write_run(root, "bd", "dev", set(base_dev), base_verified)}
    if holdout:
        r["candidate_holdout"] = write_run(root, "ch", "holdout", GOOD_HOLDOUT, cand_verified)
        r["baseline_holdout"] = write_run(root, "bh", "holdout", set(), base_verified)
    return r


def judgments(root, name, score):
    path = Path(root) / name
    rows = [{"case_id": c, "turn_index": 0, "status": "JUDGED",
             "value": {"fluency": score, "coherence": score, "relevance": score, "evidence": "fixture"}}
            for c in GATE["judgments"]["subset"]]
    path.write_text(json.dumps({"mode": "per_turn", "judge_model": "fixture", "judgments": rows}), encoding="utf-8")
    return str(path)


def failing(result):
    return [c["name"] for c in result["criteria"] if c["passed"] is False]


class GateV003Tests(unittest.TestCase):
    def test_a_verified_run_meeting_every_criterion_passes(self):
        with tempfile.TemporaryDirectory() as d:
            result = gate_v003.evaluate_gate(**runs(d), export_parity=True)
            self.assertEqual(result["verdict"], "pass", failing(result))

    def test_an_unverified_run_fails_the_gate(self):
        for cand, base in ((False, True), (True, False)):
            with self.subTest(candidate_verified=cand, baseline_verified=base), tempfile.TemporaryDirectory() as d:
                result = gate_v003.evaluate_gate(**runs(d, cand, base), export_parity=True)
                self.assertEqual(result["verdict"], "fail")
                self.assertTrue(failing(result))
                self.assertTrue(all(name.endswith("provenance") for name in failing(result)), failing(result))

    def test_judgments_never_change_the_verdict(self):
        for parity, dev in ((True, GOOD_DEV), (True, {"name_now"}), (None, GOOD_DEV)):
            with self.subTest(parity=parity, dev=sorted(dev)), tempfile.TemporaryDirectory() as d:
                r = runs(d, dev=dev)
                plain = gate_v003.evaluate_gate(**r, export_parity=parity)
                for score in (0, 4):
                    judged = gate_v003.evaluate_gate(**r, export_parity=parity,
                                                     judgments=[judgments(d, f"j{score}.json", score)])
                    self.assertEqual(judged["verdict"], plain["verdict"])
                    self.assertEqual(judged["criteria"], plain["criteria"])
                    self.assertFalse(judged["diagnostics"]["judgments"][0]["affects_verdict"])
        self.assertEqual(GATE["judgments"]["role"], "diagnostic_only")
        self.assertIs(GATE["judgments"]["affects_verdict"], False)

    def test_dev_exact_denominator_is_the_scored_development_cases(self):
        self.assertEqual(len(gate_v003.scored_cases(SUITE, "dev")), 11)
        self.assertEqual(len([c for c in SUITE["cases"] if c["split"] == "dev"]), 13)
        with tempfile.TemporaryDirectory() as d:   # 3 of 11 = 0.27 meets 0.25; 3 of 13 = 0.23 would not
            result = gate_v003.evaluate_gate(**runs(d), export_parity=True)
            dev = next(c for c in result["criteria"] if c["name"] == "dev exact")
            self.assertEqual((dev["detail"]["passed"], dev["detail"]["denominator"], dev["passed"]), (3, 11, True))
        self.assertIn("Claude Opus 5.5, 13 Nov 2026", GATE["units"]["chosen_by"])
        self.assertIn("Jonathan may override", GATE["units"]["chosen_by"])

    def test_missing_holdout_or_parity_is_incomplete_never_pass(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(gate_v003.evaluate_gate(**runs(d, holdout=False), export_parity=True)["verdict"],
                             "incomplete")
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(gate_v003.evaluate_gate(**runs(d), export_parity=None)["verdict"], "incomplete")

    def test_a_case_regression_fails_even_with_a_higher_total(self):
        with tempfile.TemporaryDirectory() as d:
            dev = GOOD_DEV | {"object_now", "format_only"}
            result = gate_v003.evaluate_gate(**runs(d, dev=dev, base_dev={"colour_corrected"}), export_parity=True)
            self.assertEqual(result["verdict"], "fail")
            self.assertEqual(failing(result), ["no dev regression"])

    def test_the_missing_information_limit_is_stated(self):
        unknown = [c for c in SUITE["cases"] if c["family"] in ("missing_information", "transfer_unknown")]
        self.assertEqual(sorted(c["id"] for c in unknown), ["transfer_unknown", "unknown_name", "unknown_pet"])
        self.assertTrue(all(c["check"]["answers"] == ["no"] for c in unknown))
        self.assertTrue(any("unknown_name, unknown_pet, transfer_unknown" in s for s in GATE["stated_limits"]))
        self.assertIn("Not built here", GATE["proposed_next_suite"])

    def test_manifest_and_acceptance_bind_the_checked_out_files(self):
        manifest = json.loads((ROOT / "evaluation_manifest_v003.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest, freeze_v003.manifest(ROOT))
        self.assertEqual(manifest["files"]["evaluate.py"], GATE["instrument"]["harness_sha256"])
        self.assertEqual(manifest["files"]["suite_release_v002.json"], GATE["instrument"]["suite_sha256"])
        acceptance = json.loads((ROOT / "GATE_ACCEPTANCE_V003.json").read_text(encoding="utf-8"))
        sha = lambda n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()  # noqa: E731
        self.assertEqual(acceptance["gate_spec_sha256"], sha("release_gate_v003.json"))
        self.assertEqual(acceptance["evaluation_manifest_sha256"], sha("evaluation_manifest_v003.json"))
        self.assertIs(acceptance["fixed_before_any_v003_candidate_evaluated"], True)
        self.assertIs(acceptance["v003_candidate_outputs_evaluated_before_acceptance"], False)
        self.assertIs(acceptance["holdout_opened"], False)

    def test_v002_files_are_untouched(self):
        acceptance = json.loads((ROOT / "GATE_ACCEPTANCE_V002.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256((ROOT / "release_gate_v002.json").read_bytes()).hexdigest(),
                         acceptance["gate_spec_sha256"])
        self.assertEqual(hashlib.sha256((ROOT / "evaluation_manifest_v002_final.json").read_bytes()).hexdigest(),
                         acceptance["evaluation_manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
