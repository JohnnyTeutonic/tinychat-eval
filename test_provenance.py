"""Provenance checks (provenance.py) against a fake server: no network, no model."""
import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import evaluate
import provenance

ROOT = Path(__file__).parent
FROZEN = ("evaluate.py", "context_sweep.py", "rag_eval.py")


def ember_provenance(sha, instance="a" * 32, status="sha256", env=None, settings=None, side_files=None):
    return {"schema": "ember.provenance/1", "instance_id": instance,
            "weights": {"status": status, "sha256": sha if status == "sha256" else None, "bytes": 7,
                        "path": "/models/model.gguf", "monitored": True},
            "build": {"git_commit": "0" * 40, "git_dirty": False, "backend": "cpu", "config": "Release",
                      "compiler": "GNU 13.3.0"},
            "executable": {"path": "/bin/tinyllama_server", "sha256": "e" * 64},
            "runtime": {"gpu_layers_requested": 0, "ember_env": env or [], "ember_settings": settings or {},
                        "ember_side_files": side_files or {}}}


class FakeServer:
    """/health, /v1/models and /v1/chat/completions; `swap_after_chat` replaces /health after N chat calls."""

    def __init__(self, prov, swap_after_chat=None, swapped=None):
        self.prov, self.swap_after_chat, self.swapped, self.chats, self.calls = prov, swap_after_chat, swapped, 0, []

    def health(self):
        h = {"status": "ok", "model": "/models/model.gguf", "context": 1024}
        prov = self.swapped if self.swap_after_chat is not None and self.chats >= self.swap_after_chat else self.prov
        if prov is not None:
            h["provenance"] = copy.deepcopy(prov)
        return h

    def __call__(self, url, payload=None, timeout=45):
        path = url.split("8081", 1)[1]
        self.calls.append(path)
        if path == "/health":
            return self.health(), 0.1
        if path == "/v1/models":
            return {"data": [{"id": "model.gguf"}]}, 0.1
        self.chats += 1
        return {"choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}]}, 1.0


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.weights = self.root / "model.gguf"
        self.weights.write_bytes(b"fixture")
        self.sha = hashlib.sha256(b"fixture").hexdigest()
        case = {"id": "a", "family": "qa", "split": "dev", "users": ["Hi", "Again"],
                "check": {"kind": "exact", "answers": ["Hello"]}}
        (self.root / "suite.json").write_text(json.dumps({"cases": [case]}))
        self.frozen = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FROZEN}

    def tearDown(self):
        self.tmp.cleanup()
        # The instruments are unchanged by the test run: compared with the snapshot taken in setUp. The
        # v0.0.2 manifest binds only evaluate.py among them (test_frozen_evaluate_matches_v002_manifest).
        self.assertEqual(self.frozen, {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FROZEN})

    def args(self, out="out"):
        return ["--suite", str(self.root / "suite.json"), "--repeats", "1", "--weights", str(self.weights),
                "--server-identity", "operator-label", "--candidate", "fixture", "--out", str(self.root / out)]

    def run_with(self, server, allow_unverified=False, out="out"):
        with patch("evaluate.curl_json", side_effect=server), redirect_stdout(io.StringIO()):
            return provenance.run("evaluate", self.args(out), allow_unverified)

    def test_verified_run_records_server_identity_and_verifies(self):
        server = FakeServer(ember_provenance(self.sha))
        self.assertEqual(self.run_with(server), 0)
        record = json.loads((self.root / "out/provenance.json").read_text())
        self.assertEqual(record["status"], "verified")
        self.assertEqual(record["start"]["provenance"]["weights"]["sha256"], self.sha)
        self.assertEqual(record["end"]["provenance"], record["start"]["provenance"])
        self.assertEqual(record["instrument_sha256"], self.frozen["evaluate.py"])
        result = provenance.verify(self.root / "out")
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["build"]["git_commit"], "0" * 40)
        self.assertEqual(server.calls[0], "/health")  # preflight before the instrument's first request

    def test_identity_mismatch_refuses_before_any_output(self):
        server = FakeServer(ember_provenance("f" * 64))
        with self.assertRaises(provenance.ProvenanceRefused):
            self.run_with(server)
        self.assertFalse((self.root / "out").exists())
        self.assertEqual(server.calls, ["/health"])
        with patch("evaluate.curl_json", side_effect=server), redirect_stderr(io.StringIO()):
            self.assertEqual(provenance.main(["run", "evaluate", "--"] + self.args("cli")), 3)
        self.assertFalse((self.root / "cli").exists())

    def test_server_restarted_on_other_weights_mid_run_is_mismatch(self):
        # Changed mid-run, and the end server serves other weights: the worse status, mismatch, stands.
        server = FakeServer(ember_provenance(self.sha), swap_after_chat=1,
                            swapped=ember_provenance("f" * 64, instance="b" * 32))
        self.assertEqual(self.run_with(server), 2)  # the frozen instrument's own stability check fails too
        record = json.loads((self.root / "out/provenance.json").read_text())
        self.assertEqual(record["status"], "mismatch")
        self.assertIn("server provenance changed between start and end of the run", record["reasons"])
        self.assertEqual(record["end"]["provenance"]["weights"]["sha256"], "f" * 64)
        report = json.loads((self.root / "out/report.json").read_text())
        self.assertFalse(report["server_and_weights_stable"])
        self.assertEqual(provenance.verify(self.root / "out")["status"], "mismatch")

    def test_restart_on_the_same_weights_is_still_a_change(self):
        server = FakeServer(ember_provenance(self.sha), swap_after_chat=1,
                            swapped=ember_provenance(self.sha, instance="b" * 32))
        self.assertEqual(self.run_with(server), 2)
        self.assertEqual(json.loads((self.root / "out/provenance.json").read_text())["status"], "changed")

    def test_server_without_digest_is_refused_or_recorded_unverified(self):
        for prov in (None, ember_provenance(None, status="disabled"),
                     ember_provenance(None, status="unsupported_format"), ember_provenance("not-hex")):
            with self.subTest(prov=prov):
                out = "out-" + hashlib.sha256(json.dumps(prov).encode()).hexdigest()[:8]
                with self.assertRaises(provenance.ProvenanceRefused):
                    self.run_with(FakeServer(prov), out=out)
                self.assertFalse((self.root / out).exists())
                self.assertEqual(self.run_with(FakeServer(prov), allow_unverified=True, out=out), 0)
                record = json.loads((self.root / out / "provenance.json").read_text())
                self.assertEqual(record["status"], "unverified")
                self.assertTrue(record["allow_unverified"])
                self.assertEqual(provenance.verify(self.root / out)["status"], "unverified")

    def test_unverified_runs_never_compare_silently(self):
        self.run_with(FakeServer(None), allow_unverified=True, out="old")
        self.run_with(FakeServer(ember_provenance(self.sha)), out="new")
        with self.assertRaises(ValueError):
            provenance.compare(self.root / "old", self.root / "new")
        result = provenance.compare(self.root / "old", self.root / "new", allow_unverified=True)
        self.assertEqual(result["provenance"]["before"]["status"], "unverified")
        self.assertEqual(result["provenance"]["after"]["status"], "verified")

    def test_changed_run_refuses_comparison_even_when_allowed(self):
        self.run_with(FakeServer(ember_provenance(self.sha)), out="a")
        self.run_with(FakeServer(ember_provenance(self.sha), swap_after_chat=1,
                                 swapped=ember_provenance("f" * 64, instance="b" * 32)), out="b")
        with self.assertRaises(ValueError):
            provenance.compare(self.root / "a", self.root / "b", allow_unverified=True)

    def test_legacy_receipt_loads_unchanged_and_is_unverified(self):
        # The shape of the 3 Nov 2026 receipts: /health without provenance, no provenance.json.
        legacy = self.root / "legacy"
        legacy.mkdir()
        meta = {"health": {"model": "/m/tc_sft_Sfull_r1.gguf", "status": "ok", "context": 1024},
                "weights_sha256": "b" * 64, "server_identity": "archived-ember-cpu-server-Sfull-20261103",
                "suite_sha256": 1, "harness_sha256": 1, "settings": 1, "split": "dev", "repeats": 1}
        report = {"metadata": meta, "server_and_weights_stable": True, "episodes": [],
                  "summary": {"operational_errors": 0}}
        raw = evaluate.canonical(report)
        (legacy / "report.json").write_bytes(raw)
        result = provenance.verify(legacy)
        self.assertEqual(result["status"], "unverified")
        self.assertIn("legacy receipt: no provenance.json", result["reasons"])
        self.assertEqual(result["operator_server_identity"], "archived-ember-cpu-server-Sfull-20261103")
        self.assertEqual((legacy / "report.json").read_bytes(), raw)
        self.assertEqual(sorted(p.name for p in legacy.iterdir()), ["report.json"])
        evaluate.compare(report, copy.deepcopy(report))  # the frozen analysis still reads it

    def test_receipt_whose_server_digest_differs_from_local_hash_is_mismatch(self):
        run = self.root / "tampered"
        run.mkdir()
        meta = {"health": {"model": "m.gguf", "provenance": ember_provenance("f" * 64)}, "weights_sha256": self.sha}
        (run / "report.json").write_bytes(evaluate.canonical({"metadata": meta, "server_and_weights_stable": True}))
        self.assertEqual(provenance.verify(run)["status"], "mismatch")

    def test_unstable_receipt_is_not_verified(self):
        run = self.root / "unstable"
        run.mkdir()
        meta = {"health": {"model": "m.gguf", "provenance": ember_provenance(self.sha)}, "weights_sha256": self.sha}
        for stable in (False, None):
            with self.subTest(stable=stable):
                report = {"metadata": meta}
                if stable is not None:
                    report["server_and_weights_stable"] = stable
                (run / "report.json").write_bytes(evaluate.canonical(report))
                self.assertEqual(provenance.verify(run)["status"], "changed")

    def test_preflight_and_receipt_disagreement_is_changed(self):
        # A server swapped between the preflight and the instrument's first /health.
        server = FakeServer(ember_provenance(self.sha))
        first = {"n": 0}

        def swap_once(url, payload=None, timeout=45):
            if url.endswith("/health"):
                first["n"] += 1
                if first["n"] == 2:
                    return {"status": "ok", "model": "/models/model.gguf", "context": 1024,
                            "provenance": ember_provenance(self.sha, instance="c" * 32)}, 0.1
            return server(url, payload, timeout)
        with patch("evaluate.curl_json", side_effect=swap_once), redirect_stdout(io.StringIO()):
            code = provenance.run("evaluate", self.args(), False)
        self.assertNotEqual(code, 0)
        self.assertEqual(json.loads((self.root / "out/provenance.json").read_text())["status"], "changed")

    def test_existing_output_directory_is_refused_before_querying(self):
        (self.root / "out").mkdir()
        server = FakeServer(ember_provenance(self.sha))
        with self.assertRaises(ValueError):
            self.run_with(server)
        self.assertEqual(server.calls, [])

    # ---- review round (Explore review, 13 Nov 2026) ----

    def test_frozen_evaluate_matches_v002_manifest(self):
        manifest = json.loads((ROOT / "evaluation_manifest_v002_final.json").read_text())
        self.assertEqual(manifest["files"]["evaluate.py"], self.frozen["evaluate.py"])

    def test_settings_that_replace_the_model_are_never_verified(self):
        side = {"sha256": "d" * 64, "bytes": 10}
        for name in ("EMBER_TEST_SCRIPTED_REPLIES", "EMBER_IR"):
            with self.subTest(setting=name):
                prov = ember_provenance(self.sha, env=[name], side_files={name: side})
                status, reasons = provenance.classify({"provenance": prov}, self.sha)
                self.assertEqual(status, "unverified")
                self.assertIn(name, reasons[0])
                out = "replaced-" + name
                with self.assertRaises(provenance.ProvenanceRefused):
                    self.run_with(FakeServer(prov), out=out)
                self.assertFalse((self.root / out).exists())
        # A setting that only tunes the computation is recorded, not refused.
        prov = ember_provenance(self.sha, env=["EMBER_KV"], settings={"EMBER_KV": "q8_0"})
        self.assertEqual(provenance.classify({"provenance": prov}, self.sha)[0], "verified")

    def test_modified_since_load_and_unknown_schema_are_not_verified(self):
        modified = ember_provenance(None, status="modified_since_load")
        status, reasons = provenance.classify({"provenance": modified}, self.sha)
        self.assertEqual(status, "unverified")
        self.assertIn("modified_since_load", reasons[0])
        other = dict(ember_provenance(self.sha), schema="ember.provenance/99")
        self.assertEqual(provenance.classify({"provenance": other}, self.sha)[0], "unverified")
        # Modified mid-run: the end /health differs from the start, and the run is changed.
        server = FakeServer(ember_provenance(self.sha), swap_after_chat=1, swapped=modified)
        self.assertEqual(self.run_with(server), 2)
        record = json.loads((self.root / "out/provenance.json").read_text())
        self.assertEqual(record["status"], "changed")
        self.assertEqual(provenance.verify(self.root / "out")["status"], "changed")

    def test_unreachable_server_is_refused_cleanly(self):
        def down(url, payload=None, timeout=45):
            raise RuntimeError("curl transport failed, code 7")
        with patch("evaluate.curl_json", side_effect=down), redirect_stderr(io.StringIO()) as err:
            self.assertEqual(provenance.main(["run", "evaluate", "--"] + self.args()), 3)
        self.assertIn("no preflight /health", err.getvalue())
        self.assertFalse((self.root / "out").exists())

    def test_no_end_observation_is_changed(self):
        server = FakeServer(ember_provenance(self.sha))
        calls = {"health": 0}

        def flaky(url, payload=None, timeout=45):
            if url.endswith("/health"):
                calls["health"] += 1
                if calls["health"] == 4:  # preflight, instrument start, instrument end, postflight
                    raise RuntimeError("curl transport failed, code 7")
            return server(url, payload, timeout)
        with patch("evaluate.curl_json", side_effect=flaky), redirect_stdout(io.StringIO()):
            self.assertEqual(provenance.run("evaluate", self.args(), False), 4)
        record = json.loads((self.root / "out/provenance.json").read_text())
        self.assertEqual(record["status"], "changed")
        self.assertIn("end_error", record)

    def _receipt(self, name, server_sha, stable=True):
        run = self.root / name
        run.mkdir()
        meta = {"health": {"model": "m.gguf", "provenance": ember_provenance(server_sha)}, "weights_sha256": self.sha}
        (run / "report.json").write_bytes(evaluate.canonical({"metadata": meta, "server_and_weights_stable": stable}))
        return run

    def test_verify_only_ever_downgrades(self):
        garbage = ('{"schema": "garbled"}', "not json", '{"schema": "tinychat-eval/provenance/1", "start": 5, '
                   '"status": "verified", "local_weights_sha256": "' + self.sha + '"}')
        for i, text in enumerate(garbage):
            with self.subTest(provenance_json=text):
                mismatch = self._receipt(f"mm{i}", "f" * 64)
                changed = self._receipt(f"ch{i}", self.sha, stable=False)
                verified = self._receipt(f"ok{i}", self.sha)
                for run in (mismatch, changed, verified):
                    (run / "provenance.json").write_text(text)
                self.assertEqual(provenance.verify(mismatch)["status"], "mismatch")
                self.assertEqual(provenance.verify(changed)["status"], "changed")
                self.assertIn(provenance.verify(verified)["status"], ("unverified", "changed"))
                with self.assertRaises(ValueError):
                    provenance.compare(mismatch, verified, allow_unverified=True, allow_engine_change=True)
        self.assertEqual(provenance.worse("mismatch", "unverified"), "mismatch")
        self.assertEqual(provenance.worse("changed", "unverified"), "changed")
        self.assertEqual(provenance.worse("verified", "unverified"), "unverified")

    def test_compare_refuses_engine_differences_unless_allowed(self):
        self.run_with(FakeServer(ember_provenance(self.sha)), out="a")
        self.run_with(FakeServer(ember_provenance(self.sha, instance="b" * 32, env=["EMBER_KV"],
                                                  settings={"EMBER_KV": "q8_0"})), out="b")
        with self.assertRaises(ValueError) as e:
            provenance.compare(self.root / "a", self.root / "b")
        self.assertIn("runtime.ember_settings", str(e.exception))
        result = provenance.compare(self.root / "a", self.root / "b", allow_engine_change=True)
        self.assertFalse(result["provenance"]["engine"]["same"])
        self.assertIn("runtime.ember_env", result["provenance"]["engine"]["differences"])
        # Same engine, a new process: only the instance id differs, which is not an engine difference.
        self.run_with(FakeServer(ember_provenance(self.sha, instance="c" * 32)), out="c")
        self.assertTrue(provenance.compare(self.root / "a", self.root / "c")["provenance"]["engine"]["same"])

    def test_compare_is_evaluate_only_with_a_clear_message(self):
        self.run_with(FakeServer(ember_provenance(self.sha)), out="a")
        sweep = self.root / "sweep"
        sweep.mkdir()
        meta = {"instrument": "context_sweep", "health": {"provenance": ember_provenance(self.sha)},
                "weights_sha256": self.sha}
        (sweep / "report.json").write_bytes(evaluate.canonical({"metadata": meta, "server_and_weights_stable": True}))
        with self.assertRaises(ValueError) as e:
            provenance.compare(self.root / "a", sweep)
        self.assertIn("evaluate.py runs only", str(e.exception))

    def test_historical_receipts_verify_read_only(self):
        """Opt-in: TCE_HISTORICAL_RUNS=dir1:dir2 (saved runs); each is classified, none is modified."""
        dirs = [d for d in os.environ.get("TCE_HISTORICAL_RUNS", "").split(os.pathsep) if d]
        if not dirs:
            self.skipTest("TCE_HISTORICAL_RUNS not set")
        for d in dirs:
            with self.subTest(run=d):
                before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(d).iterdir() if p.is_file()}
                result = provenance.verify(d)
                self.assertIn(result["status"], ("unverified", "verified", "mismatch", "changed"))
                if "provenance" not in json.loads((Path(d) / "report.json").read_text())["metadata"].get("health", {}):
                    self.assertEqual(result["status"], "unverified")
                after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(d).iterdir() if p.is_file()}
                self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
