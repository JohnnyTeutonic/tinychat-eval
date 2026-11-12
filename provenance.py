"""Server provenance for evaluation runs: does the live server serve the file the receipt names?

    python provenance.py run evaluate [--allow-unverified] -- --weights M.gguf --server-identity ID --candidate C --out runs/X
    python provenance.py run context_sweep -- ...      (or rag_eval; the instrument's own run arguments after --)
    python provenance.py verify runs/X [runs/Y ...]
    python provenance.py compare runs/BEFORE runs/AFTER [--allow-unverified] [--allow-engine-change]

The instruments (evaluate.py, context_sweep.py, rag_eval.py) hash the local weights file and check the
model *filename* the server reports. They cannot tell what the server loaded. An ember server with
load-time provenance reports in GET /health (ember model_provenance.h, docs/USAGE.md "Provenance"): the
SHA-256 of its .gguf as hashed at start, checked against the loader's descriptor after the load and, on
POSIX, re-checked at every /health ("modified_since_load" once the mapped file changes); its build (git
commit, dirty state, backend, compiler) and executable SHA-256; its EMBER_* settings (values, side-file
digests, other names); and an id new to each process. This module uses that report without changing the
frozen instruments: their bytes, and so their harness hashes and the v0.0.2 manifest, stay as they were.

Statuses, from best to worst: verified, unverified, changed, mismatch. A check can only move a run down
this list, never up.
  verified    the server reports a well-formed digest equal to the local file's, with no setting that
              replaces the model's computation (EMBER_TEST_SCRIPTED_REPLIES, EMBER_IR)
  unverified  no usable digest (an old server, a disabled digest, an unsupported format, a malformed or
              unknown schema, a file modified since load) or such a setting
  changed     the server or the receipt changed between observations (restart, swap, unstable run)
  mismatch    the server's digest differs from the local file's

run      queries /health before the instrument starts and refuses, writing nothing (exit 3), on a mismatch, on
         an unreachable server, and on an unverified server unless --allow-unverified. It runs the unchanged
         instrument, queries /health again at the end, and writes RUN/provenance.json (exclusive create).
verify   classifies saved runs from report.json alone (the instrument saves its first /health and compares the
         final one with it), plus provenance.json when present. Receipts made before servers reported a
         digest load unchanged and are classified "unverified", never "verified".
compare  evaluate.py's compare (evaluate runs only), refused when either run is "changed" or "mismatch", when
         either is "unverified" unless --allow-unverified, and when the two servers' build, executable or
         runtime settings differ unless --allow-engine-change; the statuses and differences are added.

Trust: the digest is the server's own report. This guards against operator error (the wrong file, a stale
server, a server restarted on other weights, settings or another build), not against a server built to lie.
"""
import argparse
import datetime
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

import evaluate

SCHEMA = "tinychat-eval/provenance/1"
SERVER_SCHEMA = "ember.provenance/1"
INSTRUMENTS = ("evaluate", "context_sweep", "rag_eval")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RANK = {"verified": 0, "unverified": 1, "changed": 2, "mismatch": 3}
# EMBER_* settings under which the replies are not the weights' own computation.
REPLACING_SETTINGS = ("EMBER_TEST_SCRIPTED_REPLIES", "EMBER_IR")
TRUST = ("Digest reported by the server itself: guards against operator error and stale or swapped servers, "
         "not against a server that misreports.")


class ProvenanceRefused(Exception):
    """The server's identity does not support the run (raised before the instrument writes anything)."""


def worse(status, other):
    """The worse of two statuses: checks only ever move a run down (verified > unverified > changed > mismatch)."""
    return status if RANK.get(status, 1) >= RANK.get(other, 1) else other


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def server_provenance(health):
    """The server's provenance object from a /health body, or None (an old server reports none)."""
    prov = health.get("provenance") if isinstance(health, dict) else None
    return prov if isinstance(prov, dict) else None


def classify(health, local_sha256):
    """(status, reasons) for one /health against the local weights digest."""
    prov = server_provenance(health)
    if prov is None:
        return "unverified", ["server reported no provenance (a server older than load-time digests)"]
    if prov.get("schema") != SERVER_SCHEMA:
        return "unverified", ["unknown server provenance schema " + json.dumps(prov.get("schema"))]
    weights = prov.get("weights") if isinstance(prov.get("weights"), dict) else {}
    digest = weights.get("sha256")
    if weights.get("status") != "sha256":
        return "unverified", ["server weights digest status is " + json.dumps(weights.get("status"))]
    if not isinstance(digest, str) or not HEX64.match(digest):
        return "unverified", ["server weights digest is not 64 lowercase hex digits"]
    if not isinstance(local_sha256, str) or not HEX64.match(local_sha256):
        return "unverified", ["local weights digest missing or malformed"]
    if digest != local_sha256:
        return "mismatch", ["server loaded weights " + digest + ", local file is " + local_sha256]
    runtime = prov.get("runtime") if isinstance(prov.get("runtime"), dict) else {}
    names = runtime.get("ember_env") if isinstance(runtime.get("ember_env"), list) else []
    replacing = [n for n in REPLACING_SETTINGS if n in names]
    if replacing:
        return "unverified", ["the server replaces the model's computation: " + ", ".join(replacing) + " set"]
    if prov.get("public") is True:
        # EMBER_PUBLIC=1 hides the EMBER_* record (ember refuses scripted replies there; EMBER_IR is not visible).
        return "verified", ["public server: weights digest verified; runtime settings not reported"]
    return "verified", []


def engine_identity(health):
    """What must match between two runs besides the weights: build, executable and runtime settings."""
    prov = server_provenance(health)
    if prov is None:
        return None
    runtime = prov.get("runtime") if isinstance(prov.get("runtime"), dict) else {}
    build = dict(prov.get("build") or {})
    if prov.get("public") is True:  # a public server reports no compiler and no EMBER_* record
        build.pop("compiler", None)
        return {"build": build, "executable_sha256": (prov.get("executable") or {}).get("sha256"),
                "runtime": None}
    return {"build": build, "executable_sha256": (prov.get("executable") or {}).get("sha256"),
            "runtime": {k: runtime.get(k) for k in ("gpu_layers_requested", "ember_env", "ember_settings",
                                                    "ember_side_files")}}


def _instrument_args(argv):
    """--url, --weights and --out from the instrument's own run arguments (defaults as the instruments)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--url", default="http://127.0.0.1:8081")
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    known, _ = p.parse_known_args(argv)
    return known


def _observe(url):
    health, _ = evaluate.curl_json(url.rstrip("/") + "/health")
    return {"observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "provenance": server_provenance(health), "health": health}


def run(instrument, argv, allow_unverified=False):
    """Preflight, the unchanged instrument, postflight; returns the process exit code."""
    if instrument not in INSTRUMENTS:
        raise ValueError("unknown instrument " + instrument)
    a = _instrument_args(argv)
    out = Path(a.out)
    if out.exists():
        raise ValueError("output directory exists: " + str(out))
    local = file_sha256(a.weights)
    try:
        start = _observe(a.url)
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as e:
        raise ProvenanceRefused("no preflight /health: " + type(e).__name__ + ": " + str(e)) from e
    status, reasons = classify(start["health"], local)
    if status == "mismatch":
        raise ProvenanceRefused("; ".join(reasons))
    if status == "unverified" and not allow_unverified:
        raise ProvenanceRefused("; ".join(reasons) + " (pass --allow-unverified to record an unverified run)")
    module = importlib.import_module(instrument)
    saved, code, error = sys.argv, 0, None
    sys.argv = [module.__file__, "run"] + list(argv)
    try:
        module.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except Exception as e:  # recorded below when the run directory exists, then raised
        error = e
    finally:
        sys.argv = saved
    if not out.is_dir():
        if error:
            raise error
        return code or 1
    record = {"schema": SCHEMA, "instrument": instrument, "instrument_argv": list(argv),
              "instrument_sha256": evaluate.digest(Path(module.__file__).read_bytes()),
              "provenance_sha256": evaluate.digest(Path(__file__).read_bytes()),
              "local_weights_sha256": local, "allow_unverified": allow_unverified,
              "start": {k: start[k] for k in ("observed_at", "provenance")},
              "instrument_exit": code, "trust": TRUST}
    if error:
        record["instrument_error"] = type(error).__name__ + ": " + str(error)
    try:
        end = _observe(a.url)
        record["end"] = {k: end[k] for k in ("observed_at", "provenance")}
        if end["provenance"] != start["provenance"]:
            status = worse(status, "changed")
            reasons = reasons + ["server provenance changed between start and end of the run"]
            end_status, end_reasons = classify(end["health"], local)
            if end_status != "verified":
                status = worse(status, end_status)
                reasons += ["at the end: " + x for x in end_reasons]
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as e:
        record["end_error"] = type(e).__name__ + ": " + str(e)
        status, reasons = worse(status, "changed"), reasons + ["no end observation: " + record["end_error"]]
    metadata_path = out / "metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if server_provenance(meta.get("health")) != start["provenance"]:
            status = worse(status, "changed")
            reasons = reasons + ["the instrument's first /health differs from the preflight"]
        if meta.get("weights_sha256") != local:
            status = worse(status, "changed")
            reasons = reasons + ["the instrument hashed different local weights"]
    record["status"], record["reasons"] = status, reasons
    with (out / "provenance.json").open("x", encoding="utf-8") as f:
        f.write(evaluate.canonical(record).decode())
    print(json.dumps({"provenance": status, "reasons": reasons}, indent=2))
    if error:
        raise error
    if code:
        return code
    return 0 if status in ("verified", "unverified") else 4


def verify(run_dir):
    """{"status", "reasons", ...} for a saved run. Reads only; old receipts are classified, never rewritten.

    The receipt's own classification comes first; every later check can only make it worse."""
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    meta = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    status, reasons = classify(meta.get("health"), meta.get("weights_sha256"))
    stable = report.get("server_and_weights_stable")
    if stable is not True:
        status = worse(status, "changed")
        reasons = reasons + ["the instrument did not record a stable server (server_and_weights_stable is "
                             + json.dumps(stable) + ")"]
    record_path = run_dir / "provenance.json"
    record = None
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except ValueError:
            record = {"schema": None}
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            status = worse(status, "unverified")
            reasons = reasons + ["provenance.json has an unknown schema or is unreadable"]
        else:
            start = record.get("start") if isinstance(record.get("start"), dict) else {}
            if record.get("local_weights_sha256") != meta.get("weights_sha256"):
                status = worse(status, "changed")
                reasons = reasons + ["provenance.json names other local weights than the receipt"]
            if start.get("provenance") != server_provenance(meta.get("health")):
                status = worse(status, "changed")
                reasons = reasons + ["preflight provenance differs from the receipt's /health"]
            if record.get("status") not in RANK:
                status = worse(status, "unverified")
                reasons = reasons + ["provenance.json records an unknown status " + json.dumps(record.get("status"))]
            elif RANK[record["status"]] > RANK[status]:
                status = record["status"]
                reasons = reasons + list(record.get("reasons") or [])
    elif server_provenance(meta.get("health")) is None:
        reasons = reasons + ["legacy receipt: no provenance.json"]
    prov = server_provenance(meta.get("health")) or {}
    return {"run": str(run_dir), "status": status, "reasons": reasons,
            "server_weights_sha256": (prov.get("weights") or {}).get("sha256"),
            "local_weights_sha256": meta.get("weights_sha256"),
            "build": prov.get("build"), "instance_id": prov.get("instance_id"),
            "provenance_record": record is not None,
            "operator_server_identity": meta.get("server_identity")}


def compare(before_dir, after_dir, allow_unverified=False, allow_engine_change=False):
    reports = [json.loads((Path(d) / "report.json").read_text(encoding="utf-8")) for d in (before_dir, after_dir)]
    for d, r in zip((before_dir, after_dir), reports):
        instrument = (r.get("metadata") or {}).get("instrument", "evaluate")
        if instrument != "evaluate":
            raise ValueError("compare supports evaluate.py runs only; " + str(d) + " is a " + str(instrument) + " run")
    checks = [verify(before_dir), verify(after_dir)]
    for c in checks:
        if c["status"] in ("mismatch", "changed"):
            raise ValueError("provenance " + c["status"] + " for " + c["run"] + ": " + "; ".join(c["reasons"]))
        if c["status"] == "unverified" and not allow_unverified:
            raise ValueError("provenance unverified for " + c["run"] + ": " + "; ".join(c["reasons"]))
    engines = [engine_identity(r["metadata"].get("health")) for r in reports]
    if None in engines:
        engine = {"same": None, "differences": ["engine identity unknown for a run without server provenance"]}
    else:
        public = engines[0]["runtime"] is None or engines[1]["runtime"] is None
        if public:  # compare what both report: the build less its compiler, and the executable
            for e in engines:
                e["build"].pop("compiler", None)
        diffs = [k for k in ("build", "executable_sha256") if engines[0][k] != engines[1][k]]
        if not public:
            diffs += ["runtime." + k for k in engines[0]["runtime"]
                      if engines[0]["runtime"][k] != engines[1]["runtime"][k]]
        engine = {"same": (False if diffs else None) if public else not diffs, "differences": diffs}
        if public:
            engine["note"] = "runtime settings not compared: a public server does not report them"
        if diffs and not allow_engine_change:
            raise ValueError("the two servers differ in " + ", ".join(diffs) +
                             " (pass --allow-engine-change to compare across engines)")
    result = evaluate.compare(*reports)
    result["provenance"] = {"before": checks[0], "after": checks[1], "engine": engine}
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if argv[:1] == ["run"]:  # the instrument's own arguments follow "--", untouched
        if "--" not in argv:
            print("run needs the instrument's arguments after --", file=sys.stderr)
            return 2
        cut = argv.index("--")
        argv, rest = argv[:cut], argv[cut + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("instrument", choices=INSTRUMENTS)
    r.add_argument("--allow-unverified", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("runs", nargs="+")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--allow-unverified", action="store_true")
    c.add_argument("--allow-engine-change", action="store_true")
    a = p.parse_args(argv)
    if a.command == "run":
        try:
            return run(a.instrument, rest, a.allow_unverified)
        except ProvenanceRefused as e:
            print("refused: " + str(e), file=sys.stderr)
            return 3
    if a.command == "verify":
        results = [verify(d) for d in a.runs]
        print(json.dumps(results, indent=2))
        return 0 if all(x["status"] == "verified" for x in results) else 1
    print(json.dumps(compare(a.before, a.after, a.allow_unverified, a.allow_engine_change), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
