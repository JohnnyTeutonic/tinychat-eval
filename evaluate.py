"""Reproducible curl-based tiny-chat evaluation. No cloud calls by default."""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time
import urllib.parse

SETTINGS = {"temperature": 0, "top_k": 1, "top_p": 1, "seed": 123, "max_tokens": 32, "stream": False}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def normalise(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def score(case, text):
    kind = case["check"]["kind"]
    if kind == "unscored":
        return None
    if kind == "exact":
        return normalise(text) in [normalise(x) for x in case["check"]["answers"]]
    raise ValueError("unregistered scoring contract")


def semantic_content(case, text):
    """Lexical content diagnostic, not semantic truth or a release pass."""
    if case["check"]["kind"] != "exact": return None
    words = " " + normalise(text) + " "
    found = [a for a in case["check"]["answers"] if " " + normalise(a) + " " in words]
    negated = bool(re.search(r"\b(no|not|never|isnt|isn't|dont|don't)\b", text.casefold()))
    return {"answer_mentioned": bool(found), "negation_flag": negated,
            "interpretation": "CONTENT_MENTION_ONLY_NOT_CORRECTNESS"}


def curl_json(url, payload=None, timeout=45):
    args = ["curl", "--silent", "--show-error", "--max-time", str(timeout),
            "--write-out", "\n%{http_code}", url]
    raw = None
    if payload is not None:
        args += ["--header", "Content-Type: application/json", "--data-binary", "@-"]
        raw = canonical(payload)
    started = time.perf_counter()
    result = subprocess.run(args, input=raw, capture_output=True, timeout=timeout + 5)
    elapsed = (time.perf_counter() - started) * 1000
    if result.returncode:
        raise RuntimeError("curl transport failed, code " + str(result.returncode))
    body, status = result.stdout.rsplit(b"\n", 1)
    if int(status) != 200:
        raise RuntimeError("HTTP " + status.decode())
    return json.loads(body), elapsed


def summarise(rows):
    groups = collections.defaultdict(list)
    complete = [r for r in rows if r["complete"]]
    for row in complete:
        if row["exact_pass"] is not None:
            groups[row["family"]].append(row["exact_pass"])
    repeats = collections.defaultdict(list)
    for row in rows:
        repeats[row["case_id"]].append(row)
    stability = []
    for group in repeats.values():
        if len(group) > 1 and all(r["complete"] for r in group):
            stability.append(len({canonical(r["replies"]) for r in group}) == 1)
    return {"attempted_episodes": len(rows), "complete_episodes": len(complete),
            "operational_errors": len(rows)-len(complete),
            "exact_checks": {k: {"passed": sum(v), "n": len(v), "rate": sum(v)/len(v)} for k,v in groups.items()},
            "repeat_identical": {"passed": sum(stability), "n": len(stability)},
            "median_request_ms": statistics.median([t["wall_ms"] for r in rows for t in r["turns"]]) if any(r["turns"] for r in rows) else None,
            "coherence": "UNJUDGED", "fluency": "UNJUDGED"}


def run(args):
    parsed = urllib.parse.urlparse(args.url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("local HTTP server required; remote access needs a separately designed adapter")
    suite_path = Path(args.suite)
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes)
    if args.split == "holdout" and not args.allow_holdout:
        raise ValueError("holdout requires --allow-holdout; reserve for release decisions")
    cases = [c for c in suite["cases"] if c["split"] == args.split]
    if not cases or args.repeats < 1 or args.repeats > 5:
        raise ValueError("nonempty split and 1-5 repeats required")
    if len({c["id"] for c in suite["cases"]}) != len(suite["cases"]):
        raise ValueError("duplicate case identity")
    health, _ = curl_json(args.url.rstrip("/")+"/health")
    models, _ = curl_json(args.url.rstrip("/")+"/v1/models")
    model = models["data"][0]["id"]
    weights = Path(args.weights)
    weight_sha = digest(weights.read_bytes())
    if Path(health["model"].replace("\\", "/")).name != weights.name:
        raise ValueError("server model filename differs from supplied provenance")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    metadata = {"suite_sha256": digest(suite_bytes), "harness_sha256": digest(Path(__file__).read_bytes()),
                "weights_sha256": weight_sha, "health": health, "model_id": model,
                "settings": SETTINGS, "split": args.split, "repeats": args.repeats,
                "started_at": started, "candidate": args.candidate, "endpoint": args.url,
                "server_identity": args.server_identity, "history_mode": getattr(args,"history_mode","generated"),
                "limitation": "Model filename and local artifact hash do not prove loaded server bytes; server identity is operator supplied."}
    (out/"metadata.json").write_bytes(canonical(metadata))
    rows = []
    with (out/"episodes.jsonl").open("x", encoding="utf-8") as journal:
        for case in cases:
            for repeat in range(args.repeats):
                messages = []
                row = {"case_id":case["id"], "family":case["family"], "repeat":repeat,
                       "complete":False, "turns":[], "replies":[], "exact_pass":None}
                try:
                    for user in case["users"]:
                        messages.append({"role":"user", "content":user})
                        request = dict(SETTINGS, model=model, messages=list(messages))
                        reply, elapsed = curl_json(args.url.rstrip("/")+"/v1/chat/completions", request)
                        row["turns"].append({"request":request, "response":reply, "wall_ms":elapsed})
                        if not isinstance(reply, dict) or not isinstance(reply.get("choices"), list) or not reply["choices"]:
                            raise ValueError("malformed choices in model reply")
                        choice = reply["choices"][0]
                        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                            raise ValueError("malformed message in model reply")
                        text = choice["message"].get("content")
                        if not isinstance(text, str): raise ValueError("nontext model reply")
                        row["replies"].append(text)
                        messages.append({"role":"assistant", "content":text if metadata["history_mode"]=="generated" else "Noted."})
                    row["exact_pass"] = score(case, row["replies"][-1])
                    row["semantic_content"] = semantic_content(case, row["replies"][-1])
                    row["complete"] = True
                except (RuntimeError, ValueError, KeyError, IndexError, subprocess.TimeoutExpired) as error:
                    row["error"] = type(error).__name__ + ": " + str(error)
                journal.write(json.dumps(row,ensure_ascii=False)+"\n")
                journal.flush()
                rows.append(row)
    final_error = None
    try:
        final_health, _ = curl_json(args.url.rstrip("/")+"/health")
        unchanged = final_health == health and digest(weights.read_bytes()) == weight_sha
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        unchanged = False
        final_error = type(error).__name__ + ": " + str(error)
    report = {"metadata":metadata,"server_and_weights_stable":unchanged,"episodes":rows,"summary":summarise(rows)}
    if final_error:
        report["final_observation_error"] = final_error
    (out/"report.json").write_bytes(canonical(report))
    print(json.dumps(report["summary"],indent=2))
    if not unchanged or report["summary"]["operational_errors"]:
        raise SystemExit(2)


def compare(a,b):
    for key in ("suite_sha256","harness_sha256","settings","split","repeats","history_mode"):
        if a["metadata"].get(key) != b["metadata"].get(key): raise ValueError("incompatible comparison: "+key)
    if not all(x["server_and_weights_stable"] and not x["summary"]["operational_errors"] for x in (a,b)):
        raise ValueError("incomplete or changing-server comparison")
    if [(r["case_id"],r["repeat"]) for r in a["episodes"]] != [(r["case_id"],r["repeat"]) for r in b["episodes"]]:
        raise ValueError("episode inventory differs")
    return {"before":a["summary"],"after":b["summary"],"note":"Descriptive paired comparison; this small pilot establishes no general superiority or statistical significance."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    r=sub.add_parser("run")
    r.add_argument("--url",default="http://127.0.0.1:8081")
    r.add_argument("--suite",default=str(Path(__file__).with_name("suite.json")))
    r.add_argument("--split",choices=["dev","holdout"],default="dev")
    r.add_argument("--allow-holdout",action="store_true")
    r.add_argument("--history-mode",choices=["generated","controlled"],default="generated",help="controlled replaces preceding assistant outputs with frozen Noted.; diagnostic only")
    r.add_argument("--repeats",type=int,default=2)
    r.add_argument("--weights",required=True)
    r.add_argument("--server-identity",required=True)
    r.add_argument("--candidate",required=True)
    r.add_argument("--out",required=True)
    c=sub.add_parser("compare")
    c.add_argument("before");c.add_argument("after")
    a=p.parse_args()
    if a.command=="run":run(a)
    else: print(json.dumps(compare(json.loads(Path(a.before).read_text()),json.loads(Path(a.after).read_text())),indent=2))

if __name__=="__main__":main()
