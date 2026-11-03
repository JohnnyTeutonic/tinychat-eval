"""Bounded retrieval phase: questions about a small fictional technical manual, with cited evidence and an 'unknown'
option (README, "First optimise reliable short conversation. Then select one bounded useful job").

Three conditions, scored separately, never averaged:
  raw        the question alone. The device is fictional, so 'unknown' is the only correct reply to every question;
             answerable-question "passes" here would be chance, and confident answers are recorded as invention.
  oracle     the gold evidence sections (for absent-answer questions, hand-picked near-miss sections), isolating
             reading and citing from retrieval.
  retrieved  the top-k sections from a deterministic stdlib BM25 retriever over the manual.

Measured: retrieval recall@k (offline, no server), exact answer, citation support (at least one citation, all of them
gold, in a reply that is not an excerpt echo), supported answer (both), refusal when the answer is absent, and false
refusal when it is present. An excerpt echo is a reply that starts by copying an excerpt's header line ("[S2] Power:");
it cites by copying, so it never counts as citation support, and echoes are counted separately. Exact
contracts as in evaluate.py; one greedy request per case and condition (greedy repeats are identical by construction).
RAG can supply facts; it cannot establish conversational memory, and this is not a release gate.

  python rag_eval.py retrieval [--k 2] [--split all]
  python rag_eval.py run --weights MODEL.gguf --server-identity ID --candidate NAME --out runs/NAME-rag [--split dev]
"""
import argparse
import collections
import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import urllib.parse

import evaluate

MANUAL = Path(__file__).with_name("manual_v001.json")
CONDITIONS = ("raw", "oracle", "retrieved")
STOPWORDS = set("a an and are at be by do does for from how in is it its long many much of on or should the this to "
                "unit what when which who why with br 40".split())
CITATION = re.compile(r"\[(S\d+)\]")
ECHO = re.compile(r"^\s*\[S\d+\]\s*(?:" + "|".join(re.escape(s["title"]) for s in
                  json.loads(MANUAL.read_bytes())["sections"]) + r")\s*:")
LENIENT_REFUSAL = re.compile(r"\b(unknown|not known|do not know|don't know|dont know|not say|not stated|no information)\b")


def load(path=MANUAL):
    data = path.read_bytes()
    manual = json.loads(data)
    ids = [s["id"] for s in manual["sections"]]
    if len(set(ids)) != len(ids) or len({c["id"] for c in manual["cases"]}) != len(manual["cases"]):
        raise ValueError("duplicate section or case identity")
    for c in manual["cases"]:
        for s in c["evidence"] + c.get("near", []):
            if s not in ids:
                raise ValueError(f"case {c['id']} cites missing section {s}")
        if bool(c["evidence"]) == (c["answers"] == ["unknown"]):
            raise ValueError(f"case {c['id']}: absent-answer cases have no evidence and answer 'unknown'")
    return manual, data


def terms(text):
    return [t for t in evaluate.normalise(text).split() if t not in STOPWORDS]


def retrieve(manual, question, k):
    """Okapi BM25 (k1 1.2, b 0.75) over title plus text; ties keep manual order."""
    docs = [terms(s["title"] + " " + s["text"]) for s in manual["sections"]]
    avg = sum(map(len, docs)) / len(docs)
    df = collections.Counter(t for d in docs for t in set(d))
    n = len(docs)
    scores = []
    for i, d in enumerate(docs):
        tf = collections.Counter(d)
        s = 0.0
        for t in set(terms(question)):
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        scores.append((-s, i))
    return [manual["sections"][i]["id"] for _, i in sorted(scores)[:k]]


def context_for(manual, case, condition, k):
    if condition == "raw":
        return []
    if condition == "oracle":
        return case["evidence"] or case["near"]
    return retrieve(manual, case["question"], k)


def prompt(manual, case, sections):
    if not sections:
        return (f"Question about the {manual['device']}: {case['question']}\n"
                "Reply with just the answer. If you do not know, reply: unknown")
    by_id = {s["id"]: s for s in manual["sections"]}
    excerpts = "\n".join(f"[{i}] {by_id[i]['title']}: {by_id[i]['text']}" for i in sections)
    return (f"Excerpts from the {manual['device']} manual:\n{excerpts}\n\nQuestion: {case['question']}\n"
            "Answer from the excerpts only. Reply with just the answer and the section it came from, "
            "for example: 5 metres [S99]. If the excerpts do not say, reply: unknown")


def score_reply(case, text):
    cites = CITATION.findall(text)
    body = evaluate.normalise(CITATION.sub(" ", text))
    refused = body == "unknown"
    echo = bool(ECHO.match(text))
    out = {"citations": cites, "refused": refused, "excerpt_echo": echo,
           "lenient_refusal": bool(LENIENT_REFUSAL.search(text.casefold()))}
    if case["evidence"]:
        exact = body in [evaluate.normalise(a) for a in case["answers"]]
        cited_gold = bool(cites) and set(cites) <= set(case["evidence"]) and not echo
        out.update(answerable=True, answer_exact=exact, citation_supported=cited_gold,
                   supported_answer=exact and cited_gold, false_refusal=refused,
                   semantic_content=evaluate.semantic_content({"check": {"kind": "exact", "answers": case["answers"]}},
                                                              CITATION.sub(" ", text)))
    else:
        out.update(answerable=False, correct_refusal=refused)
    return out


def tally(rows, key, answerable):
    pool = [r for r in rows if r["complete"] and r["answerable"] is answerable]
    return {"passed": sum(bool(r[key]) for r in pool), "n": len(pool),
            "text": f"{sum(bool(r[key]) for r in pool)} of {len(pool)}"}


def summarise(rows):
    out = {}
    for condition in CONDITIONS:
        rs = [r for r in rows if r["condition"] == condition]
        if not rs:
            continue
        out[condition] = {"answer_exact": tally(rs, "answer_exact", True),
                          "citation_supported": tally(rs, "citation_supported", True),
                          "supported_answer": tally(rs, "supported_answer", True),
                          "false_refusal": tally(rs, "false_refusal", True),
                          "excerpt_echo": {"passed": sum(bool(r.get("excerpt_echo")) for r in rs if r["complete"]),
                                           "n": sum(r["complete"] for r in rs),
                                           "text": f"{sum(bool(r.get('excerpt_echo')) for r in rs if r['complete'])}"
                                                   f" of {sum(r['complete'] for r in rs)}"},
                          "correct_refusal_when_absent": tally(rs, "correct_refusal", False),
                          "operational_errors": sum(not r["complete"] for r in rs)}
    return out


def retrieval_report(manual, split, k):
    cases = [c for c in manual["cases"] if c["evidence"] and (split == "all" or c["split"] == split)]
    rows = [{"case_id": c["id"], "evidence": c["evidence"], "retrieved": retrieve(manual, c["question"], k)}
            for c in cases]
    for r in rows:
        r["recalled"] = set(r["evidence"]) <= set(r["retrieved"])
    return {"k": k, "split": split, "recall": f"{sum(r['recalled'] for r in rows)} of {len(rows)}", "cases": rows}


def run(args):
    parsed = urllib.parse.urlparse(args.url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("local HTTP server required")
    if args.split == "holdout" and not args.allow_holdout:
        raise ValueError("holdout requires --allow-holdout; reserve for release decisions")
    manual, manual_bytes = load()
    cases = [c for c in manual["cases"] if c["split"] == args.split]
    health, _ = evaluate.curl_json(args.url.rstrip("/") + "/health")
    models, _ = evaluate.curl_json(args.url.rstrip("/") + "/v1/models")
    model = models["data"][0]["id"]
    weights = Path(args.weights)
    weight_sha = evaluate.digest(weights.read_bytes())
    if Path(health["model"].replace("\\", "/")).name != weights.name:
        raise ValueError("server model filename differs from supplied provenance")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"instrument": "rag_eval", "manual_sha256": evaluate.digest(manual_bytes),
                "harness_sha256": evaluate.digest(Path(__file__).read_bytes()),
                "evaluate_sha256": evaluate.digest(Path(evaluate.__file__).read_bytes()),
                "weights_sha256": weight_sha, "health": health, "model_id": model, "settings": evaluate.SETTINGS,
                "split": args.split, "k": args.k, "conditions": list(CONDITIONS), "candidate": args.candidate,
                "server_identity": args.server_identity, "endpoint": args.url,
                "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "limitation": "Diagnostic phase instrument, not a release gate. Exact contracts and string citation "
                              "checks are not semantic correctness."}
    (out / "metadata.json").write_bytes(evaluate.canonical(metadata))
    rows = []
    with (out / "episodes.jsonl").open("x", encoding="utf-8") as journal:
        for case in cases:
            for condition in CONDITIONS:
                sections = context_for(manual, case, condition, args.k)
                request = dict(evaluate.SETTINGS, model=model,
                               messages=[{"role": "user", "content": prompt(manual, case, sections)}])
                row = {"case_id": case["id"], "condition": condition, "sections": sections,
                       "answerable": bool(case["evidence"]), "complete": False}
                try:
                    reply, elapsed = evaluate.curl_json(args.url.rstrip("/") + "/v1/chat/completions", request)
                    row.update(request=request, response=reply, wall_ms=elapsed)
                    text = reply["choices"][0]["message"]["content"]
                    if not isinstance(text, str):
                        raise ValueError("nontext model reply")
                    row.update(reply=text, **score_reply(case, text), complete=True)
                except (RuntimeError, ValueError, KeyError, IndexError, TypeError, subprocess.TimeoutExpired) as error:
                    row["error"] = type(error).__name__ + ": " + str(error)
                journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                journal.flush()
                rows.append(row)
    try:
        final_health, _ = evaluate.curl_json(args.url.rstrip("/") + "/health")
        stable = final_health == health and evaluate.digest(weights.read_bytes()) == weight_sha
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired):
        stable = False
    summary = summarise(rows)
    report = {"metadata": metadata, "server_and_weights_stable": stable,
              "retrieval": retrieval_report(manual, args.split, args.k), "summary": summary, "episodes": rows}
    (out / "report.json").write_bytes(evaluate.canonical(report))
    print(json.dumps({"retrieval_recall": report["retrieval"]["recall"],
                      **{c: {k: v["text"] if isinstance(v, dict) else v for k, v in s.items()}
                         for c, s in summary.items()}}, indent=2))
    if not stable or any(not r["complete"] for r in rows):
        raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("retrieval")
    q.add_argument("--k", type=int, default=2)
    q.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8081")
    r.add_argument("--split", choices=["dev", "holdout"], default="dev")
    r.add_argument("--allow-holdout", action="store_true")
    r.add_argument("--k", type=int, default=2)
    r.add_argument("--weights", required=True)
    r.add_argument("--server-identity", required=True)
    r.add_argument("--candidate", required=True)
    r.add_argument("--out", required=True)
    a = p.parse_args()
    if a.command == "retrieval":
        rep = retrieval_report(load()[0], a.split, a.k)
        print(json.dumps({"k": rep["k"], "split": rep["split"], "recall": rep["recall"],
                          "missed": [(c["case_id"], c["evidence"], c["retrieved"]) for c in rep["cases"]
                                     if not c["recalled"]]}, indent=2))
    else:
        run(a)


if __name__ == "__main__":
    main()
