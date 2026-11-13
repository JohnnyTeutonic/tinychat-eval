"""CANDIDATE scorer for the candidate v0.0.3 suites: offline, reads a saved evaluate.py report and its suite.

    python score_v003.py runs/NAME/report.json --suite suite_v003_candidate_dev.json [--json]
    python score_v003.py --validate suite_v003_candidate_dev.json [suite_v003_candidate_cue.json ...]

Not frozen and not accepted: Jonathan's approval is required before any gate reads it. It changes no frozen file:
evaluate.py runs the suite unchanged and keeps its own exact score; this module adds, per distinct case,

  strict   evaluate.py's exact contract (normalised whole reply equals a gold answer), and
  content  a small per-case lexical rule (below), separating "said the right thing" from "said only that",

with a case counted once however many repeats ran (it passes only if every repeat passes), and families, twin
pairs and cue triplets as the units. Content rules are lexical: they cannot see attribution beyond the phrases a case
forbids, so transcripts remain the arbiter for any claim about meaning.

Content rule, in a case's check["content"] (every phrase is compared after evaluate.normalise):
  require_any  at least one phrase occurs as whole words
  forbid       no phrase occurs, except where "not", "no", "never", "no longer" or "instead of" stands just before it
  polarity     "negative": the reply denies (no, not, never, nothing, none, cannot, or an n't form) and forbid holds;
               "affirmative": the reply is not a denial (does not open with no or nope, and says none of
               "not told", "haven't", "have not", "didn't", "did not", "don't know", "do not know", "never told")
"""
import argparse
import hashlib
import json
from pathlib import Path

import evaluate

NEGATORS = {"no", "not", "never", "nothing", "none", "cannot", "nope", "neither", "nor"}
NT_STEMS = {"haven", "hasn", "hadn", "didn", "don", "doesn", "wasn", "weren", "isn", "aren", "can", "couldn", "won",
            "wouldn", "shouldn", "ain"}
DENIALS = ("not told", "haven t", "have not", "hasn t", "has not", "didn t", "did not", "don t know", "do not know",
           "never told")
EXCUSERS = (("not",), ("no",), ("never",), ("no", "longer"), ("instead", "of"))
FAMILY_ORDER = ("basic_qa", "immediate_recall", "delayed_recall", "correction", "told_or_not", "instruction",
                "cue_sensitivity", "open_conversation")


def tokens(text):
    return evaluate.normalise(text).split()


def find(toks, phrase):
    """Start indices where the normalised phrase occurs as whole words."""
    p = tokens(phrase)
    if not p:
        raise ValueError("empty content phrase " + repr(phrase))
    return [i for i in range(len(toks) - len(p) + 1) if toks[i:i + len(p)] == p]


def excused(toks, i):
    return any(tuple(toks[max(0, i - len(e)):i]) == e for e in EXCUSERS)


def is_denial(toks):
    if any(t in NEGATORS for t in toks):
        return True
    return any(toks[i] == "t" and toks[i - 1] in NT_STEMS for i in range(1, len(toks)))


def content_correct(case, text):
    """True/False under the case's content rule; None for a case without one (unscored)."""
    rule = case["check"].get("content")
    if rule is None:
        return None
    if not isinstance(text, str):
        return False
    toks = tokens(text)
    if not toks:
        return False
    for phrase in rule.get("forbid", []):
        if any(not excused(toks, i) for i in find(toks, phrase)):
            return False
    polarity = rule.get("polarity")
    if polarity == "negative" and not is_denial(toks):
        return False
    if polarity == "affirmative" and (toks[0] in ("no", "nope") or any(find(toks, d) for d in DENIALS)):
        return False
    required = rule.get("require_any", [])
    return not required or any(find(toks, p) for p in required)


def validate_suite(suite):
    """Raise ValueError unless the suite's cases, twins and triplets are well formed; return a summary."""
    cases = suite["cases"]
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case identity")
    for c in cases:
        check = c["check"]
        if check["kind"] == "unscored":
            continue
        if check["kind"] != "exact":
            raise ValueError(c["id"] + ": unregistered check kind")
        answers = check.get("answers")
        if not isinstance(answers, list) or not answers:
            raise ValueError(c["id"] + ": answers must be a non-empty list")
        for a in answers:
            if not isinstance(a, str) or not a.isascii() or not evaluate.normalise(a):
                raise ValueError(c["id"] + ": gold must be ASCII and normalise to something: " + repr(a))
        rule = check.get("content")
        if not isinstance(rule, dict):
            raise ValueError(c["id"] + ": a scored case needs a content rule")
        if set(rule) - {"require_any", "forbid", "polarity"} or rule.get("polarity") not in (None, "negative",
                                                                                               "affirmative"):
            raise ValueError(c["id"] + ": unknown content rule key or polarity")
        for p in rule.get("require_any", []) + rule.get("forbid", []):
            if not isinstance(p, str) or not tokens(p):
                raise ValueError(c["id"] + ": empty or non-text content phrase " + repr(p))
        for a in answers:
            if not content_correct(c, a):
                raise ValueError(c["id"] + ": its own gold " + repr(a) + " fails its content rule")
    pairs, triplets = {}, {}
    for c in cases:
        if "twin" in c:
            pairs.setdefault(c["twin"]["pair"], []).append(c)
        if "cue" in c:
            triplets.setdefault(c["cue"]["triplet"], {})[c["cue"]["role"]] = c
    for name, members in pairs.items():
        if len(members) != 2 or {m["twin"]["told"] for m in members} != {True, False}:
            raise ValueError("twin pair " + name + " needs one told and one not-told case")
        told, untold = sorted(members, key=lambda m: not m["twin"]["told"])
        if told["users"][-1] != untold["users"][-1] or len(told["users"]) != len(untold["users"]):
            raise ValueError("twin pair " + name + ": the twins must ask the same final question at the same turn")
        if sum(a != b for a, b in zip(told["users"], untold["users"])) != 1:
            raise ValueError("twin pair " + name + ": the twins must differ in exactly one user message")
        for one, other in ((told, untold), (untold, told)):
            if any(content_correct(one, a) or evaluate.score(one, a) for a in other["check"]["answers"]):
                raise ValueError("twin pair " + name + ": " + other["id"] + "'s gold also passes " + one["id"])
    for name, roles in triplets.items():
        if set(roles) != {"A", "B", "N"}:
            raise ValueError("cue triplet " + name + " needs roles A, B and N")
        a, b, n = roles["A"], roles["B"], roles["N"]
        for x in (b, n):
            if len(x["users"]) != len(a["users"]) or sum(p != q for p, q in zip(a["users"], x["users"])) != 1:
                raise ValueError("cue triplet " + name + ": " + x["id"] + " must differ from A in one user message")
        if sorted(map(evaluate.normalise, n["check"]["answers"])) != sorted(map(evaluate.normalise, a["check"]["answers"])):
            raise ValueError("cue triplet " + name + ": the nuisance twin must keep A's gold")
        if any(content_correct(a, g) or evaluate.score(a, g) for g in b["check"]["answers"]) or \
                any(content_correct(b, g) or evaluate.score(b, g) for g in a["check"]["answers"]):
            raise ValueError("cue triplet " + name + ": the decisive twin's gold must differ from A's")
        if not all(x.get("diagnostic") is True for x in roles.values()):
            raise ValueError("cue triplet " + name + ": cue cases are diagnostic")
    families = {}
    for c in cases:
        families.setdefault(c["family"], {"cases": 0, "scored": 0})
        families[c["family"]]["cases"] += 1
        families[c["family"]]["scored"] += c["check"]["kind"] == "exact"
    return {"cases": len(cases), "families": families, "twin_pairs": len(pairs), "cue_triplets": len(triplets)}


def score_report(report, suite):
    """Per distinct case, family, twin pair and cue triplet: strict and content, each 'k of n'."""
    validate_suite(suite)
    cases = {c["id"]: c for c in suite["cases"]}
    replies = {}
    for e in report["episodes"]:
        if e["case_id"] not in cases:
            raise ValueError("report case " + e["case_id"] + " is not in the suite")
        last = e["replies"][-1] if e.get("complete") and e.get("replies") else None
        replies.setdefault(e["case_id"], []).append(last)
    per_case = {}
    for cid, texts in replies.items():
        case = cases[cid]
        if case["check"]["kind"] == "unscored":
            per_case[cid] = {"family": case["family"], "scored": False, "repeats": len(texts)}
            continue
        complete = all(t is not None for t in texts)
        strict = [t is not None and evaluate.score(case, t) for t in texts]
        content = [t is not None and content_correct(case, t) for t in texts]
        per_case[cid] = {"family": case["family"], "scored": True, "repeats": len(texts), "complete": complete,
                         "strict": complete and all(strict), "content": complete and all(content),
                         "repeats_agree": len(set(texts)) == 1, "diagnostic": bool(case.get("diagnostic"))}
    families = {}
    for cid, r in per_case.items():
        if not r["scored"]:
            continue
        f = families.setdefault(r["family"], {"n": 0, "strict": 0, "content": 0, "diagnostic": r["diagnostic"],
                                              "content_not_strict": [], "strict_not_content": []})
        f["n"] += 1
        f["strict"] += r["strict"]
        f["content"] += r["content"]
        if r["content"] and not r["strict"]:
            f["content_not_strict"].append(cid)
        if r["strict"] and not r["content"]:
            f["strict_not_content"].append(cid)
    for f in families.values():
        f["strict_text"] = f"{f['strict']} of {f['n']}"
        f["content_text"] = f"{f['content']} of {f['n']}"
    pairs = {}
    for cid, case in cases.items():
        if "twin" in case and cid in per_case:
            p = pairs.setdefault(case["twin"]["pair"], {})
            p["told" if case["twin"]["told"] else "untold"] = per_case[cid]
    twin = {"pairs": len(pairs),
            "both_strict": sum(1 for p in pairs.values() if len(p) == 2 and p["told"]["strict"] and p["untold"]["strict"]),
            "both_content": sum(1 for p in pairs.values() if len(p) == 2 and p["told"]["content"] and p["untold"]["content"]),
            "note": "a constant yes or a constant no is right on exactly one twin of every pair"}
    triplets = {}
    for cid, case in cases.items():
        if "cue" in case and cid in per_case:
            triplets.setdefault(case["cue"]["triplet"], {})[case["cue"]["role"]] = per_case[cid]
    cue = {}
    for name, t in sorted(triplets.items()):
        if set(t) != {"A", "B", "N"}:
            continue
        cue[name] = {"A": t["A"]["content"], "B_followed_decisive_edit": t["B"]["content"],
                     "N_ignored_nuisance_edit": t["N"]["content"],
                     "selective": t["A"]["content"] and t["B"]["content"] and t["N"]["content"],
                     "selective_strict": t["A"]["strict"] and t["B"]["strict"] and t["N"]["strict"]}
    cue_summary = {"triplets": len(cue), "selective": sum(v["selective"] for v in cue.values()),
                   "selective_strict": sum(v["selective_strict"] for v in cue.values()),
                   "role": "diagnostic only, never a release criterion"} if cue else None
    ordered = {k: families[k] for k in sorted(families, key=lambda k: (FAMILY_ORDER.index(k)
                                                                        if k in FAMILY_ORDER else 99, k))}
    return {"suite_version": suite.get("version"), "unit": "distinct case; a case passes only if every repeat passes",
            "families": ordered, "twin_pairs": twin if pairs else None, "cue_triplets": cue or None,
            "cue_summary": cue_summary, "cases": per_case}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", nargs="?")
    p.add_argument("--suite")
    p.add_argument("--validate", nargs="+")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    if a.validate:
        for path in a.validate:
            print(path, json.dumps(validate_suite(json.loads(Path(path).read_text(encoding="utf-8")))))
        return 0
    if not (a.report and a.suite):
        p.error("a report and --suite, or --validate")
    suite_bytes = Path(a.suite).read_bytes()
    report = json.loads(Path(a.report).read_text(encoding="utf-8"))
    if hashlib.sha256(suite_bytes).hexdigest() != report["metadata"]["suite_sha256"]:
        raise SystemExit("the report was not run on this suite (suite_sha256 differs)")
    result = score_report(report, json.loads(suite_bytes))
    if a.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"{'family':20} {'strict':>9} {'content':>9}  content-only cases")
    for name, f in result["families"].items():
        tag = " (diagnostic)" if f["diagnostic"] else ""
        print(f"{name + tag:20} {f['strict_text']:>9} {f['content_text']:>9}  {', '.join(f['content_not_strict'])}")
    if result["twin_pairs"]:
        t = result["twin_pairs"]
        print(f"twin pairs: both strict {t['both_strict']} of {t['pairs']}, both content {t['both_content']} of {t['pairs']}")
    if result["cue_summary"]:
        c = result["cue_summary"]
        print(f"cue triplets (diagnostic): selective {c['selective']} of {c['triplets']} by content, "
              f"{c['selective_strict']} strictly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
