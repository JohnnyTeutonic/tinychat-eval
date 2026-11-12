"""Offline v0.0.3 release-gate evaluator: reads saved runs, calls no server, model, judge or network.

    python gate_v003.py --candidate-dev runs/C --baseline-dev runs/B
                        [--candidate-holdout runs/CH --baseline-holdout runs/BH]
                        [--export-parity passed|failed] [--judgments J.json ...]

Prints every criterion and the verdict (release_gate_v003.json, "verdicts"); exit 0 pass, 1 fail, 2 incomplete.
Every run the verdict reads must be provenance 'verified'. Judgments are reported and never enter the verdict.
"""
import argparse
import json
from pathlib import Path

import evaluate
import provenance

ROOT = Path(__file__).parent
GATE = ROOT / "release_gate_v003.json"


def load_gate(path=GATE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scored_cases(suite, split):
    return [c["id"] for c in suite["cases"] if c["split"] == split and c["check"]["kind"] == "exact"]


def case_passes(report, case_ids):
    """{case_id: bool}: a case passes when it has episodes and every repeat passes its exact check."""
    by = {}
    for e in report["episodes"]:
        by.setdefault(e["case_id"], []).append(e.get("exact_pass"))
    return {c: bool(by.get(c)) and all(v is True for v in by[c]) for c in case_ids}


def criterion(name, passed, detail):
    return {"name": name, "passed": passed, "detail": detail}


def _fraction(passes, ids):
    return sum(passes[c] for c in ids), len(ids)


def _run_checks(role, run_dir, report, gate, split):
    meta, summary, inst = report["metadata"], report["summary"], gate["instrument"]
    out = []
    status = provenance.verify(run_dir)
    out.append(criterion(role + ": provenance", status["status"] == gate["provenance"]["required_status"],
                         {"status": status["status"], "reasons": status["reasons"]}))
    binding = {"suite_sha256": inst["suite_sha256"], "harness_sha256": inst["harness_sha256"],
               "settings": evaluate.SETTINGS, "history_mode": inst["history_mode"], "split": split}
    wrong = sorted(k for k, v in binding.items() if meta.get(k) != v)
    out.append(criterion(role + ": instrument", not wrong, {"differs": wrong}))
    complete = summary["attempted_episodes"] > 0 and summary["complete_episodes"] == summary["attempted_episodes"]
    out.append(criterion(role + ": complete", complete and summary["operational_errors"] == 0,
                         {"complete": summary["complete_episodes"], "attempted": summary["attempted_episodes"]}))
    rep = summary["repeat_identical"]
    enough = meta.get("repeats", 0) >= inst["min_repeats"] and rep["n"] > 0
    out.append(criterion(role + ": repeat identical", enough and rep["passed"] == rep["n"],
                         {"identical": rep["passed"], "cases": rep["n"], "repeats": meta.get("repeats")}))
    out.append(criterion(role + ": stable server", report.get("server_and_weights_stable") is True,
                         {"server_and_weights_stable": report.get("server_and_weights_stable")}))
    return out


def _pair(name, before, after):
    try:
        evaluate.compare(before, after)
        return criterion(name, True, {})
    except ValueError as e:
        return criterion(name, False, {"refused": str(e)})


def _judgment_summary(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    judged = [j for j in data.get("judgments", []) if j.get("status") == "JUDGED" and isinstance(j.get("value"), dict)]
    out = {"file": str(path), "mode": data.get("mode"), "judge_model": data.get("judge_model"),
           "calls": len(data.get("judgments", [])), "judged": len(judged), "affects_verdict": False}
    for key in ("relevance", "coherence"):
        hits = sum(1 for j in judged if j["value"].get(key, -1) >= 3)
        out[key + "_at_least_3"] = f"{hits} of {len(judged)}"
    return out


def evaluate_gate(candidate_dev, baseline_dev, candidate_holdout=None, baseline_holdout=None,
                  export_parity=None, judgments=(), gate=None):
    """{"verdict", "criteria", "diagnostics", "stated_limits"} from saved run directories."""
    gate = gate or load_gate()
    suite = json.loads((ROOT / gate["instrument"]["suite"]).read_text(encoding="utf-8"))
    dev_ids, holdout_ids = scored_cases(suite, "dev"), scored_cases(suite, "holdout")
    families = {c["id"]: c["family"] for c in suite["cases"]}
    load = lambda d: json.loads((Path(d) / "report.json").read_text(encoding="utf-8"))  # noqa: E731
    cd, bd = load(candidate_dev), load(baseline_dev)
    crit = _run_checks("candidate dev", candidate_dev, cd, gate, "dev")
    crit += _run_checks("baseline dev", baseline_dev, bd, gate, "dev")
    crit.append(_pair("dev comparability", bd, cd))
    crit.append(criterion("export parity", export_parity, {"evidence": "operator supplied" if export_parity is not None
                                                           else "not supplied"}))
    cp, bp = case_passes(cd, dev_ids), case_passes(bd, dev_ids)
    k, n = _fraction(cp, dev_ids)
    tc = gate["task_correctness"]
    crit.append(criterion("dev exact", n > 0 and k / n >= tc["dev_exact_min"],
                          {"passed": k, "denominator": n, "unit": "scored development cases", "min": tc["dev_exact_min"]}))
    gained = sorted(c for c in dev_ids if cp[c] and not bp[c])
    regressed = sorted(c for c in dev_ids if bp[c] and not cp[c])
    crit.append(criterion("gain over baseline", len(gained) >= tc["min_additional_distinct_cases_over_matched_baseline"],
                          {"gained": gained}))
    crit.append(criterion("no dev regression", len(regressed) <= tc["max_regressed_distinct_cases"],
                          {"regressed": regressed}))
    for key, family in gate["recall"]["families"].items():
        ids = [c for c in dev_ids if families[c] == family]
        k, n = _fraction(cp, ids)
        crit.append(criterion(family, n > 0 and k / n >= gate["recall"][key],
                              {"passed": k, "denominator": n, "min": gate["recall"][key]}))
    if candidate_holdout is None or baseline_holdout is None:
        crit.append(criterion("holdout", None, {"evidence": "holdout runs not supplied"}))
    else:
        ch, bh = load(candidate_holdout), load(baseline_holdout)
        crit += _run_checks("candidate holdout", candidate_holdout, ch, gate, "holdout")
        crit += _run_checks("baseline holdout", baseline_holdout, bh, gate, "holdout")
        crit.append(_pair("holdout comparability", bh, ch))
        hp, hb = case_passes(ch, holdout_ids), case_passes(bh, holdout_ids)
        k, n = _fraction(hp, holdout_ids)
        tr = gate["transfer"]
        crit.append(criterion("holdout exact", n > 0 and k / n >= tr["holdout_exact_min"],
                              {"passed": k, "denominator": n, "min": tr["holdout_exact_min"]}))
        lost = sorted(c for c in holdout_ids if hb[c] and not hp[c])
        crit.append(criterion("no holdout regression", len(lost) <= tr["max_regressed_cases"], {"regressed": lost}))
    # The verdict is computed from the required criteria only, before any judgment is read.
    states = [c["passed"] for c in crit]
    verdict = "fail" if False in states else ("incomplete" if None in states else "pass")
    diagnostics = {"judgments": [_judgment_summary(j) for j in judgments],
                   "judgments_role": gate["judgments"]["role"]}
    return {"gate_version": gate["version"], "verdict": verdict, "criteria": crit, "diagnostics": diagnostics,
            "stated_limits": gate["stated_limits"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate-dev", required=True)
    p.add_argument("--baseline-dev", required=True)
    p.add_argument("--candidate-holdout")
    p.add_argument("--baseline-holdout")
    p.add_argument("--export-parity", choices=["passed", "failed"])
    p.add_argument("--judgments", nargs="*", default=[])
    a = p.parse_args(argv)
    parity = None if a.export_parity is None else a.export_parity == "passed"
    result = evaluate_gate(a.candidate_dev, a.baseline_dev, a.candidate_holdout, a.baseline_holdout, parity,
                           a.judgments)
    print(json.dumps(result, indent=2))
    return {"pass": 0, "fail": 1, "incomplete": 2}[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
