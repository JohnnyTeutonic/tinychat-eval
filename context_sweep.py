"""Context-boundary sweep: recall of an early fact against prompt length, and where the server starts dropping history.

A separate diagnostic instrument (README: "A dedicated context-boundary sweep is a later, separate instrument"). It is
not a release gate and not a conversation-quality score. Each point is one request: a planted fact in the first user
turn, then F neutral filler exchanges with the frozen acknowledgement "Noted." as every assistant turn (so lengths are
deterministic), then a question about the fact with an exact answer contract. The server's reported
usage.prompt_tokens places each point. A calibration pass first measures the exact token cost of each filler exchange
(F = 0..len(FILLER) on the first probe), so every point's full, untruncated length is predicted exactly. A point is in
context only if the server reports exactly that length. A shorter report means the server dropped older turns to fit
its context. The planted fact is in the oldest exchange, so from that point on it is no longer in what the model sees.
A longer report than predicted means token costs are not additive; it is recorded as an error, never as context.

Reported per point: F, prompt_tokens, the predicted length, in_context, the reply, exact pass, and the
answer-mention flag. Per probe: the last in-context point, and the first point where history was dropped. After
truncation, a correct answer is impossible from context, and the reply shows whether the model says it does not know
or invents something; this script records, it does not judge.

Settings are evaluate.py's SETTINGS (greedy, 32 tokens). Probes and filler are synthetic and are not training data.

  python context_sweep.py run --weights MODEL.gguf --server-identity ID --candidate NAME --out runs/NAME-context
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

import evaluate

# Facts distinct from suite.json / suite_release_v002.json and from the v0.0.3 augmentation's skills' surface forms.
PROBES = [
    {"id": "locker_code", "fact": "The storage locker code is 4719.",
     "question": "What is the storage locker code? Reply with just the number.", "answers": ["4719"]},
    {"id": "neighbour_name", "fact": "My neighbour is called Halvard.",
     "question": "What is my neighbour called? Reply with just the name.", "answers": ["Halvard"]},
    {"id": "key_place", "fact": "The spare key is in the mailbox.",
     "question": "Where is the spare key? Reply with just the place.",
     "answers": ["mailbox", "the mailbox", "in the mailbox"]},
    {"id": "platform", "fact": "My train leaves from platform nine.",
     "question": "Which platform does my train leave from? Reply with just the number.", "answers": ["9", "nine"]},
]
# Neutral filler: no numbers, names or places a probe asks about.
FILLER = [
    "Here is an unrelated note about my day: the morning was quiet.",
    "I read a few pages of a novel over lunch.",
    "The kettle took a long time to boil this afternoon.",
    "I tidied the kitchen and watered the plants.",
    "There was a light breeze when I went outside.",
    "I listened to some music while I cooked dinner.",
    "The evening was calm and I went to bed early.",
    "I wrote a short list of things to buy at the shops.",
]
SCHEDULE = [0, 1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]


def messages_for(probe, fillers):
    m = [{"role": "user", "content": probe["fact"]}, {"role": "assistant", "content": "Noted."}]
    for i in range(fillers):
        m.append({"role": "user", "content": FILLER[i % len(FILLER)]})
        m.append({"role": "assistant", "content": "Noted."})
    m.append({"role": "user", "content": probe["question"]})
    return m


def filler_costs(calibration):
    """Token cost of each filler exchange: differences of reported prompt lengths at F = 0..len(FILLER)."""
    t = [c.get("prompt_tokens") for c in calibration]
    if len(t) != len(FILLER) + 1 or any(x is None for x in t):
        raise ValueError("calibration incomplete")
    costs = [t[i + 1] - t[i] for i in range(len(FILLER))]
    if any(c <= 0 for c in costs):
        raise ValueError(f"calibration found a nonpositive filler cost {costs}: history dropped during calibration")
    return costs


def mark_context(points, costs):
    """Per point: predicted_tokens = the probe's F = 0 length plus the costs of its F filler exchanges, and
    in_context = (reported == predicted). A report below the prediction means older turns were dropped; that point and
    every later one are out of context. A report above it means costs are not additive: in_context None, and
    'length_mismatch' is set (an error to inspect, not a context result)."""
    base = next((p["prompt_tokens"] for p in points if p["fillers"] == 0 and p.get("prompt_tokens") is not None), None)
    lost = False
    for p in points:
        tokens = p.get("prompt_tokens")
        if tokens is None or base is None:
            p["in_context"] = None
            continue
        p["predicted_tokens"] = base + sum(costs[i % len(costs)] for i in range(p["fillers"]))
        if tokens > p["predicted_tokens"]:
            p["in_context"] = None
            p["length_mismatch"] = True
            continue
        if tokens < p["predicted_tokens"]:
            lost = True
        p["in_context"] = not lost
    return points


def run(args):
    health, _ = evaluate.curl_json(args.url.rstrip("/") + "/health")
    models, _ = evaluate.curl_json(args.url.rstrip("/") + "/v1/models")
    model = models["data"][0]["id"]
    weights = Path(args.weights)
    weight_sha = evaluate.digest(weights.read_bytes())
    if Path(health["model"].replace("\\", "/")).name != weights.name:
        raise ValueError("server model filename differs from supplied provenance")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"instrument": "context_sweep", "harness_sha256": evaluate.digest(Path(__file__).read_bytes()),
                "evaluate_sha256": evaluate.digest(Path(evaluate.__file__).read_bytes()),
                "probes_sha256": evaluate.digest(evaluate.canonical({"probes": PROBES, "filler": FILLER,
                                                                      "schedule": SCHEDULE})),
                "weights_sha256": weight_sha, "health": health, "model_id": model, "settings": evaluate.SETTINGS,
                "history": "controlled: every assistant turn is 'Noted.'", "candidate": args.candidate,
                "server_identity": args.server_identity, "endpoint": args.url,
                "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "limitation": "Diagnostic only, not a release gate. Truncation is inferred from reported prompt "
                              "lengths, not reported by the server; controlled acknowledgements are not natural "
                              "conversation."}
    (out / "metadata.json").write_bytes(evaluate.canonical(metadata))

    def ask(probe, fillers, journal, stage):
        request = dict(evaluate.SETTINGS, model=model, messages=messages_for(probe, fillers))
        point = {"stage": stage, "probe": probe["id"], "fillers": fillers, "complete": False}
        try:
            reply, elapsed = evaluate.curl_json(args.url.rstrip("/") + "/v1/chat/completions", request)
            point.update(request=request, response=reply, wall_ms=elapsed)
            choice = reply["choices"][0]
            text = choice["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("nontext model reply")
            case = {"check": {"kind": "exact", "answers": probe["answers"]}}
            point.update(reply=text, finish_reason=choice.get("finish_reason"),
                         prompt_tokens=reply.get("usage", {}).get("prompt_tokens"),
                         exact_pass=evaluate.score(case, text),
                         semantic_content=evaluate.semantic_content(case, text), complete=True)
        except (RuntimeError, ValueError, KeyError, IndexError, TypeError, subprocess.TimeoutExpired) as error:
            point["error"] = type(error).__name__ + ": " + str(error)
        journal.write(json.dumps(point, ensure_ascii=False) + "\n")
        journal.flush()
        return point

    results = []
    with (out / "points.jsonl").open("x", encoding="utf-8") as journal:
        calibration = [ask(PROBES[0], f, journal, "calibration") for f in range(len(FILLER) + 1)]
        try:
            costs = filler_costs(calibration)
        except ValueError as error:
            report = {"metadata": metadata, "calibration_error": str(error), "probes": []}
            (out / "report.json").write_bytes(evaluate.canonical(report))
            print("calibration failed: " + str(error))
            raise SystemExit(2)
        for probe in PROBES:
            points = [ask(probe, fillers, journal, "sweep") for fillers in SCHEDULE]
            mark_context(points, costs)
            inside = [p for p in points if p.get("in_context")]
            dropped = [p for p in points if p.get("in_context") is False]
            results.append({
                "probe": probe["id"],
                "points": [{k: p.get(k) for k in ("fillers", "prompt_tokens", "predicted_tokens", "in_context",
                                                  "exact_pass", "finish_reason", "reply")} | {
                               "answer_mentioned": (p.get("semantic_content") or {}).get("answer_mentioned"),
                               "length_mismatch": bool(p.get("length_mismatch"))}
                           for p in points],
                "last_in_context": {k: inside[-1][k] for k in ("fillers", "prompt_tokens")} if inside else None,
                "first_history_dropped": {k: dropped[0][k] for k in ("fillers", "prompt_tokens")} if dropped else None,
                "exact_in_context": f"{sum(bool(p.get('exact_pass')) for p in inside)} of {len(inside)}",
                "mentioned_in_context": f"{sum(bool((p.get('semantic_content') or {}).get('answer_mentioned')) for p in inside)} of {len(inside)}",
            })
    errors = sum(1 for line in (out / "points.jsonl").open(encoding="utf-8") if not json.loads(line)["complete"])
    try:
        final_health, _ = evaluate.curl_json(args.url.rstrip("/") + "/health")
        stable = final_health == health and evaluate.digest(weights.read_bytes()) == weight_sha
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired):
        stable = False
    mismatches = sum(p["length_mismatch"] for r in results for p in r["points"])
    report = {"metadata": metadata, "server_and_weights_stable": stable, "operational_errors": errors,
              "length_mismatches": mismatches, "filler_costs": costs, "context": health.get("context"),
              "probes": results}
    (out / "report.json").write_bytes(evaluate.canonical(report))
    print(json.dumps([{k: r[k] for k in ("probe", "last_in_context", "first_history_dropped", "exact_in_context",
                                         "mentioned_in_context")} for r in results], indent=2))
    if not stable or errors or mismatches:
        raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8081")
    r.add_argument("--weights", required=True)
    r.add_argument("--server-identity", required=True)
    r.add_argument("--candidate", required=True)
    r.add_argument("--out", required=True)
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
