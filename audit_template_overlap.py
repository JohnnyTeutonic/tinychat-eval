"""How close are the v0.0.3 programmatic augmentation prompts to the evaluation suite's, once numbers and names are
masked? For each suite case's user turns: the closest augmentation user turn by token-set Jaccard, and the longest
common run of masked tokens. Read-only; prints a table."""
import json
import re
import sys
from collections import defaultdict

suite_path, train_path = sys.argv[1], sys.argv[2]
NUMS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
           "seventeen eighteen nineteen twenty thirty forty fifty hundred".split())
DAYS = set("monday tuesday wednesday thursday friday saturday sunday".split())
COLOURS = set("red blue green yellow purple orange black white pink brown grey gray".split())


def mask(text):
    out = []
    for i, w in enumerate(re.findall(r"[A-Za-z']+|\d+|[^\sA-Za-z\d]", text)):
        lw = w.lower()
        if w.isdigit() or lw in NUMS:
            out.append("<n>")
        elif lw in DAYS:
            out.append("<day>")
        elif lw in COLOURS:
            out.append("<colour>")
        elif w[0].isupper() and i > 0 and lw not in ("i",):
            out.append("<name>")
        else:
            out.append(lw)
    return out


def lcs_run(a, b):
    best = 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


aug = []  # (skill, masked user turn, raw)
for line in open(train_path, encoding="utf-8"):
    r = json.loads(line)
    if r.get("provenance", {}).get("kind") != "programmatic_synthetic":
        continue
    skill = r["provenance"].get("skill")
    for seg in r["segments"]:
        if seg["loss"]:
            continue
        for u in re.findall(r"<\|user\|>(.*?)(?=<\|assistant\|>|$)", seg["text"], flags=re.S):
            aug.append((skill, mask(u.strip()), u.strip()))
print(f"programmatic augmentation user turns: {len(aug)}")

suite = json.load(open(suite_path, encoding="utf-8"))
print(f"{'case':24} {'split':8} {'family':26} {'closest skill':14} {'jacc':>5} {'run':>4}  eval turn | closest augmentation turn")
summary = defaultdict(list)
for c in suite["cases"]:
    best = (0.0, 0, None, None, None)
    for u in c["users"]:
        mu = mask(u)
        su = set(mu)
        for skill, ma, raw in aug:
            sa = set(ma)
            j = len(su & sa) / len(su | sa) if su | sa else 0.0
            if j > best[0] - 1e-12:
                run = lcs_run(mu, ma)
                if (j, run) > (best[0], best[1]):
                    best = (j, run, skill, u, raw)
    summary[c["split"]].append(best[0])
    print(f"{c['id']:24} {c['split']:8} {c['family']:26} {str(best[2]):14} {best[0]:5.2f} {best[1]:4d}  "
          f"{best[3][:60]!r} | {best[4][:60]!r}")
for k, v in summary.items():
    print(k, "cases", len(v), "with jaccard >= 0.6:", sum(x >= 0.6 for x in v), "mean", round(sum(v) / len(v), 2))
