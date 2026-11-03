# Independent review of the v0.0.2 release gates

Claude Opus 5.5, 3 November 2026, at Jonathan's request, while candidate A trains and before any candidate output has
been generated or read. This is a review, not a change to the gates. `release_gate_v002.json`, the suites, gold and
every receipt are untouched. Acceptance, and any change below, is Jonathan's and Codex's to make, and should be
made before candidate A is evaluated.

Read: `release_gate_v002.json`, `V002_RELEASE.md`, `V003_ADOPTION.md`, `training_data_v003.json`, `evaluate.py`'s
scorer, `suite_release_v002.json`, and the v0.0.3 training packet
(`C:/ml_artifacts/sfull_uplift_20261103/token-packet-v3/train.jsonl`).

## Summary

The design is sound for its purpose:
- strict exact scoring, kept separate from judge ratings;
- development selection, then a single holdout run, with a predeclared tie-break;
- a failure on any dimension keeps the baseline;
- no weighted average;
- bounded candidates;
- evaluation data never reaches the trainer.

Five points matter before candidate A is read. Two should change first (1 and 2). Three change how a result is
reported (3, 4 and 5).

## 1. Accept and freeze the gates before candidate A is evaluated (do first)

`release_gate_v002.json` still says `"accepted": false`. `V003_ADOPTION.md` calls its criteria the intended
acceptance targets, and training is under way. If the gates are accepted after a candidate's outputs are seen, they
become post hoc, which is the failure the README's own iteration rules guard against.

**Proposal:** record acceptance, with any changes from points 2-5, as a new file or a new version with its own hash.
Commit it before the first candidate run of `evaluate.py`. The candidate's report should then cite that hash.

## 2. The judge gates cannot pass until their calibration exists

The judgment block sets relevance and coherence floors (at least 0.7 of turns rated 3 or more), but it also says
`"calibration": "human or independent rubric review before accepting thresholds"`. No calibration has been done, so
as written the release cannot pass. The alternative is accepting uncalibrated Gemini thresholds, and nothing defines
whether that is allowed.

**Proposal:** either
- (a) define the calibration now: for example, Jonathan rates the fixed five-case subset blind (all turns, first
  repeat) on the same 0-4 rubric for the baseline, and an agreement criterion with Gemini is fixed before any
  candidate is judged; or
- (b) mark the judgment gates diagnostic for v0.0.2, and say so in the accepted file.

## 3. Report counts, not fractions: each floor is one or two cases

| gate | floor | cases behind it | what passing means |
|---|---|---:|---|
| `dev_exact_min` | 0.25 | 13 dev | at least 4 of 13 |
| `immediate_exact_min` | 0.5 | 2 | at least 1 of 2 |
| `delayed_exact_min` | 0.25 | 2 | at least 1 of 2 (the same bar as 0.5) |
| `correction_exact_min` | 0.5 | 2 | at least 1 of 2 |
| `holdout_exact_min` | 0.5 | 8 holdout, at most 2 per family | at least 4 of 8 |
| `min_additional_distinct_cases_over_matched_baseline` | 2 | baseline 0/13 | at least 2 passes, weaker than the 4 above |

Greedy repeats are identical by construction, so there is no sampling variance to average over. A single case flips
a family gate. That is acceptable for a bounded engineering gate, but the report should say "1 of 2", never "50%",
and should not describe a family as learned on one case. Note also that `delayed_exact_min` 0.25 and 0.5 are
identical on two cases. If the intent was a lower bar for delayed recall, it does not exist.

## 4. "Holdout transfer" is new wordings of trained skills, not transfer to untrained behaviour

v0.0.3 adds 1,077 programmatic conversations covering eight skills: qa, choice, format, immediate, delayed,
correction, unknown and continuity. Those are the same skills the development families and all six `transfer_*`
holdout families test. So the holdout measures whether trained skills generalise to new surface forms. It does not
measure transfer to behaviour the training did not target.

The surface overlap is low, which is to the generator's credit.
- **Method:** `audit_template_overlap.py` masks numbers, days, colours and capitalised names, then compares every suite
  user turn with all 2,511 programmatic augmentation user turns.
- **Result:** no case's closest match exceeds a token-set Jaccard of 0.50, and no shared run is longer than 3 masked
  tokens. The development and holdout means are both 0.31.
- **Structural kinship:** the holdout `transfer_choice` case ("The copper ticket belongs to Mira; the silver ticket
  belongs ...") and the augmentation's choice rows ("Elena carries a white wallet; Marco carries a pink wallet ...")
  share a task schema while sharing few words.

**Proposal:** describe the holdout as "held-out instances of trained skills". If transfer to untrained behaviour
matters for release, add a diagnostic family the augmentation does not target, frozen before candidate A is read.
Candidates are multi-fact recall across three or more turns with a distractor, or the open-conversation relevance the
judge rates.

## 5. Read every exact gain together with `semantic_content`

The scorer passes a reply only if the whole normalised reply equals an accepted answer, so "The answer is 5" fails.
The augmentation's assistant targets are bare answers ("13"). A candidate can therefore gain exact passes purely by
learning the bare-answer format, without any recall it lacked before. The baseline shows this: Nora and blue were
mentioned in replies that failed only the format.

**Proposal:** for every case that flips from fail to pass, also report the baseline's `answer_mentioned`.
- **A flip where the baseline already mentioned the answer** is a format gain.
- **A flip where it did not** is a content gain.

This uses the existing diagnostic and needs no new gate.

## Also found

- **Fixed in this change:** the manifest check failed on LF checkouts. `evaluation_manifest_v002_final.json` holds the
  hashes of its ten files with CRLF line endings, as a Windows checkout writes them. The repository stores LF, so
  `python freeze.py --check evaluation_manifest_v002_final.json` reported "manifest drift" on Linux, WSL and any fresh
  clone, although the content had not changed: all ten match after LF-to-CRLF conversion. `.gitattributes` now
  checks those ten files out with CRLF, so the existing manifest verifies everywhere. No frozen file or hash changed.
  Verified on a WSL checkout: "Frozen evaluation manifest verified", and the tests pass.
- **Only the final reply of a multi-turn case is scored** (`score(case, replies[-1])`). That is what the suite
  intends. A model that answers correctly early and then contradicts itself is scored on the contradiction, which is
  right.
- **The selection and stopping rules look good as written:** development-only selection, the lower-learning-rate
  tie-break, one holdout run for the selected candidate and the preserved baseline, and no automatic extra candidates.
