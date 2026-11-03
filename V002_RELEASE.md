# Tiny-chat v0.0.2: frozen proposal, acceptance required

No training admission is granted by this document. Original suite.json, gold and
runs/sfull-baseline-* receipts remain unchanged. Baseline must be rerun with this
new harness before paired comparisons; historical reports are not rewritten.

## Evaluation instrument

suite_release_v002.json copies the original scenarios without changing gold and
adds six release-only families: alternative choice, correction, delayed recall,
unknown information, literal formatting and immediate recall. There are 13 dev
and eight holdout scenarios. They are authored regression fixtures, not a validated
population sample. Public holdout means operator separation, not secret gold.
Never copy either suite, answer keys or evaluation transcripts into a training upload.

Every completed episode has separate exact_pass and semantic_content fields.
The latter detects whole-answer mentions and flags negation; it intentionally does
not declare correctness. Nora in 'My name is Nora' survives this diagnostic while
strict answer format fails. The separate `judge --semantic --suite suite_release_v002.json` mode provides
reference-based semantic-content ratings with quoted evidence and a hash-bound answer
contract. These fallible meaning ratings ignore format, and cannot overwrite exact
scores or be uploaded as final-turn relevance ratings. No new semantic judging calls
have been made. Role reversal, contradictory alternatives and valid
paraphrases require transcript adjudication: mention is not semantic truth. These
content flags must never enter release gates as semantic accuracy.

--history-mode controlled replaces preceding generated assistant replies with the
frozen 'Noted.' acknowledgement. Actual outputs remain in receipts. This isolates
user-fact use from self-generated contamination imperfectly: artificial acknowledgements
also change discourse. Generated history alone is primary. Never pool the modes.

--per-turn Gemini judgment uses all turns of first repeats in the fixed five-case
subset named in release_gate_v002.json; --limit selects a fixed bounded prefix.
Use limit 20 and max-calls 20 to cover the current subset. Incomplete judgments
cannot satisfy a release gate. Quote evidence and keep every turn, failure and
usage record. No cloud judging has been run for the new instrument. Calibrate the
rubric and inspect role/negation/paraphrase disagreements before relying on ratings.

The versioned gate separates operations, strict task success, immediate/delayed
recall, correction, held-out transfer and per-turn relevance/coherence. Its numerical
floors are prospective engineering targets requiring acceptance, not empirically
validated population thresholds. Failure on any dimension retains v0.0.1; no weighted
quality average. Exact case-level regression must be checked, not just family means.
Two greedy repeats measure repeatability, not independent statistical evidence.

## Training matrix

| Candidate | Start | Data | Updates | Learning rate | Purpose |
|---|---|---|---|---|---|
| preserved v0.0.1 | SFT1500 | unchanged | 0 | none | matched baseline |
| A | identical SFT1500 | same frozen v2 packet | 256 | 3e-5 | conservative adaptation |
| B | identical SFT1500 | same frozen v2 packet | 256 | 1e-4 | matched LR contrast |

This is continued SFT of an independently trained model, not fresh pretraining.
Both candidate specs are copied here for inspection. Architecture d768, eight
layers, 12 heads, FFN2240, four state tokens, tokenizer/vocabulary16384, context1024,
chat template, inference precision and greedy evaluation settings remain fixed.
Both arms use fresh AdamW, batch16, seed1, clip1, 32-update warmup/cosine,
minimum LR fraction0.1 and full-state checkpoints every25updates/roughly3minutes.
These are proposed endpoints; do not reuse or weaken historical S-full validators.

Use the already balanced packet-v2, not the earlier instruction-heavy packet-v1:
4072 training rows, 2997 multi-turn, 484 everyday conversations. Source is
HuggingFaceTB/smol-smoltalk at f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc;
shard SHA256 498cd4580014f42c40cfde066573564ed28e3e1548a5566715b615f5ea932856.
The pinned card declares Apache2.0; retain individual component provenance.
Training packet SHA256 adce0e30d73c0dd6e30f3faaa2c0fea9e568f699c23f11e79da40d2d39a8f241.
Token vocabulary SHA256 0b880aa073a7e27c0770d5ecc5215b55d6d415080d20bd3dc7b169ecabadf198.
Full intake/token receipts are in AI_ML/sfull_uplift_20261103 and local artifacts.

Exact/whitespace conversation duplicates are grouped before hash partitioning;
source round-robin selection, seeded shuffle and whole-exchange context fitting
are already implemented. No semantic near-duplicate guarantee exists. Before
admission, audit retained text for the seven requested skills and document counts;
source labels do not prove recall/correction/alternative-question coverage. Reject
exact evaluation-message overlaps and shared long n-grams; inspect fuzzy/template
matches, deduplicate by conversation/family before splits. Do not build synthetic
training conversations by renaming these evaluation fixtures. If missing categories
need supplementation, use independently authored/generator templates with source,
model, prompt and seed provenance, then freeze a NEW common packet before either
arm. This changes the proposal manifest and requires reacceptance. Evaluation gold
is read only by a separate contamination auditor, never the trainer or generator.

## Bounds and admission

Two candidates only, at most512 optimizer updates total, at most8388608 padded
context positions total (512*16*1024); actual assistant target tokens must be logged.
Proposal: one L4 serially, maximum US$3 per arm including startup/export/parity/recovery,
US$6 total GPU envelope including failed attempts and storage; new Gemini judging
is separately bounded at20calls per candidate/mode, at most120calls overall
(three model identities, semantic plus per-turn),1024outputtokens/call, no retries.
No dollar estimate is asserted for these API calls before checking the configured
model price and billing account. Do not spend this proposed API envelope before acceptance, preserving the account US$10 reserve.
At the recently observed US$0.59/hour this buys less than5.1GPU-hours per arm
before storage/overheads. Four-hour absolute deadline per arm. This is a hard envelope,
not a measured runtime forecast. Measure startup and first25updates; use remaining
cadence plus export/eval/recovery and25% headroom; refuse continuation if it cannot
fit. Stop on nonfinite loss, checkpoint guard or budget/deadline; preserve evidence.
Full-state relay/manager integration and new source freeze remain unverified and
block admission. No new campaign is launched by the preparation scripts.

## Commands (remote payload after separate reviewed watchdog admission)

Run in a pinned Paperkiln CUDA build with allowlisted inputs installed at the spec's
/content/chat/upload paths. Do not run the training command on the laptop GPU.

```bash
./build/mtstudio plan /content/chat/upload/training_lr3e5_v002.json
./build/mtstudio run /content/chat/upload/training_lr3e5_v002.json
./build/mtstudio export /content/chat/upload/training_lr3e5_v002.json --from /content/chat/sfull_uplift_lr3e5/model.safetensors --formats gguf --out-dir /content/chat/sfull_uplift_lr3e5/export
```

The exact proposed training payload is the `mtstudio run` command above; B substitutes
lr1e4. These are real existing CLI commands, not a complete provider admission command.
Final output filenames, source pin and checkpoint manager must be validated by dry-run
before wiring the watchdog; no fictional launcher is supplied as executable admission.
Use CPU Ember and Paperkiln servers on separate ports with candidate export/spec:

```bash
PARITY_EMBER_PORT=8090 PARITY_MTS_PORT=8091 PARITY_GGUF=/path/candidate.gguf PARITY_PROMPTS=/path/frozen-parity.jsonl PARITY_MAX_NEW=32 PAPERKILN_TOOLS=/path/paperkiln/tools python /path/ember.cpp/tools/chat_parity.py 20 /path/new-parity.json
```

Parity must retain token IDs/counts, text and stop reasons, finite scores and the
existing numerical tolerance; parity prompts are not training data. Existing
chat_parity.py compares chat behavior and tokenizer counts, not complete logit
parity; full score/logit cross-engine validation remains required before release.

```bash
python freeze.py --check evaluation_manifest_v002_final.json
python evaluate.py run --suite suite_release_v002.json --weights /path/model.gguf --server-identity PINNED_SERVER --candidate baseline-v002-instrument --out runs/new-baseline
python evaluate.py run --suite suite_release_v002.json --history-mode controlled --weights /path/model.gguf --server-identity PINNED_SERVER --candidate baseline-controlled --out runs/new-baseline-controlled
python integrations.py judge --per-turn --env-file /explicit/config.env --report runs/new-baseline/report.json --out runs/new-baseline/per-turn.json --model gemini-3.1-pro-preview --max-calls 20 --limit 20
python evaluate.py compare runs/new-baseline/report.json runs/candidate-A/report.json
```

Select only on dev, with predeclared tie-break lower LR; run release holdout once
for the selected candidate AND preserved baseline under the same new instrument
(`--split holdout --allow-holdout`). Do not inspect holdout to choose A versus B.
No training loss-only winner, no automatic extra candidates after a failed release.

Rollback: keep original weights, GGUF, tokenizer, serving binary and receipts. Run
candidates on separate ports; switch the public/local primary only after every gate
and explicit acceptance. Otherwise leave the original 8081 server untouched.

## Acceptance checklist

Implemented and unit-tested: versioned suite, controlled history, conservative
content flags, per-turn judge option and hash manifest. Still unverified: broader
semantic adjudication, complete per-turn ratings, actual transfer performance,
data skill coverage/near-duplicate audit, trainer plan in exact pinned CUDA image,
full-state checkpoint relay, end-to-end export/logit parity and measured runtime.
No new gold or training outcome was used to set candidate-dependent thresholds.
