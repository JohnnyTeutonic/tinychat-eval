# tinychat-eval

A small, independent evaluation repo for the Paperkiln → GGUF → Ember chat loop.
The core uses Python's standard library and actual curl requests. No PyTorch is needed.
Langfuse is an experiment dashboard; local immutable receipts remain authoritative.

## What this first version establishes

The frozen pilot has 13 development scenarios and two release-only transfer scenarios.
It is a regression instrument, not a validated population benchmark or a claim to SoTA.
It measures basic Q&A, immediate/delayed recall, correction, missing information,
output instructions, and open conversation. All people and situations are synthetic.
The server generates every assistant turn: the runner sends actual generated history,
not ideal assistant replies. Each episode starts with fresh explicit messages.

Greedy controls are frozen: temperature 0, top-k 1, top-p 1, seed 123, at most 32 tokens
per reply. Two whole-episode repeats *measure* repeatability; setting temperature 0
alone does not establish determinism. Repeats are not independent task samples.

Exact normalized answers are deliberately strict contracts: 'Tuesday or Thursday'
does not pass a Thursday-only question. They measure successful answer-and-format
contracts, not all possible semantically correct paraphrases. Unknown-information cases
ask whether a fact was supplied; they do not rely on a judge guessing factual truth.
Every such case's gold answer is 'no', and no case asks about a fact that was supplied,
so a model that answers 'No.' to any yes/no question passes them all. Read their passes
with the replies, not as evidence that the model tracks what it was told.

Coherence, fluency and relevance are separate 0–4 rubric judgments with evidence.
They are UNJUDGED until the opt-in judge runs. A fluent answer can be irrelevant or
wrong. Judge scores never overwrite exact task results. Gemini is a fallible external
annotator, not an oracle; inspect the transcript and disagreement before acting.

## Run

The server must already be running. Preserve any existing interactive chat session.

```powershell
python evaluate.py run --weights C:/ml_artifacts/sfull_local_chat_20261103/tc_sft_Sfull_r1.gguf --server-identity archived-ember-cpu-server-Sfull-20261103 --candidate sfull-baseline --out runs/baseline
python -m unittest discover -q
python evaluate.py compare runs/baseline/report.json runs/candidate/report.json
```

For longer work, use the workspace CPU-budget wrapper. Runs refuse existing output
directories, save requests, responses, server usage/finish reasons and wall times,
and journal completed episodes. Transport failure remains an operational error;
there are no silent retries, fabricated replies or dropped failures. Compare refuses
changed suite, harness, settings, split, repeat count or episode inventory, and refuses
operationally incomplete or changing-server runs. Candidate model weights can change.
Record whether engine, template, tokenizer, precision or hardware also changed before
attributing a score difference to training. The model filename plus a file hash does
not prove what bytes the live server loaded; `--server-identity` is an operator label.

### Server provenance

An ember server with load-time provenance (ember `docs/USAGE.md`, "Provenance") reports in
`/health` the SHA-256 of its `.gguf` file as hashed at start, checked against the loader's
descriptor after the load and, on POSIX, re-checked at every `/health` (`modified_since_load`
once the mapped file changes); its build and executable SHA-256; its `EMBER_*` settings; and an
id new to each process. `provenance.py` uses that without changing the frozen instruments
(their bytes, and so the v0.0.2 manifest's hash of `evaluate.py`, stay as they were):

```powershell
python provenance.py run evaluate -- --weights MODEL.gguf --server-identity ID --candidate NAME --out runs/NAME
python provenance.py verify runs/NAME
python provenance.py compare runs/BEFORE runs/AFTER
```

`run` (all three instruments) refuses before writing anything when the server is unreachable,
when its digest differs from the local file's, or when it reports none or runs with
`EMBER_TEST_SCRIPTED_REPLIES` or `EMBER_IR` (`--allow-unverified` records such a run as
unverified). It queries `/health` again at the end and writes `runs/NAME/provenance.json`; a
restart, swap or file modification mid-run is `changed` (or `mismatch` when the new weights
differ). `verify` reads saved receipts only, and a check can only make a run's status worse:
receipts from servers without a digest, including every run before 13 November 2026, are
`unverified`, never `verified`. `compare` works on `evaluate.py` runs only; it refuses
`mismatch` and `changed` runs, unverified ones unless `--allow-unverified`, and two servers
whose build, executable or settings differ unless `--allow-engine-change`. A public server
(`EMBER_PUBLIC=1`) reports no paths, compiler or `EMBER_*` settings: its runs still verify on the
weights digest, but runtime settings cannot be compared against it (compare says so). The digest is the
server's own report: it catches the wrong file, a stale server, a mid-run restart and changed
settings, not a server that misreports, nor a rewrite that keeps a file's size, modification
time and inode.

The runner is local HTTP only. Client wall time is diagnostic, not a warmed hardware
speed benchmark. Verify token usage/context capacity before interpreting long-history
failures: this pilot's prompts are short, but silent server truncation is not a memory
capability result. `context_sweep.py` is that separate instrument: it plants a fact,
adds F neutral filler exchanges, asks for the fact, and marks the point where the
server's reported prompt length falls short of the exactly calibrated untruncated
length, which is where history was dropped. It is diagnostic, not a release gate:
`python context_sweep.py run --weights MODEL.gguf --server-identity ID --candidate NAME --out runs/NAME-context`.

## Langfuse and optional Gemini judgment

Optional versions are in requirements-optional.txt. Use the explicitly designated
instrumentals environment file, or its already configured workspace-root fallback;
never copy secrets into this repo. Read only the needed keys at runtime.

```powershell
python integrations.py publish --env-file ../.env --report runs/baseline/report.json --name sfull-baseline
python integrations.py judge --env-file ../.env --report runs/baseline/report.json --out runs/baseline/judgment.json --model gemini-3.1-pro-preview --max-calls 4 --limit 4
python integrations.py publish --env-file ../.env --report runs/baseline/report.json --judgments runs/baseline/judgment.json --name sfull-baseline-judged
```

Publication uploads synthetic episodes and selected hashes, not health paths, private
conversations or credentials. The Langfuse namespace includes the frozen suite hash.
Its experiment replays saved outputs and scores; it does not call the model again.
Judge calls have an explicit count cap and 1,024-output-token cap; no retry. This is a
call/token bound, not a promised dollar price. Actual exposed usage and errors are saved.
Selection is fixed: open conversations first, then case ID, first repeat only. The judge
sees neither candidate identity nor deterministic scores. Holdout runs require
--split holdout --allow-holdout and should be reserved for release decisions.

Langfuse API and evaluation references:
- https://langfuse.com/docs/evaluation/get-started/offline
- https://langfuse.com/docs/evaluation/core-concepts
- https://python.reference.langfuse.com/langfuse

## Iteration and stopping design

1. Preserve this S-full baseline and the existing chat model. Never train on suite.json,
   held-out answers or evaluation transcripts. Synthetic *training* conversations need
   separate provenance, generation templates and source/content deduplication.
2. Change one training factor per exploratory comparison where practical. Keep matched
   checkpoints, data budget, seed, export and decoding settings. Report changes together
   when isolating them is impossible. Token loss alone cannot select a chat release.
3. Use development scenarios for rapid engineering and fresh, independently generated
   transfer families for confirmation. Freeze added cases, scorer and thresholds *before*
   looking at the new candidate's responses; version the instrument instead of moving gold.
4. A promising development result earns a held-out check and a blinded paired human/judge
   review. Judge order must be counterbalanced for a paired release study; this MVP's
   single-model ratings are not such a study. Cluster uncertainty by scenario/family,
   not by identical greedy repeats. The current pilot is too small for significance claims.
5. Before a confirmatory campaign, register a minimum useful improvement, recall and
   correction floors, a tolerated regression bound, completeness/repeatability criteria,
   a finite candidate budget and a futility rule. Set these from the chosen user task
   and a fresh calibration set, not from whichever candidate happens to win this pilot.
   No arbitrary 'three flat runs means optimal' rule; no one-number quality average.

First optimise reliable short conversation. Then select one bounded useful job, such
as asking about a small technical manual with cited evidence and an 'unknown' option.
Evaluate raw model, retrieval with an oracle answer, and model+retrieval separately.
Measure retrieval recall, supported-answer accuracy, citation support and refusal when
information is absent. RAG can supply facts; it cannot establish conversational memory
or repair an incoherent model merely by being attached. The instrument exists
(`rag_eval.py`, `manual_v001.json`: a fictional 20-section device manual, 15 dev and
11 holdout questions, 7 of them absent-answer, and a stdlib BM25 retriever), so it is
frozen before any candidate is read. It is a diagnostic, not a release gate, and
training for it waits until short conversation is reliable.
`python rag_eval.py retrieval` reports retrieval recall without a server.
In the raw condition the device is fictional, so 'unknown' is the correct reply to every
question, yet the summary files a raw 'unknown' to an answerable question under
`false_refusal`: read raw `false_refusal` as correct abstention, not as an error.

## First baseline and review (3 November 2026)

The preserved S-full SFT1500 CPU server completed 26 development episodes,
with all 13 complete-conversation repeat pairs identical. None of the 11 scored
cases passed its strict answer/format check. The summary's 22 checks count each
case's two identical greedy repeats, which are not independent evidence, so the
unit is 11 cases. That is not zero recall: Nora and blue survived
in replies that violated the requested answer format; delayed Basil became Ted.
Keep the raw replies when interpreting these scores. Four separately recorded
Gemini pilot judgments suggest sentence fluency can coexist with poor relevance;
these selected cases are not an overall ability estimate. Holdout remains unused.

That paragraph describes the S-full pilot, not later models. From 3 to 11 November
2026, 71 screening models trained in the parent workspace were read on
`suite_release_v002.json`'s dev split with this harness, decoding, two repeats and
generated history; every run completed, with identical repeats. The most any passes
is 4 of 11 scored cases (five models, among them S69 and S70: `name_now`,
`meeting_corrected`, `colour_corrected`, `format_only`); the latest, S71s2 on 11 November,
passes 3 of 11. Delayed recall passed strictly only for three early screens
(S12, S24 and S26: `pet_delayed`, "Basil"), and no model read after S26 on 5 November
passes a delayed-recall, basic Q&A or missing-information case. These are development reads of screening models,
not release candidates, and the holdout has still never been run. Their receipts
are local to that workspace.

**What a strict miss means.** These checks pass only an exact, bare answer: the
right fact inside a longer sentence, with extra words, or with a hedge fails, and so
does a reply that is right in content but wrong in form. A strict score is therefore
a floor on content, not a measure of it, and a low one does not say whether the model
lacked the fact or only failed the format. The judged and rule-scored reads of the
same models moved much further than the strict checks: on the parent programme's
13-turn judged relevance read, S-full SFT1500 scored 19 of 52 and S71 36; on its
told-recall rule, 7 of 108 and 62. Those are different instruments with their own
protocols, recorded in paperkiln's `docs/CAPABILITY.md`; they are not comparable
with this suite's counts. Two limits of this suite also bear on any reading: the
three "Have I told you…?" cases all have gold "no", so a model that always answers
"No." passes them; and repeats are greedy and identical, so two repeats are one
observation, not two.

**How little these models were trained.** The models are small (72,761,088
parameters) and saw little data for their size. S-full's base saw 163.84M positions,
about 2.3 tokens per parameter; base A, under every model from 9 November, saw
983.04M, about 13.5. That is near the order of a compute-optimal budget (roughly 20
tokens per parameter) but far below current small open models: SmolLM2-135M was
trained on about 2 trillion tokens, roughly 15,000 per parameter. Low strict scores
should be read in that light. The next base (base 2, 18 × 512, in training from
13 November 2026) is registered at 250,000 steps of 16,384 positions, about 59
tokens per parameter; nothing is claimed for it until it is evaluated.

**An open small model on the same fine-tuning (A29, 11 November 2026).** The parent
programme put S71's fine-tuning packet and recipe on SmolLM2-135M, an open model
pretrained on far more text, to ask whether its own base limits relevance. This was
not run on this repository's suites: it was read on the programme's wide judged
relevance instrument (532 planned turns, one judge, the same plan for every row;
a mean judged score of 0 to 4 per turn times 13, so 0 to 52; higher is better):

| Model | Wide judged relevance (points) |
|---|---:|
| S70, base A | 30.6 |
| S71, base A (the programme's best model of its own) | 31.7 |
| SmolLM2-135M-Instruct as released | 24.5 |
| SmolLM2-360M-Instruct as released | 28.3 |
| S71's packet on SmolLM2-135M, seed 1 | 37.0 (+5.3 over S71, 95% +3.2 to +7.4) |
| the same, seed 2 | 36.4 (+4.7, 95% +2.6 to +6.9) |

The open-base arm read higher at both seeds (verdict `base_up`). It changed the base,
size (135M against 73M parameters), tokeniser, architecture, pretraining, the authors'
instruction tuning, trainer and engine together, so it shows that the package helps,
not which ingredient does. Its replies were generated by `transformers` in bf16 on a
GPU, ours by ember on the CPU; the reply cap counts each model's own tokens, and
SmolLM2's larger vocabulary fits more text into the cap; the two released models hit
the cap on most turns, so their reads carry truncation. The programme does not adopt
an open base as its model (the owner's ruling, 11 November 2026); A29 is a diagnostic,
and it is why base 2 exists. Record: the parent workspace's `PROTOCOL.md` section 45;
summary in paperkiln's `docs/CAPABILITY.md`.

Original local receipts: runs/sfull-baseline-20261103/. Langfuse contains the
original baseline and a separately named judged experiment. Review found malformed
HTTP-200 response shapes could abort the original runner before recording an
episode. The repaired runner preserves malformed JSON response shapes and final-health
failures in its report; six offline tests pass. An HTTP-200 body that is not JSON is
still recorded only as a JSONDecodeError operational error, without its bytes. A fresh baseline under the repaired
harness lives in runs/sfull-baseline-fixed-20261103/. Original receipts and judge
ratings retain their original hashes and are not relabelled as repaired-run data.

Runs are ignored local artifacts, not committed training material. Repository: https://github.com/JohnnyTeutonic/tinychat-eval.

## Licence

MIT; see [LICENSE](LICENSE). The licence covers the code and the data files alike.
One request that is not a licence term: please do not train models on the suites, the
manual, the context-sweep probes or recorded evaluation conversations. Training on
them makes these instruments useless for the models trained on them.

## v0.0.2 preparation

See [V002_RELEASE.md](V002_RELEASE.md) for the controlled training matrix,
proposed release gates, bounded compute, rollback and verified CLI forms.
[evaluation_manifest_v002_final.json](evaluation_manifest_v002_final.json) binds
the new instrument and proposal; validate with `python freeze.py --check
evaluation_manifest_v002_final.json`. The original pilot suite/results stay unchanged.
New reference-based semantic ratings and per-turn ratings are optional separate
diagnostics; no new cloud judging or training was launched during preparation.
[GATE_ACCEPTANCE_V002.json](GATE_ACCEPTANCE_V002.json) accepts the unchanged draft
criteria, recorded on 3 November 2026 before any candidate output was read. Its
judgment gates stay blocked until the calibration requirement is met; the four blind
human ratings in `calibration/` are final-reply ratings, while the gate rates every
turn of its five cases, and no agreement criterion has been fixed. As of 13 November
2026 no evaluation of a v0.0.2 training candidate against these gates is recorded.
Admission prerequisites remain.

## v0.0.3 release specification

[release_gate_v003.json](release_gate_v003.json) is the release specification for
candidates evaluated from 13 November 2026, fixed before any v0.0.3 candidate was
evaluated and before any holdout run ([GATE_ACCEPTANCE_V003.json](GATE_ACCEPTANCE_V003.json)).
It keeps v0.0.2's thresholds, suite and harness, and v0.0.2's files stay unchanged. It differs in
three ways:

- Every run the verdict reads, candidate and baseline alike, must be `verified` by
  `provenance.py`. Receipts from servers without a load-time digest, including the
  S-full baseline, are not, so a baseline is rerun on a provenance-reporting server.
- Judge ratings are diagnostic: reported, never pass or fail (Jonathan, 13 November 2026).
- Each threshold says what it is a fraction of: `dev_exact_min` is taken over the scored
  development cases (11 of 13), the family and holdout criteria over their scored cases,
  and a case passes when every repeat passes. Claude Opus 5.5 chose this on 13 November
  2026, because unscored cases cannot pass; Jonathan may override it in a later version.

It states the missing-information limit above; a suite version with matched 'yes' cases
is proposed there and not built. Evaluate a candidate offline with
`python gate_v003.py --candidate-dev runs/C --baseline-dev runs/B --candidate-holdout runs/CH
--baseline-holdout runs/BH --export-parity passed` (exit 0 pass, 1 fail, 2 incomplete), and
check the binding with `python freeze_v003.py --check evaluation_manifest_v003.json`.

## Candidate suite v0.0.3: what changes and why (CANDIDATE, for Jonathan's approval)

Proposed by Claude Opus 5.5 on 13 November 2026, at Jonathan's request ("we do need to
improve tinychat-eval"). Nothing here is frozen or accepted. No gate reads it, and no
model has been evaluated on it. The v0.0.2 and v0.0.3 files, suites, gold and receipts
are unchanged. The v0.0.3 gate keeps reading `suite_release_v002.json`.

**Gate version.** Adopting this suite would need its own v0.0.4 release specification,
manifest and acceptance record, fixed before any candidate is evaluated on it. A v0.0.3
verdict and a verdict under this suite are not comparable.

**What changes**

- **Twin cases replace the one-sided "Have I told you…?" cases (B1).** The family
  `told_or_not` has four matched pairs. In each pair the told twin gives the fact and the
  untold twin does not, they ask the same final question, and the two differ in exactly
  one user message. The told twin's gold is `yes` and the untold twin's is `no`. Two
  pairs ask at once and two after filler turns. A constant answer is right on exactly
  half the family. The two v0.0.2 cases are the untold halves of two of the pairs.
- **More cases per family: six scored development cases where v0.0.2 had one or two.**
  `told_or_not` has eight (four pairs).
  - Why six: models trained on the same recipe with different seeds differ by one to
    two of the 11 v0.0.2 cases (S67 3 against S67s2 1; S71 1 against S71s2 3; S72 2
    against S72s2 3), and single cases flip between seeds.
  - With two cases a family moves in 50-point steps, so one seed flip decides it. With
    six it moves in 17-point steps.
  - More would cost run time, below.
- **A cue-sensitivity diagnostic** (`suite_v003_candidate_cue.json`), never a release
  criterion. It has four hand-written triplets:
  - a base conversation A;
  - a decisive twin B whose edit changes the gold reply;
  - a nuisance twin N whose edit leaves it unchanged.

  A triplet is selective when A, B and N are all answered correctly. The design follows
  the lane's scenario engine 2.1, but nothing is copied from it: no code, output or seed
  stream. The triplets share no 8-word span with that engine's 20,000-triplet training
  sample, and no name with its name banks (`suite_v003_candidate_audit.json`). This makes
  them the out-of-engine check the cue-selective plan asks for.
- **Strict and content-correct are reported separately** (`score_v003.py`, offline, on
  the saved report of the unchanged `evaluate.py`).
  - **Strict** is `evaluate.py`'s exact contract.
  - **Content** is a small per-case rule: a required phrase; forbidden phrases, which
    are excused after "not" or "no longer"; and, for yes/no cases, the polarity.
  - Forbidden phrases include each case's stale value, contradictory alternatives and
    role reversals such as "My name is Nora."
  - The unit is the distinct case: a case passes only if every repeat passes. Families,
    twin pairs and triplets are reported as "k of n".
  - The rules are lexical, so transcripts stay the arbiter of meaning. A rule cannot
    tell "I like the colour blue" from an answer about the bicycle.
- **The split is rebuilt, and the holdout is sealed.**
  - Development: 40 cases. These are the 13 v0.0.2 development cases, with their wording
    and strict gold unchanged and a content rule added, plus 27 new ones.
  - The public v0.0.2 holdout is left out, so the v0.0.3 gate's holdout stays unread.
  - Holdout: 24 new cases, four per scored family and two twin pairs. Only its SHA-256
    and its counts are in the repository (`suite_v003_candidate_holdout.seal.json`).
  - The content is outside the repository: `C:/ml_artifacts/tinychat_eval_sealed/`.
    It is not encrypted, and it has no second copy yet.
  - It shares no message and no 8-word span with any suite or instrument text here, or
    with the engine sample.

**Cases.** Development: `basic_qa` 6, `immediate_recall` 6, `delayed_recall` 6,
`correction` 6, `told_or_not` 8, `instruction` 6, and `open_conversation` 2 (unscored).
Holdout: four in each of the six scored families. Cue diagnostic: 12 cases (four triplets).

**Run time.** Counting one request per user turn:
- v0.0.2 development is 29 requests a repeat, 58 at the usual two repeats.
- The candidate development set is 90 requests a repeat: about 1.5 times today's cost at
  one repeat, three times at two.
- The cue diagnostic adds 24, and the holdout is 54 a repeat.

Repeats have been identical in all 71 recorded runs, so the proposal is one repeat for
development screening and two at release.

**Use**

```powershell
python evaluate.py run --suite suite_v003_candidate_dev.json --repeats 1 --weights MODEL.gguf --server-identity ID --candidate NAME --out runs/NAME-v003c
python score_v003.py runs/NAME-v003c/report.json --suite suite_v003_candidate_dev.json
python score_v003.py --validate suite_v003_candidate_dev.json suite_v003_candidate_cue.json
```
