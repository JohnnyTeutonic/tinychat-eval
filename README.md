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
`/health` the SHA-256 of the weights it hashed and loaded, its build (git commit, dirty state,
backend, compiler), its executable's SHA-256 and an id new to each process. `provenance.py`
uses that without changing the frozen instruments (their hashes and the v0.0.2 manifest stay valid):

```powershell
python provenance.py run evaluate -- --weights MODEL.gguf --server-identity ID --candidate NAME --out runs/NAME
python provenance.py verify runs/NAME
python provenance.py compare runs/BEFORE runs/AFTER
```

`run` refuses before writing anything when the server's digest differs from the local file's,
or when the server reports none (`--allow-unverified` records such a run as unverified). It
queries `/health` again at the end and writes `runs/NAME/provenance.json`; a restart or swap
mid-run is `changed`. `verify` reads saved receipts only: receipts from servers without a
digest, including every run before 13 November 2026, are `unverified`, never `verified`.
`compare` refuses `mismatch` and `changed` runs, and unverified ones unless
`--allow-unverified`. The digest is the server's own report: it catches the wrong file, a stale
server and a mid-run restart, not a server that misreports.

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

## First baseline and review (3 November 2026)

The preserved S-full SFT1500 CPU server completed 26 development episodes,
with all 13 complete-conversation repeat pairs identical. None of the 22 strict
answer/format checks passed. That is not zero recall: Nora and blue survived
in replies that violated the requested answer format; delayed Basil became Ted.
Keep the raw replies when interpreting these scores. Four separately recorded
Gemini pilot judgments suggest sentence fluency can coexist with poor relevance;
these selected cases are not an overall ability estimate. Holdout remains unused.

Original local receipts: runs/sfull-baseline-20261103/. Langfuse contains the
original baseline and a separately named judged experiment. Review found malformed
HTTP-200 response shapes could abort the original runner before recording an
episode. The repaired runner preserves malformed raw responses and final-health
failures in its report; six offline tests pass. A fresh baseline under the repaired
harness lives in runs/sfull-baseline-fixed-20261103/. Original receipts and judge
ratings retain their original hashes and are not relabelled as repaired-run data.

Runs are ignored local artifacts, not committed training material. Repository: https://github.com/JohnnyTeutonic/tinychat-eval.
No licence has been selected yet.

## v0.0.2 preparation

See [V002_RELEASE.md](V002_RELEASE.md) for the controlled training matrix,
proposed release gates, bounded compute, rollback and verified CLI forms.
[evaluation_manifest_v002_final.json](evaluation_manifest_v002_final.json) binds
the new instrument and proposal; validate with `python freeze.py --check
evaluation_manifest_v002_final.json`. The original pilot suite/results stay unchanged.
New reference-based semantic ratings and per-turn ratings are optional separate
diagnostics; no new cloud judging or training was launched during preparation.
Release specification acceptance and admission prerequisites are still required.
