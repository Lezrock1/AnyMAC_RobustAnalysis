# Isolated development diagnostics

Run only in a **separate GPU allocation**, not in a running research terminal:

```bash
bash scripts/diagnostics/run.sh
```

The runner uses the existing Ramses environment, a new results directory and port
11436. It refuses to reuse a server on that port, starts its own server without
detachment, and stops only its own server. It does not modify code, checkpoints,
splits, existing results or the running pilot. Request scheduling is limited to
eight hours; use a ten-hour allocation for startup/shutdown headroom. Closing the
interactive terminal can terminate the job.

The suite uses the first ten development tasks, never evaluation tasks or truth
in prompts. The 480-request matrix crosses server slots (1/12), client requests
(serial/three parallel), prompt interface (original flattened `/api/generate`
versus structured `/api/chat`) and output limit (1536/3072), with three repeats.
Context size is 8192, the role is Quantitative Reasoner and the marker is blue.
The server disables Flash Attention and custom KV-cache type to match the 4B
pilot's default configuration. Conditions run on the same allocated GPU; requests
use seed 42, temperature 0 and the original generation options.

`REPORT.txt` is updated after every request. `summary.json`, `manifest.json`,
`responses.jsonl` and server logs retain the detailed evidence. Responses include
the complete request and backend output, strict final-line answer, reference
answer for scoring only, digest, truncation and errors. Errors are not retried or
silently replaced. A budget-exhausted run is explicitly marked; partially
completed cells must not be treated as complete experiments.

The report also reads existing pilot JSONL files for invalid lines, duplicate
rounds/episodes and errors, and evaluates the actual router source slices on
labelled tokens. The source probe does **not** execute the real transformer: it demonstrates
index mapping and the gradient loop boundary, not their effect on trained
policies. A separate small probe executes the actual gradient method with
synthetic embeddings and a synthetic transformer, comparing gradients after
flipping non-final versus final context masks. It needs the existing torch
installation but no model downloads; missing torch is explicitly reported.
No implementation fixes are applied.

These are exploratory diagnostics, not attack evidence or significance tests.
Fixed condition order, cache state and server restarts can confound comparisons.
Strict parsing intentionally differs from the pipeline's fallback to the last
integer. Stable outputs need not be correct; identical parsed numbers can hide
different texts. Report raw counts including errors and incomplete answers.

Local validation (standard library only):

```bash
python -m unittest discover -s scripts/diagnostics -p 'test_diagnose.py' -v
bash -n scripts/diagnostics/run.sh
```

## Extended 24-hour allocation

```bash
bash scripts/diagnostics/run.sh --extended
```

This creates an **extended** result folder and uses a 22-hour request deadline
inside a separately requested 24-hour GPU allocation. Existing diagnostics are
not overwritten. Only the already installed 4B model is used; no downloads or
requests to the running 30B server are made.

The fixed design is ten dev tasks, three repeats, two prompt formats, two token
limits, three conditions (one slot/serial, twelve slots/serial, twelve
slots/three parallel requests), and three fresh-server blocks: **1,080 requests**.
Server order alternates 1-12 / 12-1 / 1-12; serial/parallel order alternates in
the twelve-slot blocks. Task/format/limit groups are deterministically shuffled
per block, identically across conditions. This reduces, but does not eliminate,
order/cache confounding. Budget exhaustion may leave a partial matrix; there is
no guarantee all requests fit in 22 hours.

The runner stores backend duration counters, wall times, request deadlines,
strict answers and complete raw replies. Errors are explicit, not retried.
Client timeouts have a 1,200-second limit (capped by the remaining budget).
After a timeout the current server part ends so that uncancelled backend
generation cannot contaminate later requests on that server. Subsequent blocks
start fresh. `REPORT.txt` distinguishes regular and budget-limited timeouts,
complete task repeats, and same-task/repeat differences across server starts.
Metadata captures model digests and Ollama versions; model changes abort the run.
This is still exploratory, not a controlled cache ablation or a power study.

Read-only audits compare the real pipeline parser against strict final-line
parsing and store up to four deterministically chosen disagreement texts per
evaluation file in `parser_samples.json` for manual review. Neither parser is
declared the semantic ground truth. No old answers or rewards are rewritten.

The synthetic gradient probe compares original code, corrected token indices,
and corrected indices plus inclusion of an **early non-forced decision** in the
gradient loop, only in memory. A positive gradient difference validates this
synthetic path, not a production patch: forced stopping, masked actions,
single-step traces and policy-gradient correctness need separate regression
tests before any real retraining.

```bash
python -m unittest discover -s scripts/diagnostics -p 'test_*.py' -v
```
