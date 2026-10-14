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
