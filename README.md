# QuanBench Plus

LLM evaluation pipelines for quantum coding tasks across `cirq`, `qiskit`, and `pennylane`.

This fork supports two **generation providers**:

- `openrouter` — the original behaviour. Generates code via the OpenRouter chat-completions API.
- `coda` — sends prompts to the [Coda agent](https://docs.conductorquantum.com/api-reference/coda-api/agents/run) (`POST /v0/coda/agents`) and benchmarks Coda as the code generator.

The **target framework** (`--framework cirq | qiskit | pennylane`) is independent of the provider. The provider decides who writes the code; the framework decides which quantum SDK runs and grades it.

## 1) Setup

Run all commands from the repository root. Python `3.10+` is required.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

`pip install -e .` uses the project configuration from `pyproject.toml`.

## 2) Configure API keys

Create `.env` in the repo root with whichever provider you plan to use:

```env
# OpenRouter (original behaviour)
API_KEY=your_openrouter_api_key

# Coda agent
CODA_API_KEY=your_coda_bearer_token
# (CONDUCTOR_API_KEY is also accepted as an alias, matching the `conductorquantum` SDK convention.)

# Optional Coda overrides
# CODA_API_BASE_URL=https://api.conductorquantum.com/v0/coda
# CODA_AGENT_MODE=build              # 'build' (default) or 'learn'
# CODA_AGENT_FAST=false              # 'true' to use Coda's fast model
# CODA_AGENT_PAYLOAD_WRAPPER=direct  # 'direct' (default, matches OpenAPI) or 'body'
# CODA_PREFER_STRUCTURED_RESPONSE=0  # default 0 = use streamed tokens (LLM's
#                                    #   raw function-form output). Set 1 to use
#                                    #   Coda's post-pipeline transpiled output
#                                    #   from `structured_response.data.code`.
# CODA_TARGET_FRAMEWORK=qiskit       # only used when CODA_PREFER_STRUCTURED_RESPONSE=1;
#                                    #   the pipelines auto-set this from --framework.
# CODA_MAX_RETRIES=4                 # transparent exponential-backoff retries on
# CODA_RETRY_BASE_DELAY=1.0          #   transient errors (502/503/504/429/408/500/303
# CODA_RETRY_MAX_DELAY=30.0          #   and connection-level failures); honours
#                                    #   Retry-After when present.
```

Notes:

- `API_KEY` is only needed when `--provider openrouter` is used.
- `CODA_API_KEY` is only needed when `--provider coda` is used.
- `CODA_AGENT_PAYLOAD_WRAPPER=body` is an escape hatch in case the Coda gateway temporarily requires the `{"body": ...}` wrapper that some Fern-generated SDK examples show. The default `direct` matches the published OpenAPI schema and the Python SDK.

## 3) Run Pass@k pipeline

```bash
# OpenRouter
bash pass_at_k_pipeline/runner.sh --framework cirq --pass_k 5 "openai/gpt-4.1"
bash pass_at_k_pipeline/runner.sh --framework qiskit --pass_k 5 "openai/gpt-4.1"
bash pass_at_k_pipeline/runner.sh --framework pennylane --pass_k 5 "openai/gpt-4.1"

# Coda agent (default model label = coda/build)
bash pass_at_k_pipeline/runner.sh --provider coda --framework qiskit --pass_k 1 coda/build
bash pass_at_k_pipeline/runner.sh --provider coda --framework cirq --pass_k 1 coda/build
bash pass_at_k_pipeline/runner.sh --provider coda --framework pennylane --pass_k 1 coda/build
```

## 4) Run Feedback-loop pipeline

```bash
# OpenRouter
bash feedback_loop/runner.sh --framework cirq --feedback_num 5 "openai/gpt-4.1"
bash feedback_loop/runner.sh --framework qiskit --feedback_num 5 "openai/gpt-4.1"
bash feedback_loop/runner.sh --framework pennylane --feedback_num 5 "openai/gpt-4.1"

# Coda agent
bash feedback_loop/runner.sh --provider coda --framework qiskit --feedback_num 5 coda/build
bash feedback_loop/runner.sh --provider coda --framework cirq --feedback_num 5 coda/build
bash feedback_loop/runner.sh --provider coda --framework pennylane --feedback_num 5 coda/build
```

## 5) Smoke-test before a full Coda run

The Coda agent runs a full validation + multi-framework transpilation pipeline per request, so each call takes ~30s (much slower than a raw chat completion). A full 42-task Pass@1 run takes ~20 minutes; cross-framework Pass@1 takes ~60 minutes. Verify the integration on a single task first:

```bash
# One prompt directly through the provider, prints normalized response.
# Exits 0 on success, 1 if the response carries an error envelope, so this
# is safe to chain with `&&` in setup scripts.
python -m utils.providers --provider coda \
  --prompt "Write a Python function that returns a Qiskit Bell state circuit. Reply with code only."

# Show the raw Coda events too (useful for parser debugging):
python -m utils.providers --provider coda --show-events --prompt "..."

# Or use the pipeline with --limit 1
bash pass_at_k_pipeline/runner.sh --provider coda --framework qiskit --pass_k 1 --limit 1 coda/build
```

## 6) Run the full Coda sweep unattended

`scripts/run_full_coda_benchmark.sh` runs Pass@1, Pass@5, and the feedback loop across all three frameworks sequentially, and writes per-phase logs plus a running `SUMMARY.md` to `logs/coda_<timestamp>/`. A single phase failure does not abort the rest of the sweep.

```bash
# Detached run that survives a closed terminal and prevents the laptop from sleeping.
screen -dmS coda-bench bash -lc \
  'caffeinate -i bash scripts/run_full_coda_benchmark.sh'

# Reattach later
screen -r coda-bench

# Live tail of the high-level timeline
tail -f logs/coda_*/run.log
```

Tunable via env vars: `MODEL_LABEL`, `FRAMEWORKS`, `RUN_PASS1`, `RUN_PASS5`, `RUN_FEEDBACK`, `FEEDBACK_NUM`, `PASS5_K`, `LOG_DIR`.

## 7) Where results are saved

Result paths are keyed off the **model label** you pass on the command line, so Coda runs land in their own files (`coda_build_*.json`):

- Pass@k raw model outputs: `model_responses/<framework>/pass_at_*/*.json`
- Pass@k parsed responses: `responses/<framework>/pass_at_*/*.json`
- Pass@k evaluated results: `results/<framework>/pass_at_*/*.json`
- Feedback attempts + final state: `model_responses/<framework>/feedback_loop/*_attempts.json` and `*_final.json`

## 8) Tests

Unit tests for the provider layer (SSE parser, request projection, error envelopes) live in `tests/test_providers.py`. They use mocked HTTP responses, so no network or API key is required.

```bash
pip install pytest
pytest
```

## Notes

- For `--provider openrouter`, model names must be OpenRouter IDs (example: `"openai/gpt-4.1"`).
- For `--provider coda`, the underlying API has no model selector; the model argument is just a label used for filenames and grouping. Keep it as `coda/build` (or `coda/build-fast`) so it stays distinct from OpenRouter rows.
- Use `--help` for CLI options:
  - `bash pass_at_k_pipeline/runner.sh --help`
  - `bash feedback_loop/runner.sh --help`
- Coda is benchmarked as the **generator**; `--framework` still chooses the target quantum SDK whose probability distribution the generated code is graded against.
