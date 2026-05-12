#!/usr/bin/env bash
# Run the full QuanBench+ sweep (Pass@1, Pass@5, feedback loop) across the
# selected target frameworks (qiskit, cirq, pennylane) sequentially, against
# any of the supported generation providers (openrouter, coda, bedrock).
#
# Logs each phase to its own file under $LOG_DIR (default:
# logs/<provider>_<timestamp>/) and writes a running SUMMARY.md with timing
# and exit codes so you can tail the directory and immediately see status.
#
# Designed to be invoked under `screen` + `caffeinate -i` for unattended runs.
# A single phase failure does NOT abort the rest of the sweep; the summary
# tracks which phases failed.
#
# Examples
#   PROVIDER=coda    MODEL_LABEL=coda/build-clean    bash scripts/run_full_benchmark.sh
#   PROVIDER=bedrock MODEL_LABEL=bedrock/opus-4-6    bash scripts/run_full_benchmark.sh
#   PROVIDER=bedrock FRAMEWORKS="qiskit cirq" RUN_PASS5=0 bash scripts/run_full_benchmark.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# --- Config (overridable via env) -------------------------------------------
PROVIDER="${PROVIDER:-coda}"
case "$PROVIDER" in
  openrouter|coda|bedrock) ;;
  *)
    echo "Error: unknown PROVIDER='$PROVIDER'. Use openrouter | coda | bedrock." >&2
    exit 1
    ;;
esac

# Sensible default model label per provider so a bare invocation just works.
case "$PROVIDER" in
  coda)       DEFAULT_MODEL_LABEL="coda/build" ;;
  bedrock)    DEFAULT_MODEL_LABEL="bedrock/opus-4-6" ;;
  openrouter) DEFAULT_MODEL_LABEL="anthropic/claude-3-5-sonnet" ;;
esac
MODEL_LABEL="${MODEL_LABEL:-$DEFAULT_MODEL_LABEL}"

FRAMEWORKS_RAW="${FRAMEWORKS:-qiskit cirq pennylane}"
read -r -a FRAMEWORKS <<< "$FRAMEWORKS_RAW"
RUN_PASS1="${RUN_PASS1:-1}"
RUN_PASS5="${RUN_PASS5:-1}"
RUN_FEEDBACK="${RUN_FEEDBACK:-1}"
FEEDBACK_NUM="${FEEDBACK_NUM:-5}"
PASS5_K="${PASS5_K:-5}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/${PROVIDER}_${TIMESTAMP}}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/SUMMARY.md"
LIVE_LOG="$LOG_DIR/run.log"

# --- Helpers ----------------------------------------------------------------
say() {
  printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LIVE_LOG"
}

format_duration() {
  local s="$1"
  printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

record_phase() {
  local phase="$1" framework="$2" log="$3" status="$4" duration_s="$5"
  printf '| %s | %s | %s | %s | %s |\n' \
    "$phase" "$framework" "$status" "$(format_duration "$duration_s")" "$(basename "$log")" \
    >> "$SUMMARY"
}

run_phase() {
  local phase="$1"
  local framework="$2"
  shift 2
  local log_path="$LOG_DIR/${phase}_${framework}.log"

  say ">>> START $phase / $framework"
  say "    log: $log_path"
  local t0 t1 dur status
  t0=$(date +%s)
  if "$@" > >(tee "$log_path") 2>&1; then
    status="OK"
  else
    status="FAIL"
  fi
  t1=$(date +%s)
  dur=$((t1 - t0))
  say "<<< END   $phase / $framework  status=$status  duration=$(format_duration $dur)"
  echo "" >> "$LIVE_LOG"
  record_phase "$phase" "$framework" "$log_path" "$status" "$dur"
}

# --- Pre-flight -------------------------------------------------------------
{
  echo "# QuanBench+ benchmark sweep — $TIMESTAMP"
  echo
  echo "- repo: \`$REPO_ROOT\`"
  echo "- provider: \`$PROVIDER\`"
  echo "- model label: \`$MODEL_LABEL\`"
  echo "- frameworks: ${FRAMEWORKS[*]}"
  echo "- phases: pass1=$RUN_PASS1 pass5=$RUN_PASS5 feedback=$RUN_FEEDBACK (feedback_num=$FEEDBACK_NUM)"
  echo "- python: $(python -V 2>&1)"
  echo "- pid: $$"
  echo
  echo "## Phase results"
  echo
  echo "| Phase | Framework | Status | Duration | Log |"
  echo "| --- | --- | --- | --- | --- |"
} > "$SUMMARY"

say "Starting $PROVIDER sweep. log_dir=$LOG_DIR model_label=$MODEL_LABEL"
say "frameworks=${FRAMEWORKS[*]} pass1=$RUN_PASS1 pass5=$RUN_PASS5 feedback=$RUN_FEEDBACK (n=$FEEDBACK_NUM)"

say "Pre-flight: provider auth smoke (1 short prompt)..."
PRE_LOG="$LOG_DIR/preflight_provider.log"
PREFLIGHT_ARGS=(--provider "$PROVIDER" --prompt "Reply with the single word OK.")
if [ "$PROVIDER" = "openrouter" ]; then
  PREFLIGHT_ARGS+=(--model "$MODEL_LABEL")
fi
if python -m utils.providers "${PREFLIGHT_ARGS[@]}" > "$PRE_LOG" 2>&1; then
  if grep -q '"error"' "$PRE_LOG"; then
    say "FATAL: pre-flight reported an error envelope. See $PRE_LOG. Aborting."
    echo
    echo "## FATAL: pre-flight failed" >> "$SUMMARY"
    echo "See \`preflight_provider.log\`." >> "$SUMMARY"
    exit 2
  fi
  say "Pre-flight OK."
else
  say "FATAL: pre-flight smoke failed (non-zero exit). See $PRE_LOG. Aborting."
  echo
  echo "## FATAL: pre-flight failed" >> "$SUMMARY"
  echo "See \`preflight_provider.log\`." >> "$SUMMARY"
  exit 2
fi
echo "" >> "$LIVE_LOG"

SWEEP_T0=$(date +%s)

if [ "$RUN_PASS1" = "1" ]; then
  for fw in "${FRAMEWORKS[@]}"; do
    run_phase "pass1" "$fw" \
      bash pass_at_k_pipeline/runner.sh \
        --provider "$PROVIDER" --framework "$fw" --pass_k 1 "$MODEL_LABEL"
  done
fi

if [ "$RUN_PASS5" = "1" ]; then
  for fw in "${FRAMEWORKS[@]}"; do
    run_phase "pass${PASS5_K}" "$fw" \
      bash pass_at_k_pipeline/runner.sh \
        --provider "$PROVIDER" --framework "$fw" --pass_k "$PASS5_K" "$MODEL_LABEL"
  done
fi

if [ "$RUN_FEEDBACK" = "1" ]; then
  for fw in "${FRAMEWORKS[@]}"; do
    run_phase "feedback" "$fw" \
      bash feedback_loop/runner.sh \
        --provider "$PROVIDER" --framework "$fw" --feedback_num "$FEEDBACK_NUM" "$MODEL_LABEL"
  done
fi

SWEEP_T1=$(date +%s)
SWEEP_DUR=$((SWEEP_T1 - SWEEP_T0))

{
  echo
  echo "## Totals"
  echo
  echo "- wall clock: $(format_duration $SWEEP_DUR)"
  echo "- finished: $(date)"
} >> "$SUMMARY"

say "Sweep complete. wall=$(format_duration $SWEEP_DUR). Summary: $SUMMARY"

exit 0
