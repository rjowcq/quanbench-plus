#!/usr/bin/env bash
# Run the full QuanBench+ Coda sweep (Pass@1, Pass@5, feedback loop) across
# all three target frameworks (qiskit, cirq, pennylane) sequentially.
#
# Logs each phase to its own file under $LOG_DIR (default:
# logs/coda_<timestamp>/) and writes a running SUMMARY.md with timing and
# exit codes so you can tail the directory and immediately see status.
#
# Designed to be invoked under `screen` + `caffeinate -i` for unattended runs.
# A single phase failure does NOT abort the rest of the sweep; the summary
# tracks which phases failed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- Activate venv if present so `python` resolves to the project env. -------
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# --- Config (overridable via env) -------------------------------------------
MODEL_LABEL="${MODEL_LABEL:-coda/build}"
FRAMEWORKS_RAW="${FRAMEWORKS:-qiskit cirq pennylane}"
read -r -a FRAMEWORKS <<< "$FRAMEWORKS_RAW"
RUN_PASS1="${RUN_PASS1:-1}"
RUN_PASS5="${RUN_PASS5:-1}"
RUN_FEEDBACK="${RUN_FEEDBACK:-1}"
FEEDBACK_NUM="${FEEDBACK_NUM:-5}"
PASS5_K="${PASS5_K:-5}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/coda_${TIMESTAMP}}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/SUMMARY.md"
LIVE_LOG="$LOG_DIR/run.log"

# --- Helpers ----------------------------------------------------------------
say() {
  printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LIVE_LOG"
}

# Append a row to the markdown summary.
record_phase() {
  local phase="$1" framework="$2" log="$3" status="$4" duration_s="$5"
  printf '| %s | %s | %s | %s | %s |\n' \
    "$phase" "$framework" "$status" "$(format_duration "$duration_s")" "$(basename "$log")" \
    >> "$SUMMARY"
}

format_duration() {
  local s="$1"
  printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

run_phase() {
  local phase="$1"
  local framework="$2"
  shift 2
  local log_name
  log_name="$(printf '%s_%s.log' "$phase" "$framework")"
  local log_path="$LOG_DIR/$log_name"

  say ">>> START $phase / $framework"
  say "    log: $log_path"
  local t0 t1 dur status
  t0=$(date +%s)
  # Run the command, tee'ing to both the per-phase log and (with a short
  # tag) the live aggregate log. We deliberately do not propagate a non-
  # zero exit beyond this function so subsequent phases still run.
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
  echo "# Coda benchmark sweep — $TIMESTAMP"
  echo
  echo "- repo: \`$REPO_ROOT\`"
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

say "Starting Coda sweep. log_dir=$LOG_DIR"
say "frameworks=${FRAMEWORKS[*]} pass1=$RUN_PASS1 pass5=$RUN_PASS5 feedback=$RUN_FEEDBACK (n=$FEEDBACK_NUM)"

# Quick auth pre-check: hit the providers smoke path once to make sure the
# CODA/CONDUCTOR key is loaded. If this fails, do not waste hours.
say "Pre-flight: provider auth smoke (1 short prompt)..."
PRE_LOG="$LOG_DIR/preflight_provider.log"
if python -m utils.providers --provider coda --prompt "Reply with the single word OK." > "$PRE_LOG" 2>&1; then
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

# --- Pass@1 -----------------------------------------------------------------
if [ "$RUN_PASS1" = "1" ]; then
  for fw in "${FRAMEWORKS[@]}"; do
    run_phase "pass1" "$fw" \
      bash pass_at_k_pipeline/runner.sh \
        --provider coda --framework "$fw" --pass_k 1 "$MODEL_LABEL"
  done
fi

# --- Pass@5 -----------------------------------------------------------------
if [ "$RUN_PASS5" = "1" ]; then
  for fw in "${FRAMEWORKS[@]}"; do
    run_phase "pass${PASS5_K}" "$fw" \
      bash pass_at_k_pipeline/runner.sh \
        --provider coda --framework "$fw" --pass_k "$PASS5_K" "$MODEL_LABEL"
  done
fi

# --- Feedback loop ----------------------------------------------------------
if [ "$RUN_FEEDBACK" = "1" ]; then
  for fw in "${FRAMEWORKS[@]}"; do
    run_phase "feedback" "$fw" \
      bash feedback_loop/runner.sh \
        --provider coda --framework "$fw" --feedback_num "$FEEDBACK_NUM" "$MODEL_LABEL"
  done
fi

SWEEP_T1=$(date +%s)
SWEEP_DUR=$((SWEEP_T1 - SWEEP_T0))

# --- Final summary ----------------------------------------------------------
{
  echo
  echo "## Totals"
  echo
  echo "- wall clock: $(format_duration $SWEEP_DUR)"
  echo "- finished: $(date)"
} >> "$SUMMARY"

say "Sweep complete. wall=$(format_duration $SWEEP_DUR). Summary: $SUMMARY"

# Exit 0 even if some phases failed - the SUMMARY captures status. Use the
# `grep FAIL "$SUMMARY"` exit code if a CI workflow needs strict propagation.
exit 0
