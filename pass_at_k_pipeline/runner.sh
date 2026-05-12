PASS_K=1
FRAMEWORK="cirq"
PROVIDER="openrouter"
LIMIT=""
MODELS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

PIPELINE_DIR="$REPO_ROOT/pass_at_k_pipeline"
API_SCRIPT="$PIPELINE_DIR/api_pass_at_k.py"

# Sanity: python exists
if [ ! -x "$PYTHON_BIN" ] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: could not find a Python interpreter (looked for $PYTHON_BIN and python)." >&2
    exit 1
  fi
fi

print_help () {
  echo "Usage: bash $(basename "$0") [--framework F] [--pass_k K] [--provider P] [--limit N] model_1 model_2 ..."
  echo ""
  echo "  --framework, --lang    Target quantum SDK: cirq | qiskit | pennylane   (default: cirq)"
  echo "  --pass_k               Pass@k samples (default: 1)"
  echo "  --provider             Generation provider: openrouter | coda | bedrock (default: openrouter)"
  echo "  --limit                Smoke-test: only run the first N tasks per model"
  echo ""
  echo "Provider notes:"
  echo "  openrouter             Reads API_KEY from .env; pass OpenRouter model ids."
  echo "  coda                   Reads CODA_API_KEY from .env. Use a label like 'coda/build'"
  echo "                         so result files group by Coda mode. The Coda agent is the"
  echo "                         generator; --framework still selects the target SDK."
  echo "  bedrock                Calls AWS Bedrock Converse directly with no system prompt,"
  echo "                         no tools, and no agent harness. Honours BEDROCK_MODEL,"
  echo "                         BEDROCK_REGION, BEDROCK_EFFORT (low|medium|high|max),"
  echo "                         BEDROCK_THINKING (default on). Pass any label as the"
  echo "                         model arg; it is only used to name the result files."
  echo ""
  echo "Examples:"
  echo "  bash $(basename "$0") --framework cirq --pass_k 5 \"openai/gpt-4.1\""
  echo "  bash $(basename "$0") --framework qiskit \"deepseek/deepseek-chat\""
  echo "  bash $(basename "$0") --provider coda --framework qiskit --pass_k 1 coda/build"
  echo "  bash $(basename "$0") --provider coda --framework cirq --limit 1 coda/build"
  echo "  bash $(basename "$0") --provider bedrock --framework qiskit bedrock/opus-4-6"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pass_k)
      PASS_K="$2"
      shift 2
      ;;
    --framework|--lang)
      FRAMEWORK="$2"
      shift 2
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      MODELS+=("$1")
      shift
      ;;
  esac
done

if [ ${#MODELS[@]} -eq 0 ]; then
  if [ "$PROVIDER" = "coda" ]; then
    MODELS=("coda/build")
    echo "No model passed; defaulting to 'coda/build' for --provider coda."
  elif [ "$PROVIDER" = "bedrock" ]; then
    MODELS=("bedrock/opus-4-6")
    echo "No model passed; defaulting to 'bedrock/opus-4-6' for --provider bedrock."
  else
    print_help
    exit 1
  fi
fi


case "$FRAMEWORK" in
  cirq)
    RESULTS_SCRIPT="$PIPELINE_DIR/cirq_pip/get_cirq_results.py"
    ;;
  qiskit)
    RESULTS_SCRIPT="$PIPELINE_DIR/qiskit_pip/get_qiskit_results.py"
    ;;
  pennylane)
    RESULTS_SCRIPT="$PIPELINE_DIR/pennylane_pip/get_pennylane_results.py"
    ;;
  *)
    echo "Error: unknown framework '$FRAMEWORK'. Use: cirq | qiskit | pennylane" >&2
    exit 1
    ;;
esac

case "$PROVIDER" in
  openrouter|coda|bedrock) ;;
  *)
    echo "Error: unknown provider '$PROVIDER'. Use: openrouter | coda | bedrock" >&2
    exit 1
    ;;
esac

cd "$REPO_ROOT" || exit 1

echo "Configuration:"
echo "  Provider:       $PROVIDER"
echo "  Framework:      $FRAMEWORK"
echo "  Pass@k samples: $PASS_K"
if [ -n "$LIMIT" ]; then
  echo "  Limit:          $LIMIT task(s) per model (smoke test)"
fi
echo "  Models:"
for m in "${MODELS[@]}"; do
  echo "    - $m"
done
echo "---"

API_ARGS=( --framework "$FRAMEWORK" --pass_k "$PASS_K" --provider "$PROVIDER" )
if [ -n "$LIMIT" ]; then
  API_ARGS+=( --limit "$LIMIT" )
fi

"$PYTHON_BIN" "$API_SCRIPT" "${API_ARGS[@]}" "${MODELS[@]}"
"$PYTHON_BIN" "$RESULTS_SCRIPT" "${MODELS[@]}" "$PASS_K"

echo "---"
echo "All evaluations complete."
