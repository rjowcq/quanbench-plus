FEEDBACK_NUM=5
FRAMEWORK="cirq"
PROVIDER="openrouter"
LIMIT=""
MODELS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

PIPELINE_DIR="$REPO_ROOT/feedback_loop"
API_SCRIPT="$PIPELINE_DIR/api.py"

if [ ! -x "$PYTHON_BIN" ] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: could not find a Python interpreter (looked for $PYTHON_BIN and python)." >&2
    exit 1
  fi
fi

print_help () {
  echo "Usage: bash $(basename "$0") [--framework F] [--feedback_num N] [--provider P] [--limit N] model_1 model_2 ..."
  echo ""
  echo "  --framework, --lang    Target quantum SDK: cirq | qiskit | pennylane   (default: cirq)"
  echo "  --feedback_num         Max attempts per task (default: 5)"
  echo "  --provider             Generation provider: openrouter | coda          (default: openrouter)"
  echo "  --limit                Smoke-test: only run the first N tasks per model"
  echo ""
  echo "Provider notes:"
  echo "  openrouter             Reads API_KEY from .env; pass OpenRouter model ids."
  echo "  coda                   Reads CODA_API_KEY from .env. Use 'coda/build' as the model"
  echo "                         label so result files group by Coda mode. The Coda agent is"
  echo "                         the generator; --framework still selects the target SDK."
  echo ""
  echo "Examples:"
  echo "  bash $(basename "$0") --framework cirq --feedback_num 5 deepseek/deepseek-r1"
  echo "  bash $(basename "$0") --framework qiskit openai/gpt-4.1"
  echo "  bash $(basename "$0") --provider coda --framework qiskit --feedback_num 5 coda/build"
  echo "  bash $(basename "$0") --provider coda --framework cirq --limit 1 coda/build"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --feedback_num)
      FEEDBACK_NUM="$2"
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
  else
    print_help
    exit 1
  fi
fi

case "$PROVIDER" in
  openrouter|coda|bedrock) ;;
  *)
    echo "Error: unknown provider '$PROVIDER'. Use: openrouter | coda | bedrock" >&2
    exit 1
    ;;
esac

cd "$REPO_ROOT" || exit 1

echo "Configuration:"
echo "  Provider:          $PROVIDER"
echo "  Framework:         $FRAMEWORK"
echo "  Feedback attempts: $FEEDBACK_NUM"
if [ -n "$LIMIT" ]; then
  echo "  Limit:             $LIMIT task(s) per model (smoke test)"
fi
echo "  Models:"
for m in "${MODELS[@]}"; do
  echo "    - $m"
done
echo "---"

ARGS=( --framework "$FRAMEWORK" --feedback_num "$FEEDBACK_NUM" --provider "$PROVIDER" )
if [ -n "$LIMIT" ]; then
  ARGS+=( --limit "$LIMIT" )
fi
"$PYTHON_BIN" "$API_SCRIPT" "${ARGS[@]}" "${MODELS[@]}"

echo "---"
echo "Feedback-loop run complete."
