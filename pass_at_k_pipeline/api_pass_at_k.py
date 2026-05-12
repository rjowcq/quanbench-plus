from pass_at_k_pipeline.cirq_pip.paths import CIRQ_JSONL
from pass_at_k_pipeline.pennylane_pip.paths import PENNYLANE_JSONL
from pass_at_k_pipeline.qiskit_pip.paths import QISKIT_JSONL
from pass_at_k_pipeline.pennylane_pip.paths import (
    MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_PENNYLANE,
)
from pass_at_k_pipeline.cirq_pip.paths import (
    MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_CIRQ,
)
from pass_at_k_pipeline.qiskit_pip.paths import (
    MODEL_RESPONSES_DIR as MODEL_RESPONSES_DIR_QISKIT,
)
import os
import time
from typing import Any, Dict, List
from utils.parse_prompt import parse_prompt
from utils.get_function_signature_from_prompt import get_function_signature_from_prompt
from utils.read_jsonl import read_jsonl
from utils.parse_response import parse_response
from utils.providers import (
    CODA_PROVIDER,
    OPENROUTER_PROVIDER,
    SUPPORTED_PROVIDERS,
    send_generation_request,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import json
import argparse
from pass_at_k_pipeline.defaults import DEFAULT_MODELS

load_dotenv()


def send_request(request: Dict[str, Any], provider: str = OPENROUTER_PROVIDER):
    """Dispatch a single request via the chosen provider.

    Returns ``(normalized_response, chat_completion_prefill)`` so the
    existing ``parse_response`` tuple contract still holds.
    """
    return send_generation_request(request, provider)


def parse_requests(json_path: str, models: list):
    tasks = read_jsonl(json_path)
    requests = []
    tasks_info = []
    for model in models:
        for task in tasks:
            chat_completion_part = get_function_signature_from_prompt(
                task.get("complete_prompt")
            )
            requests.append(
                parse_prompt(
                    task.get("complete_prompt"),
                    chat_completion_part,
                    model,
                    n=1,
                    prefill=False,
                )
            )
            tasks_info.append(
                {
                    "task_id": task.get("task_id"),
                    "entry_point": task.get("entry_point"),
                    "category": task.get("category"),
                    "model": model,
                }
            )

    return requests, tasks_info


def process_single_task_pass_k(args):
    """
    Process ONE (task, version) sample.
    Returns (task_index, version, enriched_result).
    """
    task_index, request, task_info, version, pass_k, provider = args
    entry_point = task_info.get("entry_point")
    single_request = {k: v for k, v in request.items() if k != "n"}
    single_request.setdefault("temperature", 0.8)
    if pass_k > 1:
        single_request["temperature"] = 0.8
    try:
        raw_response = send_request(single_request, provider=provider)
        parsed_response = parse_response(raw_response, entry_point)

        enriched_result = {
            **parsed_response,
            **task_info,  # task_id, entry_point, category, requested model label, ...
            "version": version,
        }
        return task_index, version, enriched_result
    except Exception as exc:
        print(
            f"Request task {task_info.get('task_id')} v{version} generated an exception: {exc}"
        )
        error_result = {
            "id": f"error-{int(time.time())}-{task_index}-v{version}",
            "error": {"status_code": 0, "body": str(exc)},
            "choices": [
                {
                    "finish_reason": "error",
                    "native_finish_reason": "error",
                    "message": {
                        "role": "assistant",
                        "content": f"Error: {str(exc)}",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            **task_info,
            "version": version,
        }
        return task_index, version, error_result


def _generation_max_workers(provider: str) -> int:
    env_name = "CODA_MAX_WORKERS" if provider == CODA_PROVIDER else "GENERATION_MAX_WORKERS"
    default = 4 if provider == CODA_PROVIDER else 16
    try:
        return max(1, int(os.getenv(env_name, str(default))))
    except ValueError:
        return default


def process_requests_pass_k(requests, tasks_info, pass_k, provider=OPENROUTER_PROVIDER):
    """
    Fully-parallel pass@k:
    Submits len(requests) * pass_k independent jobs to the thread pool.
    in deterministic order: task i occupies [i*pass_k ... i*pass_k + pass_k-1].
    """
    total_tasks = len(requests)
    total_calls = total_tasks * pass_k
    completed = 0
    max_workers = _generation_max_workers(provider)
    print(
        f"   Sending {total_calls} requests with {max_workers} concurrent workers "
        f"(pass@{pass_k}, provider={provider})..."
    )
    results = [None] * total_calls
    all_args = []
    for i, (req, info) in enumerate(zip(requests, tasks_info)):
        for v in range(1, pass_k + 1):
            all_args.append((i, req, info, v, pass_k, provider))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(process_single_task_pass_k, args): (args[0], args[3])
            for args in all_args
        }

        for future in as_completed(future_to_key):
            task_index, version = future_to_key[future]
            out_index = task_index * pass_k + (version - 1)

            try:
                _, _, result = future.result()
                results[out_index] = result
            except Exception as exc:
                print(
                    f"Request task {task_index} v{version} generated an exception: {exc}"
                )
                results[out_index] = {
                    "id": f"error-{int(time.time())}-{task_index}-v{version}",
                    "choices": [
                        {
                            "finish_reason": "error",
                            "native_finish_reason": "error",
                            "message": {
                                "role": "assistant",
                                "content": f"Error: {str(exc)}",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    **tasks_info[task_index],
                    "version": version,
                }

            completed += 1
            if completed % 10 == 0:
                print(
                    f"   Progress: {completed}/{total_calls} ({completed / total_calls * 100:.1f}%)"
                )

            time.sleep(0.01)

    print(f"\n   ✅ Completed all {total_calls} requests (pass@{pass_k})")
    return results


def get_jsonl_path(framework: str):
    if framework == "cirq":
        return CIRQ_JSONL
    elif framework == "pennylane":
        return PENNYLANE_JSONL
    else:
        return QISKIT_JSONL


def get_model_reponses_dir(framework: str):
    if framework == "cirq":
        return MODEL_RESPONSES_DIR_CIRQ
    elif framework == "pennylane":
        return MODEL_RESPONSES_DIR_PENNYLANE
    else:
        return MODEL_RESPONSES_DIR_QISKIT


def main(
    models: list,
    framework: str,
    pass_k: int = 1,
    provider: str = OPENROUTER_PROVIDER,
    limit: int | None = None,
):
    all_results: Dict[str, List[Dict[str, Any]]] = {f"{framework}": []}
    jsonl_path = get_jsonl_path(framework=framework)
    model_response_dir = get_model_reponses_dir(framework=framework)
    if provider == CODA_PROVIDER:
        # Tell the Coda response parser which framework's translation to
        # prefer when the agent emits a structured_response with multiple
        # framework variants. This makes the parser robust to Coda's
        # internal pivot framework.
        os.environ.setdefault("CODA_TARGET_FRAMEWORK", framework)
    print(f"Starting API requests via provider={provider}...")
    requestss, tasks_info = parse_requests(jsonl_path, models)

    if limit is not None and limit > 0:
        # Slice each model's task block while preserving model ordering.
        kept_requests: List[Dict[str, Any]] = []
        kept_info: List[Dict[str, Any]] = []
        per_model_count: Dict[str, int] = {m: 0 for m in models}
        for req, info in zip(requestss, tasks_info):
            model_name = info.get("model")
            if per_model_count.get(model_name, 0) < limit:
                kept_requests.append(req)
                kept_info.append(info)
                per_model_count[model_name] = per_model_count.get(model_name, 0) + 1
        requestss, tasks_info = kept_requests, kept_info
        print(f"   --limit {limit}: keeping {len(requestss)} task(s) total")

    print(f"   Generated {len(requestss)} requests across {len(models)} models")
    results = process_requests_pass_k(requestss, tasks_info, pass_k, provider=provider)

    all_results[f"{framework}"] = results
    print(f"Completed {len(results)} responses")

    # Ensure the directory exists before saving
    print(f"Saving results to file: {model_response_dir}...")
    model_response_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        model_name = model.replace("/", "_")
        model_results = [r for r in results if r.get("model") == model]
        output_file = model_response_dir / f"{model_name}_{framework}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print(f"   Saved {len(model_results)} results to {output_file}")

    total_results = len(results)
    print(f"All done! Saved {total_results} total responses")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation on quantum computing tasks"
    )
    parser.add_argument("models", nargs="*", help="Model names to evaluate")
    parser.add_argument(
        "--framework",
        type=str,
        default="cirq",
        choices=["cirq", "pennylane", "qiskit"],
        help="Framework to use: cirq, pennylane, or qiskit (default: cirq)",
    )
    parser.add_argument(
        "--pass_k",
        type=int,
        default=1,
        help="Number of samples for pass@k evaluation (default: 1)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=OPENROUTER_PROVIDER,
        choices=list(SUPPORTED_PROVIDERS),
        help="Generation provider to use (default: openrouter). Use 'coda' to "
        "benchmark the Coda agent; pass model labels like 'coda/build' so the "
        "results files group by Coda mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, keep only the first N tasks per model (smoke testing).",
    )
    args = parser.parse_args()
    if args.models:
        models = args.models
    elif args.provider == CODA_PROVIDER:
        models = ["coda/build"]
    else:
        models = DEFAULT_MODELS
    main(
        models,
        args.framework,
        args.pass_k,
        provider=args.provider,
        limit=args.limit,
    )
