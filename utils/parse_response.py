import re


def extract_code_from_markdown(text: str, entry_point: str) -> str:
    """Extract Python code from markdown code blocks if present.

    Agent-style providers (Coda) may emit multiple ``python`` blocks in a
    single response: the LLM shows an initial draft, sees pipeline feedback
    (lint / type / test failures), then re-emits a corrected version in a
    new block. We want the LATEST revision matching the entry point, not
    the first draft.
    """
    # Try to find ```python ... ``` blocks first
    python_blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if python_blocks:
        # Walk in reverse so we pick the LAST block containing the entry
        # point definition — that is the LLM's most recent revision after
        # any pipeline-driven fixes.
        for block in reversed(python_blocks):
            if f"def {entry_point}" in block:
                return block.strip()
        # No block defines the entry point; fall back to the LAST block,
        # which is still the most recent thing the LLM emitted.
        return python_blocks[-1].strip()

    # Try generic ``` ... ``` blocks
    generic_blocks = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if generic_blocks:
        for block in reversed(generic_blocks):
            if f"def {entry_point}" in block:
                return block.strip()
        return generic_blocks[-1].strip()

    return text


def parse_response(args, entry_point):
    """Parse single response (first choice only)."""
    response = args[0]
    chat_completion = args[1]
    out = {}
    out["model"] = response.get("model")
    usage = response.get("usage") or {}

    out["usage"] = usage
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["completion_tokens"] = usage.get("completion_tokens")
    out["total_tokens"] = usage.get("total_tokens")
    cdet = usage.get("completion_tokens_details") or {}
    pdet = usage.get("prompt_tokens_details") or {}

    out["reasoning_tokens"] = cdet.get("reasoning_tokens")
    out["accepted_prediction_tokens"] = cdet.get("accepted_prediction_tokens")
    out["rejected_prediction_tokens"] = cdet.get("rejected_prediction_tokens")
    out["cached_tokens"] = pdet.get("cached_tokens")
    out["cache_write_tokens"] = pdet.get("cache_write_tokens")

    choice = (response.get("choices") or [{}])[0]
    if response.get("error") or choice.get("finish_reason") == "error":
        out["error"] = response.get("error") or {
            "body": (choice.get("message") or {}).get("content", "unknown provider error")
        }
        out["code"] = ""
        return out

    message = choice.get("message") or {}
    code = message.get("content") or ""

    code = extract_code_from_markdown(code, entry_point)

    if ("def " + entry_point) not in code:
        print("This is a chat completion model!")
        code = chat_completion + code
    out["code"] = code
    return out

