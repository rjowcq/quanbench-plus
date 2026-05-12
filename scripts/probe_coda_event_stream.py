"""Probe what Coda's agent actually emits for a benchmark-style prompt.

This is a one-shot diagnostic: send a single QuanBench+ Qiskit prompt to the
live Coda agent, capture every SSE event, and print a categorised summary so
we can decide whether to keep extracting from streamed `token` events or
switch to the `final_generated_code` tool args / `structured_response`.

Usage:
    python scripts/probe_coda_event_stream.py [task_id]
    # default task_id is "11" (Deutsch-Jozsa with bitstring "1100")
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

from utils.read_jsonl import read_jsonl

load_dotenv()


def fetch_prompt(task_id: str, framework: str = "qiskit") -> Dict[str, Any]:
    tasks = read_jsonl(f"prompts/{framework}.jsonl")
    for t in tasks:
        if str(t["task_id"]) == task_id:
            return t
    raise SystemExit(f"task_id={task_id} not found in prompts/{framework}.jsonl")


def main() -> int:
    task_id = sys.argv[1] if len(sys.argv) > 1 else "11"
    framework = sys.argv[2] if len(sys.argv) > 2 else "qiskit"

    api_key = os.getenv("CODA_API_KEY") or os.getenv("CONDUCTOR_API_KEY")
    if not api_key:
        print("ERROR: Set CODA_API_KEY in .env", file=sys.stderr)
        return 1

    task = fetch_prompt(task_id, framework)
    prompt_text = task["complete_prompt"]
    entry = task["entry_point"]

    print(f"Probing task {task_id} ({entry}) on {framework}...\n")

    body = {
        "messages": [{"role": "user", "content": prompt_text}],
        "mode": "build",
        "fast": False,
    }

    base_url = os.getenv(
        "CODA_API_BASE_URL", "https://api.conductorquantum.com/v0/coda"
    ).rstrip("/")
    resp = requests.post(
        f"{base_url}/agents",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        json=body,
        timeout=600,
        stream=True,
    )
    print(f"HTTP {resp.status_code}\n")
    if resp.status_code != 200:
        print(resp.text[:2000])
        return 1

    events: List[Dict[str, Any]] = []
    last_event_type = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        line = line.strip()
        if line.startswith("event:"):
            last_event_type = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
        else:
            payload = line
        if not payload or payload == "[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except ValueError:
            continue
        if isinstance(ev, dict) and last_event_type and "type" not in ev:
            ev = {**ev, "type": last_event_type}
        events.append(ev)

    print(f"Total events: {len(events)}\n")
    print("Event type counts:")
    counts = Counter(str(e.get("type") or "?") for e in events)
    for t, n in counts.most_common():
        print(f"  {n:4d}  {t}")
    print()

    # 1. Streamed token text concatenated
    streamed = []
    for e in events:
        if e.get("type") == "token":
            content = e.get("content")
            if isinstance(content, str):
                streamed.append(content)
    streamed_text = "".join(streamed)
    print(f"Streamed `token` text length: {len(streamed_text)} chars")
    print(f"Streamed `token` text (first 600 chars):")
    print("-" * 70)
    print(streamed_text[:600])
    print("-" * 70)
    print(f"Streamed `token` text (last 400 chars):")
    print("-" * 70)
    print(streamed_text[-400:])
    print("-" * 70)
    print()

    # 2. Tool calls — what was final_generated_code called with?
    tool_calls = [e for e in events if e.get("type") in ("tool_call", "tool_result")]
    print(f"Tool events: {len(tool_calls)}")
    for e in tool_calls[:30]:
        name = e.get("name") or (e.get("data", {}) or {}).get("name", "?")
        print(f"  - {e.get('type')}: {name}")
    print()

    # 3. structured_response payload
    structured = [e for e in events if e.get("type") == "structured_response"]
    print(f"structured_response events: {len(structured)}")
    for s in structured:
        data = s.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                pass
        if isinstance(data, dict):
            print("  keys:", list(data.keys()))
            print(f"  circuit_name: {data.get('circuit_name')}")
            print(f"  skip_compilation: {data.get('skip_compilation')}")
            for fw in ("code", "cirq", "pennylane", "braket", "pyquil", "cudaq", "openqasm3"):
                v = data.get(fw)
                if isinstance(v, str):
                    print(f"  {fw}: {len(v)} chars; first 200: {v[:200]!r}")
                elif v is None:
                    print(f"  {fw}: None")

    # 4. Save full event stream for later inspection
    out_path = f"logs/event_probe_{framework}_{task_id}.json"
    os.makedirs("logs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"task": task, "events": events}, f, indent=2, default=str)
    print(f"\nFull event stream saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
