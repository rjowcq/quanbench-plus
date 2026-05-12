from utils.parse_response import parse_response


def test_parse_response_preserves_provider_error():
    response = {
        "model": "coda/build",
        "error": {"status_code": 0, "body": "Agent call failed: HTTP 303"},
        "choices": [
            {
                "finish_reason": "error",
                "message": {"role": "assistant", "content": "# Error"},
            }
        ],
        "usage": {},
    }

    parsed = parse_response((response, "def task():\n"), "task")

    assert parsed["error"]["body"] == "Agent call failed: HTTP 303"
    assert parsed["code"] == ""
