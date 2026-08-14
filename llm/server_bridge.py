#!/usr/bin/env python3
"""Bridge between the Go web server and llm/generate.py.

Reads ONE JSON object from stdin:
    {"prompt": str, "max_new": int, "temp": float, "top_k": int, "top_p": float}
Only "prompt" is required; the rest are optional and fall back to the
defaults in config/generate.json when omitted.

Prints ONE JSON object to stdout:
    {"response": "<generated text>"}   or   {"error": "<message>"}

The "Loaded ✓ | ..." line printed while the model loads is captured and
discarded so the only thing written to stdout is the final JSON object.
"""

import contextlib
import io
import json
import os
import sys

# Make llm/ importable no matter what CWD the Go server uses.
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
if _LLM_DIR not in sys.path:
    sys.path.insert(0, _LLM_DIR)


def _coerce(req, key, cast):
    """Return the value for key cast to `cast`, or None if absent/empty."""
    if key not in req or req[key] is None:
        return None
    try:
        return cast(req[key])
    except (TypeError, ValueError):
        raise ValueError(f"invalid value for '{key}': {req[key]!r}")


def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            raise ValueError("empty request on stdin")
        req = json.loads(raw)
        if not isinstance(req, dict):
            raise ValueError("request must be a JSON object")

        prompt = req.get("prompt")
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt is empty")

        kwargs = {}
        for key, cast in (
            ("max_new", int),
            ("temp", float),
            ("top_k", int),
            ("top_p", float),
        ):
            value = _coerce(req, key, cast)
            if value is not None:
                kwargs[key] = value

        from generate import chat

        # Capture the "Loaded ✓ | ..." line (and any other stray prints)
        # emitted while the model loads and generates so it never pollutes
        # the JSON we emit on stdout.
        with contextlib.redirect_stdout(io.StringIO()):
            text = chat(prompt=str(prompt), **kwargs)

        print(json.dumps({"response": text}, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 - surface any failure to the server
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))


if __name__ == "__main__":
    main()