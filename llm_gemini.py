from __future__ import annotations

import os
import random
import time

try:
    # Preferred (google-genai). This can fail in environments where `google` is a namespace package
    # without the `genai` submodule due to dependency conflicts.
    from google import genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore
    try:
        import importlib

        genai = importlib.import_module("google.genai")  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Gemini SDK import failed. Install/upgrade the official package:\n"
            "  pip install -U google-genai\n"
            "If you have an old/conflicting `google` package installed, remove it:\n"
            "  pip uninstall -y google\n"
            f"\nOriginal error: {e}"
        ) from e


def require_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return key


def default_model() -> str:
    # gemini-flash-latest always resolves to the newest available Flash model.
    # Default can be overridden with GEMINI_MODEL.
    return os.environ.get("GEMINI_MODEL") or "gemini-flash-latest"


def _thinking_level() -> str | None:
    """
    Optional knob for Gemini "thinking".

    Supported by the Google GenAI SDK via thinking_config.thinking_level.
    Accepts: minimal, low, medium, high (case-insensitive).
    """
    raw = (os.environ.get("GEMINI_THINKING_LEVEL") or "").strip().lower()
    if not raw:
        return None
    if raw in ("min", "minimal", "none", "off"):
        return "low"
    if raw in ("low", "medium", "high"):
        return raw
    return None


def _parse_model_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    return [p for p in parts if p]


def model_chain(primary: str) -> list[str]:
    """Ordered list of models to try. Primary first, then env fallbacks, then defaults."""
    env_fallbacks = _parse_model_list(os.environ.get("GEMINI_MODEL_FALLBACKS"))
    defaults = [
        "gemini-flash-latest",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]
    out: list[str] = []
    for m in [primary, *env_fallbacks, *defaults]:
        if m and m not in out:
            out.append(m)
    return out


def _is_transient_gemini_error(msg: str) -> bool:
    needles = (
        "429",
        "RESOURCE_EXHAUSTED",
        "503",
        "UNAVAILABLE",
        "504",
        "DEADLINE",
        "INTERNAL",
        "500",
    )
    return any(n in msg for n in needles)


def _generate_once(*, client, system: str, user: str, model: str) -> str:
    prompt = f"SYSTEM INSTRUCTIONS:\n{system}\n\nUSER REQUEST:\n{user}\n"
    # Optional "thinking" control (best-effort; ignored if SDK/model doesn't support it)
    cfg = None
    tl = _thinking_level()
    if tl:
        try:
            from google.genai import types  # type: ignore

            cfg = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level=tl))
        except Exception:
            cfg = None

    if cfg is not None:
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
    else:
        resp = client.models.generate_content(model=model, contents=prompt)
    if not getattr(resp, "text", None):
        return str(resp)
    return resp.text


def generate_text(*, system: str, user: str, model: str) -> tuple[str, str]:
    """
    Generate text, automatically falling back across models on transient failures.
    Returns: (text, model_used)
    """
    api_key = require_gemini_api_key()
    client = genai.Client(api_key=api_key)  # type: ignore

    chain = model_chain(model)
    last_err: Exception | None = None

    for m in chain:
        for attempt in range(4):
            try:
                text = _generate_once(client=client, system=system, user=user, model=m)
                return text, m
            except Exception as e:
                last_err = e
                msg = str(e)
                if _is_transient_gemini_error(msg):
                    base = 0.9 * (2**attempt)
                    time.sleep(base + random.random() * 0.35)
                    continue
                break

    raise last_err or RuntimeError("Gemini call failed")
