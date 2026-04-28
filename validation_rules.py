"""
Deterministic validation rules for Agent 2 output (spec compliance).
LLM validation can miss edge cases; these rules always apply.
"""

from __future__ import annotations

import re
from typing import Any


def _severity_rank(s: str | None) -> int:
    return 2 if s == "BLOCK" else 1 if s == "WARN" else 0


def merge_flags(existing: list[dict[str, Any]], new_flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge flags; for the same (claim_id, flag_type), keep the strongest severity.
    """
    by_key: dict[tuple, dict[str, Any]] = {}
    for f in existing + new_flags:
        k = (f.get("claim_id"), f.get("flag_type"))
        if k not in by_key or _severity_rank(f.get("severity")) > _severity_rank(by_key[k].get("severity")):
            by_key[k] = f
    return list(by_key.values())


def rules_from_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for c in claims or []:
        cid = c.get("claim_id", "")
        conf = float(c.get("confidence", 1.0))
        url = (c.get("source_url") or "").strip()

        if not url:
            flags.append({
                "claim_id": cid or "unknown",
                "flag_type": "unverifiable",
                "detail": "No source URL provided for claim",
                "severity": "BLOCK",
            })
            continue

        if conf < 0.3:
            flags.append({
                "claim_id": cid,
                "flag_type": "low_confidence",
                "detail": f"Confidence {conf:.2f} is below BLOCK threshold of 0.3",
                "severity": "BLOCK",
            })
        elif conf < 0.6:
            flags.append({
                "claim_id": cid,
                "flag_type": "low_confidence",
                "detail": f"Confidence {conf:.2f} is below WARN threshold of 0.6",
                "severity": "WARN",
            })
    return flags


def compute_validation_result(flags: list[dict[str, Any]]) -> str:
    if any(f.get("severity") == "BLOCK" for f in flags):
        return "FAIL"
    if any(f.get("severity") == "WARN" for f in flags):
        return "WARN"
    return "PASS"


def rules_from_packet(claims: list[dict[str, Any]], packet: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deterministically flag claims whose source URL returned a 4xx/5xx or was unreachable.
    Spec requirement: "If a source URL returns a 404 or irrelevant content, Agent 2 must flag it."
    This runs independently of the LLM so HTTP errors are never silently swallowed.
    """
    by_url: dict[str, dict[str, Any]] = {}
    for p in packet or []:
        for key in ("url", "final_url"):
            u = (p.get(key) or "").strip()
            if u:
                by_url[u] = p

    flags: list[dict[str, Any]] = []
    for c in claims or []:
        src = (c.get("source_url") or "").strip()
        if not src:
            continue
        rec = by_url.get(src)
        if rec is None:
            continue
        code = rec.get("status_code")
        if code is None:
            continue
        try:
            code_int = int(code)
        except (TypeError, ValueError):
            continue
        if code_int >= 400:
            flags.append({
                "claim_id": c.get("claim_id") or "unknown",
                "flag_type": "unverifiable",
                "detail": f"Source URL returned HTTP {code_int} — cannot verify claim against this source.",
                "severity": "BLOCK",
            })
    return flags


def apply_hard_validation_rules(
    result: dict[str, Any],
    clean_agent1: dict[str, Any],
    validation_packet: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Mutates and returns result with merged flags and corrected validation_result / approved_email.
    clean_agent1 = Agent 1 JSON without keys starting with '_'
    validation_packet = live-fetched pages from Agent 2 (used for deterministic 404 checks)
    """
    claims = clean_agent1.get("claims") or []
    draft_email = (clean_agent1.get("draft_email") or "").strip()
    resolved_email = (clean_agent1.get("resolved_email") or "").strip()

    if not draft_email:
        # Spec: if no draft email was produced, validation cannot PASS/WARN.
        merged = merge_flags(
            result.get("flags") or [],
            [
                {
                    "claim_id": "system",
                    "flag_type": "unverifiable",
                    "detail": "No draft_email was produced by Agent 1 (insufficient sourced signal / spec requirement).",
                    "severity": "BLOCK",
                }
            ],
        )
        result["flags"] = merged
        result["validation_result"] = "FAIL"
        result["approved_email"] = None
        return result

    # If Agent 1 did not resolve an email address, Agent 2 should not raise a domain_mismatch.
    # The spec asks the validator to check domain validity when an email is produced/inferred.
    # Treat "no email found" as a research gap, not a domain mismatch.
    if not resolved_email and result.get("flags"):
        result["flags"] = [
            f
            for f in (result.get("flags") or [])
            if not (
                (f.get("flag_type") == "domain_mismatch")
                and (str(f.get("claim_id") or "").lower() in ("email", "resolved_email", "domain"))
            )
        ]

    hard_flags = rules_from_claims(claims)
    if validation_packet:
        hard_flags = merge_flags(hard_flags, rules_from_packet(claims, validation_packet))
    merged = merge_flags(result.get("flags") or [], hard_flags)
    result["flags"] = merged

    vr = compute_validation_result(merged)
    result["validation_result"] = vr

    if vr == "FAIL":
        result["approved_email"] = None
    else:
        _repair_approved_email_if_needed(result, clean_agent1)
    return result


def _looks_like_email_only(s: str) -> bool:
    s = (s or "").strip()
    if not s or "\n" in s:
        return False
    # Single-line email-ish string (model sometimes returns only the address)
    return bool(re.fullmatch(r"[\w.+-]+@[\w.-]+\.\w+", s))


def _repair_approved_email_if_needed(result: dict[str, Any], clean_agent1: dict[str, Any]) -> None:
    draft = (clean_agent1.get("draft_email") or "").strip()
    app = (result.get("approved_email") or "").strip()
    if not draft:
        return
    # Spec: on PASS/WARN, approved_email must be the full email body ready to send.
    # Some model responses omit approved_email even when validation_result is WARN.
    if not app:
        result["approved_email"] = draft
        return
    if _looks_like_email_only(app) or (app and len(app) < 160 and draft.count("\n") >= 2):
        result["approved_email"] = draft
