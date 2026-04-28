"""
Post-process Agent 1 claims to avoid common validation failures.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


_KKC_MARKERS = (
    "kyber knight",
    "kyber knight capital",
    "kkc",
    "pre-seed",
    "pre seed",
    "seed-stage",
    "seed stage",
    "commerce",
    "our fund",
    "our thesis",
    "we invest",
    "we're a",
    "we are a",
)


def _mentions_kkc_boilerplate(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in _KKC_MARKERS)


def _host(url: str) -> str:
    return (urlparse(url).netloc or "").lower()

def _is_sender_site(url: str) -> bool:
    """
    Detect links to the sender (Kyber Knight) website. We keep this loose because
    the exact domain may vary, but it should include 'kyber' or 'kkc'.
    """
    h = _host(url or "")
    return ("kyber" in h) or (h.startswith("kkc")) or (".kkc" in h)


def prune_unnecessary_kkc_references(result: dict[str, Any]) -> dict[str, Any]:
    """
    Remove KKC-only boilerplate claims/sources unless they are required to support
    a non-KKC sentence in the final email.

    This prevents the model from padding emails with sender self-description that
    often creates noisy or misaligned claim/source mappings.
    """
    claims = result.get("claims") or []
    if not claims:
        return result

    kept_claims: list[dict[str, Any]] = []
    referenced_urls: set[str] = set()
    for claim in claims:
        claim_text = (claim.get("claim_text") or "").strip()
        source_url = (claim.get("source_url") or "").strip()
        if _mentions_kkc_boilerplate(claim_text):
            continue
        kept_claims.append(claim)
        if source_url:
            referenced_urls.add(source_url)

    result["claims"] = kept_claims

    research_sources = result.get("research_sources") or []
    if research_sources:
        filtered_sources: list[dict[str, Any]] = []
        for src in research_sources:
            url = (src.get("url") or "").strip()
            if not url:
                filtered_sources.append(src)
                continue
            if _is_sender_site(url) and url not in referenced_urls:
                continue
            filtered_sources.append(src)
        result["research_sources"] = filtered_sources

    return result


def patch_kkc_claim_sources(claims: list[dict[str, Any]] | None, kkc_profile_url: str | None) -> None:
    """
    If KKC_PROFILE_URL is set, re-point KKC self-description claims to that URL so
    Agent 2 can verify them against real KKC copy (not the contact's org site).
    Mutates claims in place.
    """
    if not claims or not (kkc_profile_url or "").strip():
        return
    kkc = kkc_profile_url.strip()
    kkc_host = _host(kkc)

    for c in claims:
        ct = c.get("claim_text") or ""
        if not _mentions_kkc_boilerplate(ct):
            continue
        su = (c.get("source_url") or "").strip()
        sh = _host(su)
        # If the claim is about KKC but the cited host doesn't look like KKC, rewrite.
        if not sh or (kkc_host and kkc_host not in sh and "kyber" not in sh):
            c["source_url"] = kkc
            c["source_type"] = "org_website"
            # Keep confidence high only if we're explicitly tying to provided KKC URL.
            c["confidence"] = max(float(c.get("confidence") or 0.0), 0.85)


_QUOTE_CHARS = "\"\u201c\u201d"  # straight and curly double quotes


def strip_quotes_in_email_and_claims(
    draft_email: str | None, claims: list[dict[str, Any]] | None
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """
    Deterministically remove quotation marks from the email and claim_texts.

    This keeps the style human and prevents direct-quote vibes, while preserving
    claim traceability (claim_text stays aligned to the email text).
    """
    if draft_email is None and not claims:
        return draft_email, claims

    table = {ord(ch): None for ch in _QUOTE_CHARS}

    def _strip(s: str) -> str:
        return s.translate(table)

    new_email = _strip(draft_email) if isinstance(draft_email, str) else draft_email
    if claims:
        for c in claims:
            if isinstance(c.get("claim_text"), str):
                c["claim_text"] = _strip(c["claim_text"])

    return new_email, claims
