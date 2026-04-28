"""
Agent 2: Validator
Receives Agent 1's structured JSON output and independently re-verifies every claim.
Has NO access to Agent 1's prompt chain or intermediate reasoning — only the JSON.
Uses live web requests (DuckDuckGo HTML search + direct page fetch) to re-fetch sources and cross-check facts.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import (
    _env_float,
    _env_int,
    extract_json as _extract_json,
    fetch_url,
    guess_date_signals,
    guess_org_homepage_urls,
    is_usable_retrieval,
    unique_urls,
    utc_now_iso,
    web_search,
)
from llm_gemini import default_model, generate_text
from validation_rules import apply_hard_validation_rules, compute_validation_result, merge_flags

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it",
    "its", "me", "my", "not", "of", "on", "or", "our", "she", "that", "the",
    "their", "there", "they", "this", "to", "was", "we", "were", "will",
    "with", "you", "your",
}


JSON_REPAIR_SYSTEM_PROMPT = """You repair invalid JSON produced by an LLM.

Return ONLY a valid JSON object matching the required schema. Do not add commentary.

Rules:
- Escape quotes inside string values properly.
- Preserve all fields and data; do not invent new facts.
- If a value is clearly truncated or incomplete, set it to null rather than guessing.
"""

JSON_REPAIR_USER_TEMPLATE = """Fix this into valid JSON only.

BROKEN OUTPUT:
{broken}
"""

VALIDATOR_SYSTEM_PROMPT = """You are a fact-checking agent for Kyber Knight Capital (KKC). You receive a structured JSON record produced by a research agent and your job is to independently verify every claim in it.

You will be given a validation packet containing live-fetched web pages (URLs + excerpts). Use it to:
- Re-verify titles against the organization's own website (not just LinkedIn)
- Confirm the person is the right contact for the stated outreach purpose
- Re-fetch every source URL and confirm it actually supports the claim
- Validate the email domain and naming convention against the org's website
- Catch any hallucinated figures, names, dates, or titles

STRICT RULES:
1. You operate ONLY on the JSON you receive. You have no memory of any prior research.
2. For each claim, use the fetched content for its source_url. If it returns a 404, redirects unexpectedly, or the content does not support the claim, flag it.
3. Apply the confidence floor rules:
   - confidence < 0.3 → severity: BLOCK
   - 0.3 ≤ confidence < 0.6 → severity: WARN
   - confidence ≥ 0.6 → no flag (unless other issues found)
4. Title mismatch: if LinkedIn title ≠ org website title, flag as title_mismatch WARN.
5. Wrong contact: if the person's actual role doesn't match the outreach intent, flag as wrong_contact WARN or BLOCK.
6. Hallucination: if a specific fact (date, quote, event name, figure) cannot be confirmed by any retrievable source, flag as hallucination BLOCK.
7. Domain mismatch: if the resolved_email domain doesn't match the org's actual domain, flag as domain_mismatch BLOCK. However, if resolved_email is null (no address found), flag as unverifiable WARN — not a BLOCK. A null email means we could not confirm the address; the email content may still be valid and the sender can find the address separately. Only BLOCK on domain_mismatch when an address IS provided but does not match.
8. title_mismatch: only if the org website (in the packet) lists a **different** title for this person than `resolved_title` / the email implies. If the org site simply **does not mention** their title, that is **not** a mismatch — do not raise title_mismatch for absence alone.
9. Statements about **Kyber Knight Capital (KKC)** must be supported by a page that is **about KKC** (e.g., a KKC profile URL in the packet). If a claim about KKC cites only the contact's org site and that page does not discuss KKC, flag as unverifiable/hallucination (severity based on how central the claim is).
10. `approved_email` on PASS/WARN must be the **full email body** the sender should send (multi-line), not just an email address.

Validation result:
- PASS: no BLOCK flags, zero or minimal WARNs
- WARN: one or more WARN flags, no BLOCKs
- FAIL: one or more BLOCK flags → approved_email must be null

Return ONLY valid JSON (no markdown, no preamble):
{
  "name": "string",
  "validation_result": "PASS | WARN | FAIL",
  "flags": [
    {
      "claim_id": "string (or 'title' / 'email' / 'contact_fit' for non-claim checks)",
      "flag_type": "title_mismatch | wrong_contact | unverifiable | domain_mismatch | hallucination | low_confidence",
      "detail": "string",
      "severity": "BLOCK | WARN"
    }
  ],
  "approved_email": "string or null",
  "validation_notes": "string (brief summary of what was checked)",
  "_validated_at": "ISO8601 string"
}"""


VALIDATOR_USER_TEMPLATE = """Validate this research output from Agent 1. Re-verify every claim independently using ONLY the provided live-fetched packet.

INPUT JSON:
{agent1_json}

VALIDATION PACKET (live-fetched pages; includes refetches of claim source_url and cross-check pages):
{validation_packet}

Your checks:
1. Title accuracy: search "{name} {organization}" on the org's own website to confirm the resolved_title.
2. Right contact: given the outreach is from a pre-seed/seed VC (KKC, focused on commerce, AI, labor), is this person actually the right contact?
3. Email domain: search the org's website for contact/team pages to confirm the email domain and naming convention.
4. Claim verification: for each claim in the claims array, attempt to retrieve the source_url and confirm the content supports the claim_text.
5. Hallucination check: scan the draft_email for any specific facts (dates, event names, quotes, metrics) and confirm each is grounded in a source.

Be thorough. A pipeline that never fails validation is not actually validating anything."""


def build_validation_packet(agent1_output: dict[str, Any], clean_input: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build a packet of live-fetched pages to support independent validation.
    Includes:
      - refetch of every claim source_url
      - org team/contact pages via search
      - generic person/org pages via search
    """
    name = clean_input.get("name", "")
    org = clean_input.get("organization", "")
    claims = clean_input.get("claims") or []

    cap = _env_int("AGENT2_FETCH_CAP", 20)
    max_chars = _env_int("FETCH_MAX_CHARS", 12000)
    timeout_s = _env_float("FETCH_TIMEOUT_S", 20.0)
    min_chars = _env_int("FETCH_MIN_CHARS", 340)
    search_max_results = _env_int("SEARCH_MAX_RESULTS", 6)
    fetch_workers = _env_int("FETCH_WORKERS", 8)

    urls: list[str] = []

    # Always include KKC's own profile so Agent 2 can verify KKC self-description claims.
    kkc_profile = os.environ.get("KKC_PROFILE_URL", "").strip()
    if kkc_profile:
        urls.append(kkc_profile)

    for c in claims:
        u = (c.get("source_url") or "").strip()
        if u:
            urls.append(u)

    input_contact = agent1_output.get("_input_contact") or {}
    li = (input_contact.get("linkedin_url") or "").strip()
    if li:
        urls.append(li)
    urls.extend(guess_org_homepage_urls(org))

    queries = [
        f"{name} {org}",
        f"{org} website",
        f"{org} team",
        f"{org} leadership",
        f"{org} contact",
    ]
    search_timeout = min(10.0, timeout_s)

    def _a2_search(q: str) -> list[dict[str, str]]:
        try:
            return web_search(q, max_results=max(3, min(10, search_max_results)), timeout_s=search_timeout)
        except Exception as e:
            logger.warning(f"[Agent2] Search failed for query='{q}': {e}")
            return []

    hits: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=5) as sex:
        for result in sex.map(_a2_search, queries):
            hits.extend(result)

    urls.extend(unique_urls(hits))

    seen: set[str] = set()
    uniq: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            uniq.append(u)

    packet: list[dict[str, Any]] = []

    max_to_fetch = cap * 2
    to_fetch = uniq[:max_to_fetch]
    if to_fetch:
        ex2 = ThreadPoolExecutor(max_workers=max(1, fetch_workers))
        try:
            futs = {ex2.submit(fetch_url, u, timeout_s=timeout_s, max_chars=max_chars): u for u in to_fetch}
            for fut in as_completed(futs):
                u = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {
                        "url": u,
                        "status_code": None,
                        "final_url": None,
                        "retrieved_at": utc_now_iso(),
                        "text_excerpt": "",
                        "error": str(e),
                        "reader_used": False,
                    }
                packet.append(rec)
        finally:
            ex2.shutdown(wait=True, cancel_futures=True)

    usable = sum(1 for r in packet if is_usable_retrieval(r, min_chars=min_chars))
    if usable < max(4, cap // 2):
        extra_queries = [
            f"{name} {org} interview",
            f"{name} {org} podcast",
            f"{org} news 2026",
            f"{org} funding 2026",
        ]
        for q in extra_queries:
            try:
                for h in web_search(q, max_results=max(3, min(10, search_max_results)), timeout_s=timeout_s):
                    u = (h.get("url") or "").strip()
                    if u and u not in seen:
                        seen.add(u)
                        packet.append(fetch_url(u, timeout_s=timeout_s, max_chars=max_chars))
                        usable = sum(1 for r in packet if is_usable_retrieval(r, min_chars=min_chars))
                        if usable >= cap:
                            break
            except Exception as e:
                logger.warning(f"[Agent2] Extra search failed for query='{q}': {e}")

    return packet[:max_to_fetch]


def run_agent2(agent1_path: str | Path, model: str = default_model()) -> dict:
    """
    Run Agent 2 on Agent 1's JSON file on disk.

    Agent 2 loads ONLY the saved Agent 1 artifact (no in-memory handoff), matching the
    assignment's "structured intermediate on disk" / independent validator requirement.
    """
    path = Path(agent1_path)
    with open(path) as f:
        agent1_output = json.load(f)

    name = agent1_output.get("name", "Unknown")
    organization = agent1_output.get("organization", "Unknown")

    # Strip internal metadata before passing to Agent 2 (no context leakage)
    clean_input = {k: v for k, v in agent1_output.items() if not k.startswith("_")}

    validation_packet = build_validation_packet(agent1_output, clean_input)

    user_message = VALIDATOR_USER_TEMPLATE.format(
        agent1_json=json.dumps(clean_input, indent=2),
        name=name,
        organization=organization,
        validation_packet=json.dumps(validation_packet, indent=2),
    )

    logger.info(f"[Agent2] Starting validation for {name} @ {organization}")
    logger.info(f"[Agent2] Validating {len(clean_input.get('claims', []))} claims")

    try:
        full_text, model_used = generate_text(
            system=VALIDATOR_SYSTEM_PROMPT,
            user=user_message,
            model=model,
        )

        try:
            result = _extract_json(full_text)
        except Exception as e:
            logger.warning(f"[Agent2] JSON parse failed for {name}, attempting repair: {e}")
            repair_user = JSON_REPAIR_USER_TEMPLATE.format(broken=full_text[:12000])
            repaired_text, _ = generate_text(system=JSON_REPAIR_SYSTEM_PROMPT, user=repair_user, model=model)
            result = _extract_json(repaired_text)

        apply_hard_validation_rules(result, clean_input, validation_packet)
        _apply_recency_wording_rules(result, clean_input, validation_packet)
        _apply_claim_source_alignment_rules(result, clean_input, validation_packet)

        result["_agent"] = "agent2"
        result["_model"] = model_used
        result["_validated_at"] = utc_now_iso()
        result["_validation_packet"] = validation_packet

        result["_research_sources"] = agent1_output.get("research_sources", [])
        result["_claims"] = agent1_output.get("claims", [])

        logger.info(f"[Agent2] Done for {name}: result={result.get('validation_result')}, flags={len(result.get('flags', []))}")
        return result

    except Exception as e:
        logger.error(f"[Agent2] FAILED for {name}: {e}")
        return {
            "name": name,
            "validation_result": "FAIL",
            "flags": [{
                "claim_id": "system",
                "flag_type": "unverifiable",
                "detail": f"Validator crashed: {e}",
                "severity": "BLOCK"
            }],
            "approved_email": None,
            "validation_notes": f"Validator error: {e}",
            "_validated_at": utc_now_iso(),
            "_agent": "agent2",
            "_model": model,
            "_validation_packet": validation_packet if "validation_packet" in locals() else [],
            "_research_sources": agent1_output.get("research_sources", []),
            "_claims": agent1_output.get("claims", []),
        }


def _apply_recency_wording_rules(result: dict[str, Any], clean_input: dict[str, Any], validation_packet: list[dict[str, Any]]) -> None:
    claims = clean_input.get("claims") or []
    if not claims:
        return

    by_url: dict[str, dict[str, Any]] = {}
    for p in validation_packet or []:
        u = (p.get("url") or "").strip()
        fu = (p.get("final_url") or "").strip()
        if u:
            by_url[u] = p
        if fu:
            by_url[fu] = p

    soft_terms = ("recently", "lately", "this month", "last month", "this quarter", "in recent")
    hard_terms = ("this week", "last week", "yesterday", "today")

    new_flags: list[dict[str, Any]] = []
    for c in claims:
        cid = c.get("claim_id") or "unknown"
        text = (c.get("claim_text") or "").lower()
        if not text:
            continue
        if not any(t in text for t in soft_terms + hard_terms):
            continue

        src = (c.get("source_url") or "").strip()
        page = by_url.get(src)
        excerpt = (page.get("text_excerpt") if page else "") or ""
        sig = guess_date_signals(excerpt)
        has_date = bool(sig["year_hits"]) or bool(sig["has_month"])

        if any(t in text for t in hard_terms) and not has_date:
            new_flags.append({
                "claim_id": cid,
                "flag_type": "hallucination",
                "detail": "Claim uses strong recency wording (e.g. 'this week') but the source text excerpt does not contain any explicit date/month/year.",
                "severity": "BLOCK",
            })
        elif any(t in text for t in soft_terms) and not has_date:
            new_flags.append({
                "claim_id": cid,
                "flag_type": "unverifiable",
                "detail": "Claim uses recency wording (e.g. 'recently') but the source text excerpt does not contain any explicit date/month/year.",
                "severity": "WARN",
            })

    if new_flags:
        merged = merge_flags(result.get("flags") or [], new_flags)
        result["flags"] = merged
        vr = compute_validation_result(merged)
        result["validation_result"] = vr
        if vr == "FAIL":
            result["approved_email"] = None


def _tokenize_key_terms(text: str) -> list[str]:
    """
    Topic-agnostic key term extraction: keep alphanumerics, drop stopwords, drop very short tokens.
    """
    t = (text or "").lower()
    toks = re.findall(r"[a-z0-9]{3,}", t)
    out: list[str] = []
    for tok in toks:
        if tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


def _apply_claim_source_alignment_rules(
    result: dict[str, Any], clean_input: dict[str, Any], validation_packet: list[dict[str, Any]]
) -> None:
    claims = clean_input.get("claims") or []
    if not claims:
        return

    # Map both original and final urls to packet records.
    by_url: dict[str, dict[str, Any]] = {}
    for p in validation_packet or []:
        u = (p.get("url") or "").strip()
        fu = (p.get("final_url") or "").strip()
        if u:
            by_url[u] = p
        if fu:
            by_url[fu] = p

    new_flags: list[dict[str, Any]] = []
    for c in claims:
        cid = c.get("claim_id") or "unknown"
        claim_text = (c.get("claim_text") or "").strip()
        src = (c.get("source_url") or "").strip()
        if not claim_text or not src:
            continue

        page = by_url.get(src)
        excerpt = ((page or {}).get("text_excerpt") or "").strip()
        if not excerpt:
            continue

        # Compute overlap between claim key terms and excerpt key terms.
        c_terms = _tokenize_key_terms(claim_text)
        if len(c_terms) < 6:
            # Too short to reliably judge; skip.
            continue
        e_terms = set(_tokenize_key_terms(excerpt))
        if not e_terms:
            continue

        hits = sum(1 for t in c_terms if t in e_terms)
        overlap = hits / max(1, len(set(c_terms)))

        # Thin excerpts (JS-walled sites, 403 blocks, redirects) naturally score low
        # because there simply isn't enough text to match against. Only BLOCK on
        # substantial pages (>= 600 chars) to avoid false positives on thin content.
        excerpt_is_substantial = len(excerpt) >= 600
        conf = float(c.get("confidence", 1.0) or 1.0)
        if overlap < 0.10:
            sev = "BLOCK" if (conf >= 0.6 and excerpt_is_substantial) else "WARN"
            new_flags.append(
                {
                    "claim_id": cid,
                    "flag_type": "unverifiable",
                    "detail": (
                        "Claim appears more specific than its cited source. "
                        "Key terms from the claim are not present in the fetched source excerpt."
                        + ("" if excerpt_is_substantial else " (Source page returned limited content.)")
                    ),
                    "severity": sev,
                }
            )
        elif overlap < 0.18:
            new_flags.append(
                {
                    "claim_id": cid,
                    "flag_type": "unverifiable",
                    "detail": "Weak claim/source alignment. Only a small fraction of the claim’s key terms appear in the fetched source excerpt.",
                    "severity": "WARN",
                }
            )

    if new_flags:
        merged = merge_flags(result.get("flags") or [], new_flags)
        result["flags"] = merged
        vr = compute_validation_result(merged)
        result["validation_result"] = vr
        if vr == "FAIL":
            result["approved_email"] = None




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agent2_validator.py output/agent1_<Name>.json")
        sys.exit(1)

    result = run_agent2(sys.argv[1])
    name = result["name"].replace(" ", "_")
    out_path = f"output/agent2_{name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
