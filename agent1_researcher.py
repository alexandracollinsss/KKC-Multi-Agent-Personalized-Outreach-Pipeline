"""
Agent 1: Researcher & Drafter
Researches a contact from minimal input, builds a profile, and drafts a personalized outreach email.
Uses live web requests (DuckDuckGo HTML search + direct page fetch) + Gemini.
"""

import os
import json
import logging
from typing import Any
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import (
    _env_float,
    _env_int,
    discover_org_domain,
    extract_json as _extract_json,
    fetch_url,
    guess_org_homepage_urls,
    is_usable_retrieval,
    recency_score,
    site_probe_urls,
    unique_urls,
    utc_now_iso,
    web_search,
)
from llm_gemini import default_model, generate_text
from claim_postprocess import patch_kkc_claim_sources, prune_unnecessary_kkc_references, strip_quotes_in_email_and_claims

logger = logging.getLogger(__name__)

_HIGH_SIGNAL_NEWS_HOSTS = (
    "cnbc.com",
    "forbes.com",
    "bloomberg.com",
    "techcrunch.com",
    "fortune.com",
    "wsj.com",
    "reuters.com",
    "businessinsider.com",
    "axios.com",
    "theinformation.com",
    "fastcompany.com",
)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_GENERIC_EMAIL_PREFIXES = {
    "hello",
    "info",
    "team",
    "press",
    "support",
    "careers",
    "jobs",
    "sales",
    "contact",
    "admin",
    "legal",
    "privacy",
    "security",
    "media",
}

def _has_affiliation_evidence(packet: list[dict[str, Any]], name: str, organization: str) -> bool:
    """
    Lightweight heuristic: do we have any retrieved excerpt that mentions both the person and org?
    This is used to stop the model from asserting employment/title at the org when the packet
    doesn't contain supporting evidence (common failure mode for ambiguous org names).
    """
    n = (name or "").strip().lower()
    o = (organization or "").strip().lower()
    if not n or not o:
        return False
    n_tokens = [t for t in re.split(r"\s+", n) if len(t) >= 3]
    o_tokens = [t for t in re.split(r"\s+", o) if len(t) >= 4]
    if not n_tokens or not o_tokens:
        return False
    for rec in packet or []:
        text = (rec.get("text_excerpt") or "").lower()
        if not text:
            continue
        if any(t in text for t in n_tokens) and any(t in text for t in o_tokens):
            return True
    return False


def _has_specific_opening_hook(result: dict[str, Any]) -> bool:
    """
    Spec guardrail: ensure the email opens with a specific, sourced hook.

    Requires at least one claim that:
    - has a non-empty source_url
    - has a source_type that reflects a concrete public artifact (podcast/panel/post/news/launch/etc.)
    - AND has meaningful keyword overlap with the email's opening 320 chars
      (token overlap >= 20% of the claim's key terms, OR verbatim 6-word fragment match)

    Using token overlap instead of strict verbatim-8-word match prevents false
    rejections when the email paraphrases rather than quotes the claim exactly.
    """
    email = (result.get("draft_email") or "").strip()
    claims = result.get("claims") or []
    if not email or not claims:
        return False

    head = email[:400].lower()
    head_terms = set(_tokenize_key_terms(head))
    allowed = {
        "conference_panel",
        "tweet",
        "linkedin_post",
        "podcast",
        "news_article",
        "product_launch",
        "other",
    }

    for c in claims:
        st = (c.get("source_type") or "").strip()
        src = (c.get("source_url") or "").strip()
        ct = (c.get("claim_text") or "").strip()
        if not src or not ct:
            continue
        if st and st not in allowed:
            continue
        # Require source_type to be one of the allowed hook types.
        # Claims with source_type=None or "org_website"/"person_profile" are affiliation
        # facts, not hooks, so they must not pass the opening-hook guard.
        if not st or st not in allowed:
            continue
        # Verbatim short-fragment check (6 words, more lenient than 8)
        frag = " ".join(ct.split()[:6]).lower()
        if frag and frag in head:
            return True
        # Token-overlap fallback: if ≥20% of the claim's key terms appear in the email head
        c_terms = set(_tokenize_key_terms(ct))
        if len(c_terms) >= 3 and head_terms:
            overlap = sum(1 for t in c_terms if t in head_terms) / len(c_terms)
            if overlap >= 0.20:
                return True
    return False


def _tokenize_key_terms(text: str) -> list[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "your", "from", "have", "has", "into",
        "their", "about", "would", "could", "really", "recent", "piece", "work", "team",
        "firm", "company", "organization", "recently", "article",
    }
    toks = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    return [t for t in toks if t not in stop]


def _normalize_url(url: str) -> str:
    """Strip trailing slash and fragment for loose matching."""
    return (url or "").rstrip("/").split("#")[0].split("?")[0].lower()


def _claim_supported_by_packet(claim_text: str, source_url: str, packet: list[dict[str, Any]]) -> bool:
    if not claim_text or not source_url:
        return False
    norm_src = _normalize_url(source_url)
    excerpt = ""
    for rec in packet or []:
        # Match on either original URL or redirected final_url, with normalization.
        rec_urls = {
            _normalize_url(rec.get("url") or ""),
            _normalize_url(rec.get("final_url") or ""),
        }
        if norm_src in rec_urls or any(norm_src in u or u in norm_src for u in rec_urls if u):
            candidate = (rec.get("text_excerpt") or "").strip()
            if len(candidate) > len(excerpt):
                excerpt = candidate
    if not excerpt:
        return False
    c_terms = _tokenize_key_terms(claim_text)
    if len(c_terms) < 4:
        return True  # Too short to reliably score; assume supported.
    e_terms = set(_tokenize_key_terms(excerpt))
    if not e_terms:
        return False
    overlap = sum(1 for t in set(c_terms) if t in e_terms) / max(1, len(set(c_terms)))
    # Thin excerpts (e.g. just a title row from a blocked page) naturally score low;
    # lower threshold proportionally so we don't reject valid hooks on thin pages.
    min_overlap = 0.12 if len(excerpt) >= 600 else 0.07
    return overlap >= min_overlap


def _has_specific_opening_hook_supported(result: dict[str, Any], packet: list[dict[str, Any]]) -> bool:
    email = (result.get("draft_email") or "").strip()
    claims = result.get("claims") or []
    if not email or not claims:
        return False
    head = email[:320].lower()
    allowed = {"conference_panel", "tweet", "linkedin_post", "podcast", "news_article", "product_launch", "other"}
    for c in claims:
        st = (c.get("source_type") or "").strip()
        src = (c.get("source_url") or "").strip()
        ct = (c.get("claim_text") or "").strip()
        if st and st not in allowed:
            continue
        if not _claim_supported_by_packet(ct, src, packet):
            continue
        frag = " ".join(ct.split()[:8]).lower()
        if frag and frag in head:
            return True
    return False


def _score_hit_record(hit: dict[str, str], name: str, organization: str, org_host: str | None) -> float:
    url = (hit.get("url") or "").strip().lower()
    title = (hit.get("title") or "").strip().lower()
    query = (hit.get("query") or "").strip().lower()
    person_tokens = [t for t in re.split(r"\s+", (name or "").lower()) if len(t) >= 3]
    org_tokens = [t for t in re.split(r"\s+", (organization or "").lower()) if len(t) >= 4]
    score = 0.0

    if query == "seed":
        score += 4.0
    if query.startswith("third-party:"):
        score += 3.5
    elif query.startswith("site-probe:"):
        score -= 2.5
    elif query.startswith("org-discovery:"):
        score += 1.0

    if any(t in title for t in person_tokens):
        score += 4.0
    if any(t in title for t in org_tokens):
        score += 2.5
    if any(t in query for t in person_tokens):
        score += 1.5
    if any(t in query for t in org_tokens):
        score += 1.0

    if any(h in url for h in _HIGH_SIGNAL_NEWS_HOSTS):
        score += 5.0
    if any(k in url for k in ("podcast", "interview", "youtube.com", "substack.com", "medium.com")):
        score += 2.5
    if "@"+(org_host or "").lstrip("www.") in title:
        score += 3.0
    if "linkedin.com/posts/" in url or "linkedin.com/feed/" in url:
        score -= 4.0

    if org_host and org_host.lower().lstrip("www.") in url:
        score += 1.5
    if any(k in url for k in ("/news", "/press", "/blog", "/stories", "/podcast", "/research")):
        score += 1.2
    if any(k in url for k in ("/contact", "/privacy", "/support", "/help", "/careers", "/jobs", "/press")):
        score += 1.0
    if any(k in url for k in ("/team", "/people", "/leadership", "/about", "/contact", "/company")):
        score -= 1.5

    return score


def _extract_email_evidence(packet: list[dict[str, Any]], name: str, organization: str, org_host: str | None) -> tuple[str | None, str | None]:
    """
    Deterministic email resolver:
    - Prefer explicit personal emails if the page also mentions the person/org.
    - Otherwise infer a pattern only when the same domain has corroborating examples.
    """
    person_tokens = [t for t in re.split(r"\s+", (name or "").lower()) if len(t) >= 2]
    org_tokens = [t for t in re.split(r"\s+", (organization or "").lower()) if len(t) >= 4]
    host_hint = (org_host or "").lower().lstrip("www.")
    first = person_tokens[0] if person_tokens else ""
    last = person_tokens[-1] if len(person_tokens) >= 2 else ""

    examples_by_domain: dict[str, list[tuple[str, str]]] = {}

    for rec in packet or []:
        text = (rec.get("text_excerpt") or "")
        if not text:
            continue
        text_l = text.lower()
        source_url = (rec.get("final_url") or rec.get("url") or "").strip()
        emails = _EMAIL_RE.findall(text)
        if not emails:
            continue
        mentions_person = all(tok in text_l for tok in person_tokens[:2]) if person_tokens else False
        mentions_org = any(tok in text_l for tok in org_tokens) if org_tokens else False
        for email in emails:
            local, _, domain = email.lower().partition("@")
            if not domain:
                continue
            if host_hint and host_hint not in domain:
                continue
            examples_by_domain.setdefault(domain, []).append((local, source_url))
            if mentions_person and (mentions_org or host_hint in domain):
                return email, source_url

    for domain, examples in examples_by_domain.items():
        unique_examples = list(dict.fromkeys(examples))
        if len(unique_examples) < 1 or not first or not last:
            continue
        locals_only = [local for local, _ in unique_examples]
        patterns: list[str] = []
        for local in locals_only:
            if local in _GENERIC_EMAIL_PREFIXES:
                continue
            if "." in local:
                parts = local.split(".")
                if len(parts) == 2:
                    if parts[0].isalpha() and parts[1].isalpha():
                        patterns.append("first.last")
                    if len(parts[0]) == 1 and parts[1].isalpha():
                        patterns.append("f.last")
            elif "_" in local:
                parts = local.split("_")
                if len(parts) == 2 and parts[0].isalpha() and parts[1].isalpha():
                    patterns.append("first_last")
            elif "-" in local:
                parts = local.split("-")
                if len(parts) == 2 and parts[0].isalpha() and parts[1].isalpha():
                    patterns.append("first-last")
            elif local.startswith(first[:1]) and local.endswith(last) and len(local) <= len(last) + 2:
                patterns.append("flast")
            elif local == f"{first}{last}":
                patterns.append("firstlast")
            elif local == first:
                patterns.append("first")
        if not patterns:
            continue
        patt = max(set(patterns), key=patterns.count)
        candidate_local = {
            "first.last": f"{first}.{last}",
            "f.last": f"{first[:1]}.{last}",
            "first_last": f"{first}_{last}",
            "first-last": f"{first}-{last}",
            "flast": f"{first[:1]}{last}",
            "firstlast": f"{first}{last}",
            "first": first,
        }.get(patt)
        if candidate_local:
            return f"{candidate_local}@{domain}", unique_examples[0][1]

    return None, None

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

EMAIL_POLISH_SYSTEM_PROMPT = """You are an expert outreach copyeditor.

You will receive an Agent 1 JSON draft (including retrieval sources and claims). Improve the EMAIL QUALITY while obeying strict constraints:

Hard constraints:
- Return ONLY valid JSON (no markdown).
- Do NOT add new factual claims that are not already present in the input `claims` list.
- Keep every factual statement in the email grounded in the provided sources.
- Keep the email 150–220 words, human, specific, and non-templated.
- No quotation marks at all in the email.
- Preserve the ask: 30-minute call.

Style goals:
- Punchier opening (1–2 sentences), concrete reference, then a crisp why-now + why-you.
- Reduce buzzwords and generic VC language.
- Vary sentence length; avoid long paragraphs.
- Make the CTA low-friction (2 time windows or “happy to work around your schedule”).

Output schema (only these keys):
{
  "draft_email": "string",
  "claims": [ ... same schema as input claims, but with claim_text updated to match the improved email ... ]
}
"""

EMAIL_POLISH_USER_TEMPLATE = """Improve this outreach email without adding any new factual claims.

INPUT JSON:
{agent1_json}
"""

RESEARCH_SYSTEM_PROMPT = """You are a meticulous research analyst for Kyber Knight Capital (KKC), a pre-seed and seed-stage venture firm focused on commerce, AI, and labor.

Your job is to deeply research a contact and return ONLY a valid JSON object — no markdown, no preamble, no explanation.

You will be given a retrieval packet containing live-fetched web pages (URLs + excerpts). Use it to find:
- The person's current title on the organization's own website (not just LinkedIn)
- Any public statements: panel quotes, podcast appearances, published essays, tweets/X posts, LinkedIn posts
- Recent news about their organization
- Their professional email (infer from domain convention if you find team pages)
- Any signals about their investment focus or interests

STRICT RULES:
1. Every claim you make must be grounded in a URL that appears in the retrieval packet. If you cannot verify something, say so explicitly — do NOT fabricate.
2. If a piece of information is unavailable, set it to null. Never guess.
3. The opening hook in the email must cite something SPECIFIC this person said or did — a real quote, a real panel, a real post. No generic flattery.
4. The email must be 150–250 words, sound human, include a clear ask (30-minute call), and must NOT read as a template. Vary sentence length. Avoid buzzwords like "synergy", "leverage", "exciting opportunity".
5. The email MUST include a brief, specific connection between KKC's work (pre-seed/seed venture, focus on commerce, AI, and labor) and something specific about this person or their organization. This connection should feel purposeful, not generic — avoid boilerplate like "we invest in AI" in isolation. Instead, make the link concrete: e.g., "Given KKC's focus on AI-native commerce tooling, your work on X is directly relevant." Do NOT write a long paragraph about KKC; one or two tight sentences connecting KKC's thesis to the person are enough.
6. Set confidence scores honestly: 1.0 = you retrieved and read the source; 0.7 = plausible inference from context; 0.4 = weak signal; <0.3 = speculation.
7. CRITICAL — claim/source alignment: each `claims[].source_url` must be a page whose excerpt ACTUALLY contains support for that exact `claim_text`. Never cite the contact's organization website as the source for sentences that are ONLY about Kyber Knight Capital (KKC). In general, avoid KKC-only factual claims in the email body unless they are essential and directly supported by a cited source.
8. Email: set `resolved_email` to null unless you can point to a team/contact page in the packet that shows the address OR a clear, repeated naming convention plus a corroborating example email on that same domain. Do not invent `first@company.com` from the domain alone.
9. Title: `resolved_title` must match wording found on the organization's own site in the packet when possible; if only LinkedIn supports the title, set `resolved_title` from LinkedIn but lower confidence and add a short caveat in `research_sources`.
10. If two domains appear (e.g. `company.com` vs `company.vc`), prefer the one whose homepage content clearly matches THIS organization (portfolio, team names) before inferring email.
11. STYLE: Do NOT use quotation marks in the email (no `"..."` and no curly quotes). When the best hook is something the person said publicly, reference it specifically but paraphrase it in your own words so the email stays natural and human.
12. IMPORTANT JSON: avoid unescaped double-quotes in `draft_email`. Prefer paraphrase with no quotes; if absolutely necessary, escape quotes properly.
13. Recency: Prefer a hook from the most recent available source. Do not use vague recency language like "recently" / "last week" / "this month" unless the source text shows an explicit date you can include in the claim.
13b. NO INVENTED SPECIFICS: Never fabricate specific years ("2025"), document titles ("year-end letter", "Q3 report"), exact quoted phrases, or named events that do not appear verbatim or near-verbatim in the source excerpt. If a source mentions "an annual letter" but not the year, write "annual letter" — not "2025 letter". When in doubt, stay at the level of detail the source actually supports.
14. CRITICAL: Never mention web scraping, retrieval packets, access errors, paywalls, 403/401/429, robots, rate limits, or any internal process in the email. If sources are thin, write a normal, concise outreach email that does NOT reference the failure; simply keep it higher-level and avoid specific claims.
15. Signature: End the email with a normal sign-off using the provided sender name and sender role. Do not leave placeholders like [My Name].
16. Affiliation accuracy: Do NOT state or imply the person works at the provided organization unless a source in the retrieval packet explicitly supports that (e.g., org site team page listing them, or a credible third-party profile explicitly stating their role at that org). If affiliation cannot be verified, keep `resolved_title` and `resolved_email` null and write the email so it does NOT assume they are at that org (you may ask if they are the right person to speak with).
17. SPEC REQUIREMENT (non-negotiable): If you cannot find a specific sourced hook for the opening (panel/podcast/interview/post/launch) in the retrieval packet, set:
   - `draft_email`: null
   - `claims`: []
   and add a `research_sources` entry explaining that no specific public hook was found. Do NOT generate a generic email in that case.

Return ONLY this JSON schema (no other text):
{
  "name": "string",
  "organization": "string",
  "resolved_title": "string or null",
  "resolved_email": "string or null",
  "email_domain_source": "string (URL) or null",
  "draft_email": "string",
  "research_sources": [
    {
      "url": "string",
      "content_summary": "string",
      "retrieved_at": "ISO8601 string"
    }
  ],
  "claims": [
    {
      "claim_id": "string (c1, c2, ...)",
      "claim_text": "string (exact text from the email this claim supports)",
      "source_url": "string",
      "source_type": "conference_panel | tweet | linkedin_post | org_website | news_article | podcast | product_launch | other",
      "confidence": float
    }
  ]
}"""


RESEARCH_USER_TEMPLATE = """Research this contact and produce the output JSON:

Name: {name}
Title (provided): {title}
Organization: {organization}
LinkedIn URL: {linkedin_url}

Sender name (for signature): {sender_name}
Sender role (for signature): {sender_role}

DISCOVERED_ORG_DOMAIN (heuristic; may be null): {discovered_org_domain}
AFFILIATION_EVIDENCE_IN_PACKET: {affiliation_evidence}

RETRIEVAL PACKET (live-fetched, use only these URLs as sources):
{retrieval_packet}

Steps to follow:
1. Search for "{name} {organization}" to find their current role and any public presence.
2. If DISCOVERED_ORG_DOMAIN is set, prioritize that domain for title + email-domain evidence (team/about pages). Otherwise, search broadly for "{organization}" website to confirm title and find the email domain/naming convention.
3. Search for "{name}" podcast, interviews, conference panels, tweets, or published writing.
4. Search for recent news about "{organization}".
5. Draft the email using only verified facts.
6. Build the claims array — one claim per factual statement in the email body.

Remember: if you cannot find a reliable source for something, do NOT include it. Surface the gap instead."""


def _parallel_fetch(
    urls: list[str],
    *,
    timeout_s: float,
    max_chars: int,
    workers: int,
    min_chars: int,
    pass_cap: int,
    fetched: list[dict[str, Any]],
    usable: list[dict[str, Any]],
) -> None:
    """
    Fetch `urls` concurrently, appending results into `fetched` and `usable`.
    Cancels pending (not-yet-started) futures and shuts down quickly once
    `pass_cap` usable sources are collected.
    """
    if not urls:
        return
    ex = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futs = {ex.submit(fetch_url, u, timeout_s=timeout_s, max_chars=max_chars): u for u in urls}
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
            rec["_score"] = recency_score(
                rec.get("final_url") or rec.get("url") or u,
                rec.get("text_excerpt") or "",
            )
            fetched.append(rec)
            if is_usable_retrieval(rec, min_chars=min_chars):
                usable.append(rec)
            if len(usable) >= pass_cap:
                break
    finally:
        # cancel_futures=True cancels queued-but-not-started futures so we don't
        # wait 12s × (N-workers) extra when we already have enough sources.
        ex.shutdown(wait=True, cancel_futures=True)


def build_retrieval_packet(
    name: str, organization: str, linkedin_url: str | None
) -> tuple[list[dict[str, Any]], str | None]:
    fetch_cap = _env_int("AGENT1_FETCH_CAP", 28)
    pass_cap = _env_int("AGENT1_USABLE_SOURCES", 18)
    max_chars = _env_int("FETCH_MAX_CHARS", 12000)
    timeout_s = _env_float("FETCH_TIMEOUT_S", 20.0)
    min_chars = _env_int("FETCH_MIN_CHARS", 340)
    search_max_results = _env_int("SEARCH_MAX_RESULTS", 6)
    fetch_workers = _env_int("FETCH_WORKERS", 8)

    org_host, discovery_hits = discover_org_domain(
        organization,
        timeout_s=min(18.0, timeout_s),
        max_chars=min(8000, max_chars),
    )

    seed_urls: list[str] = []
    if linkedin_url and linkedin_url.strip():
        seed_urls.append(linkedin_url.strip())
    if org_host:
        seed_urls.append(f"https://{org_host}/")
    else:
        seed_urls.extend(guess_org_homepage_urls(organization))

    queries = [
        f"{name} {organization}",
        f"{name} {organization} interview 2026",
        f"{name} {organization} podcast 2026",
        f"{name} podcast interview essay",
        f"{name} Substack Medium op-ed",
        f"{organization} news 2026",
        f"{organization} announcement 2026",
        f"{organization} press 2025",
    ]
    if linkedin_url and linkedin_url.strip():
        queries.append(linkedin_url.strip())

    site_queries: list[str] = []
    if org_host:
        site_queries = [
            f"site:{org_host} {name}",
            f"site:{org_host} team",
            f"site:{org_host} news 2026",
            f"site:{org_host} contact",
        ]

    all_queries = queries + site_queries
    search_timeout = min(10.0, timeout_s)

    def _run_search(q: str) -> list[dict[str, str]]:
        try:
            return [{**h, "query": q} for h in web_search(q, max_results=search_max_results, timeout_s=search_timeout)]
        except Exception as e:
            logger.warning(f"[Agent1] Search failed for query='{q}': {e}")
            return []

    hits: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=6) as sex:
        search_futs = {sex.submit(_run_search, q): q for q in all_queries}
        for fut in as_completed(search_futs):
            hits.extend(fut.result())

    hit_records: list[dict[str, str]] = []
    for u in seed_urls:
        hit_records.append({"title": "seed", "url": u, "query": "seed"})
    for h in discovery_hits:
        hit_records.append(h)
    if org_host:
        for u in site_probe_urls(org_host):
            hit_records.append({"title": "site-probe", "url": u, "query": f"site-probe:{org_host}"})
    for h in hits:
        hit_records.append({"title": h.get("title", ""), "url": h.get("url", ""), "query": h.get("query") or "search"})

    hit_records.sort(key=lambda h: _score_hit_record(h, name, organization, org_host), reverse=True)
    urls = unique_urls(hit_records)

    if not urls:
        logger.warning("[Agent1] No search URLs returned — retrieval packet will be empty unless seeds fetch.")

    fetched: list[dict[str, Any]] = []
    usable: list[dict[str, Any]] = []

    # Third-party queries used if initial retrieval comes up thin.
    third_party_queries: list[str] = [
        f"{name} interview 2026",
        f"{name} podcast 2026",
        f"{name} panel 2026",
        f"{name} keynote 2026",
        f"{name} fireside chat 2026",
        f"{name} speech 2026",
        f"{name} op-ed 2026",
        f"{name} essay 2026",
        f"{name} keynote 2025",
        f"{name} fireside chat 2025",
        f"{name} op-ed 2025",
        f"{name} essay 2025",
        f"{organization} announcement 2026",
        f"{organization} funding 2026",
        f"{organization} product launch 2026",
        f"{organization} founder interview 2026",
        f"{organization} founder profile 2026",
        f"site:youtube.com {name} {organization}",
        f"site:substack.com {name}",
        f"site:medium.com {name} {organization}",
        f"site:podcasts.apple.com {name}",
        f"site:open.spotify.com {name} podcast",
    ]

    max_to_fetch = fetch_cap * 2
    _parallel_fetch(
        urls[:max_to_fetch],
        timeout_s=timeout_s, max_chars=max_chars, workers=fetch_workers,
        min_chars=min_chars, pass_cap=pass_cap,
        fetched=fetched, usable=usable,
    )

    # If we still have too few usable sources OR the usable list is all org-site pages
    # (which won't contain any hooks), expand via third-party searches (parallel).
    # This ensures we always try to find podcast/media/interview appearances even when
    # the org's own site fills the usable cap with thin navigation pages.
    non_org_usable = sum(
        1 for r in usable
        if not (org_host and org_host.lower().lstrip("www.") in (r.get("url") or "").lower())
    )
    if len(usable) < max(4, pass_cap // 2) or non_org_usable < max(2, pass_cap // 4):
        def _tp_search(q: str) -> list[dict[str, str]]:
            try:
                return [
                    {"title": h.get("title", ""), "url": h.get("url", ""), "query": f"third-party:{q}"}
                    for h in web_search(q, max_results=search_max_results, timeout_s=search_timeout)
                    if h.get("url")
                ]
            except Exception as e:
                logger.warning(f"[Agent1] Third-party search failed: {e}")
                return []

        third_party_hits: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=6) as tpex:
            tp_futs = {tpex.submit(_tp_search, q): q for q in third_party_queries}
            for fut in as_completed(tp_futs):
                for rec in fut.result():
                    hit_records.append(rec)
                    third_party_hits.append(rec)

        seen_fetch: set[str] = {(r.get("url") or "") for r in fetched}
        third_party_hits.sort(key=lambda h: _score_hit_record(h, name, organization, org_host), reverse=True)
        more = [
            (h.get("url") or "").strip()
            for h in third_party_hits
            if (h.get("url") or "").strip() and (h.get("url") or "").strip() not in seen_fetch
        ][: max(0, max_to_fetch - len(fetched))]
        _parallel_fetch(
            more,
            timeout_s=timeout_s, max_chars=max_chars, workers=fetch_workers,
            min_chars=min_chars, pass_cap=pass_cap,
            fetched=fetched, usable=usable,
        )

    fetched.sort(key=lambda r: float(r.get("_score") or 0.0), reverse=True)
    packet: list[dict[str, Any]] = []
    for rec in fetched:
        if len(packet) >= pass_cap:
            break
        if not is_usable_retrieval(rec, min_chars=min_chars):
            continue
        packet.append({k: v for k, v in rec.items() if k != "_score"} | {"recency_score": rec.get("_score")})
    return packet, org_host


def run_agent1(contact: dict, model: str = default_model()) -> dict:
    """
    Run Agent 1 on a single contact dict.
    Returns the structured research + draft JSON.
    """
    name = contact.get("name", "")
    title = contact.get("title", "")
    organization = contact.get("organization", "")
    linkedin_url = contact.get("linkedin_url")
    sender_name = (
        contact.get("sender_name") or os.environ.get("SENDER_NAME", "").strip() or "Alex"
    )
    sender_role = (
        contact.get("sender_role") or os.environ.get("SENDER_ROLE", "").strip() or "Kyber Knight Capital"
    )

    retrieval, discovered_org_domain = build_retrieval_packet(name=name, organization=organization, linkedin_url=linkedin_url)
    retrieval_packet = json.dumps(retrieval, indent=2)
    affiliation_evidence = "yes" if _has_affiliation_evidence(retrieval, name, organization) else "no"

    user_message = RESEARCH_USER_TEMPLATE.format(
        name=name,
        title=title,
        organization=organization,
        linkedin_url=linkedin_url or "Not provided",
        sender_name=sender_name or "Not provided",
        sender_role=sender_role or "Not provided",
        discovered_org_domain=discovered_org_domain or "null",
        affiliation_evidence=affiliation_evidence,
        retrieval_packet=retrieval_packet,
    )

    logger.info(f"[Agent1] Starting research for {name} @ {organization}")

    try:
        full_text, model_used = generate_text(
            system=RESEARCH_SYSTEM_PROMPT,
            user=user_message,
            model=model,
        )
        logger.info(f"[Agent1] Got response for {name}")

        try:
            result = _extract_json(full_text)
        except Exception as e:
            logger.warning(f"[Agent1] JSON parse failed for {name}, attempting repair: {e}")
            repair_user = JSON_REPAIR_USER_TEMPLATE.format(broken=full_text[:12000])
            repaired_text, _ = generate_text(system=JSON_REPAIR_SYSTEM_PROMPT, user=repair_user, model=model)
            result = _extract_json(repaired_text)

        # Style hardening: remove quotation marks deterministically
        cleaned_email, _ = strip_quotes_in_email_and_claims(result.get("draft_email"), result.get("claims"))
        result["draft_email"] = cleaned_email

        # Optional: quality polish pass (no new facts; updates claim_texts to match)
        if os.environ.get("EMAIL_POLISH", "1") != "0":
            try:
                clean_for_polish = {k: v for k, v in result.items() if not k.startswith("_")}
                polish_user = EMAIL_POLISH_USER_TEMPLATE.format(agent1_json=json.dumps(clean_for_polish, indent=2))
                polished_text, _ = generate_text(system=EMAIL_POLISH_SYSTEM_PROMPT, user=polish_user, model=model)
                polished = _extract_json(polished_text)
                if isinstance(polished, dict) and polished.get("draft_email") and isinstance(polished.get("claims"), list):
                    # Strip quotes again just in case.
                    pe, _ = strip_quotes_in_email_and_claims(polished.get("draft_email"), polished.get("claims"))
                    result["draft_email"] = pe
                    result["claims"] = polished.get("claims")
            except Exception as e:
                logger.warning(f"[Agent1] Polish pass failed for {name}: {e}")

        # Deterministically resolve a better-grounded email when possible.
        det_email, det_email_src = _extract_email_evidence(retrieval, name, organization, discovered_org_domain)
        if det_email:
            result["resolved_email"] = det_email
            result["email_domain_source"] = det_email_src
        elif not result.get("resolved_email"):
            result["resolved_email"] = None
            result["email_domain_source"] = None

        # Drop sender boilerplate that is not needed to support recipient-specific outreach.
        prune_unnecessary_kkc_references(result)

        # Re-point any remaining KKC claims to the canonical KKC profile URL so Agent 2
        # can verify them (otherwise they're cited against the contact's org site and blocked).
        kkc_profile_url = os.environ.get("KKC_PROFILE_URL", "").strip() or None
        if kkc_profile_url:
            patch_kkc_claim_sources(result.get("claims"), kkc_profile_url)

        # Inject metadata
        result["_agent"] = "agent1"
        result["_model"] = model_used
        result["_processed_at"] = utc_now_iso()
        result["_input_contact"] = contact
        result["_retrieval_packet"] = retrieval

        # Spec requirement: no specific sourced opening hook → null email.
        has_hook = _has_specific_opening_hook(result)
        has_hook_supported = _has_specific_opening_hook_supported(result, retrieval)

        if not has_hook:
            # No hook of any kind in the email opening — spec requires null.
            result["draft_email"] = None
            result["claims"] = []
            result["resolved_title"] = None
            result["resolved_email"] = None
            result["email_domain_source"] = None
            result["research_sources"] = (result.get("research_sources") or []) + [
                {
                    "url": None,
                    "content_summary": "No specific public hook (panel/podcast/interview/post/launch) was found in the retrieval packet, so no outreach email was drafted (spec requirement).",
                    "retrieved_at": utc_now_iso(),
                }
            ]
        elif not has_hook_supported:
            # A hook exists but the source page content is too thin to confirm it.
            # Don't null the email — instead flag the hook claim as low-confidence so
            # Agent 2 can review. This prevents false rejections when pages are
            # inaccessible or return minimal text (403, JS-walled, redirect, etc.).
            claims = result.get("claims") or []
            head = (result.get("draft_email") or "")[:320].lower()
            allowed = {"conference_panel", "tweet", "linkedin_post", "podcast", "news_article", "product_launch", "other"}
            for c in claims:
                st = (c.get("source_type") or "").strip()
                ct = (c.get("claim_text") or "").strip()
                if st not in allowed:
                    continue
                frag = " ".join(ct.split()[:8]).lower()
                if frag and frag in head:
                    # Cap at 0.55 so Agent 2's confidence floor triggers a WARN (not BLOCK).
                    c["confidence"] = min(float(c.get("confidence") or 0.7), 0.55)
            logger.info(f"[Agent1] Hook present but thin source coverage for {name} — downgraded hook claim confidence; Agent 2 will review.")

        logger.info(f"[Agent1] Done for {name}: {len(result.get('claims', []))} claims, {len(result.get('research_sources', []))} sources")
        return result

    except Exception as e:
        logger.error(f"[Agent1] FAILED for {name}: {e}")
        return {
            "name": name,
            "organization": organization,
            "error": str(e),
            "resolved_title": None,
            "resolved_email": None,
            "email_domain_source": None,
            "draft_email": None,
            "research_sources": [],
            "claims": [],
            "_agent": "agent1",
            "_model": model,
            "_processed_at": utc_now_iso(),
            "_input_contact": contact,
            "_retrieval_packet": retrieval if "retrieval" in locals() else [],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open("test_inputs.json") as f:
        contacts = json.load(f)

    for contact in contacts:
        result = run_agent1(contact)
        out_path = f"output/agent1_{contact['name'].replace(' ', '_')}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {out_path}")
