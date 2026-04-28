"""
Shared utilities used by both agents.
Kept in a separate module so tests can import them without triggering the google-genai SDK import.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone
from html import unescape
from typing import Any, Optional
from urllib.parse import parse_qs, quote, quote_plus, unquote, urlparse

import httpx


def extract_json(text: str) -> dict:
    """Extract and parse a JSON object from LLM output, handling common wrapping patterns."""
    if not text or not text.strip():
        raise ValueError("Empty response from model")

    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Find the outermost JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    json_str = cleaned[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common issues: trailing commas
        json_str_fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            return json.loads(json_str_fixed)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse JSON: {e}\nRaw: {json_str[:500]}")


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_int(name: str, default: int) -> int:
    """Read an integer from an environment variable with a fallback default."""
    try:
        v = (os.environ.get(name) or "").strip()
        return int(v) if v else default
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float from an environment variable with a fallback default."""
    try:
        v = (os.environ.get(name) or "").strip()
        return float(v) if v else default
    except Exception:
        return default


def _strip_html(html: str) -> str:
    # Remove scripts/styles then strip tags.
    html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
)


def guess_date_signals(text: str) -> dict[str, Any]:
    """
    Extract lightweight recency signals from text (no heavy parsing):
    - year_hits: list of years found (e.g. 2026)
    - has_month: whether a month name appears
    """
    t = (text or "").lower()
    years = sorted({int(y) for y in re.findall(r"\b(20\d{2})\b", t)})
    has_month = any(m in t for m in _MONTHS)
    return {"year_hits": years, "has_month": has_month}


def recency_score(url: str, text: str) -> float:
    """
    Heuristic: prefer sources that look recent or news-like.
    Higher is better.
    """
    u = (url or "").lower()
    sig = guess_date_signals(text)
    years: list[int] = sig["year_hits"]

    score = 0.0
    # URL path hints
    if any(k in u for k in ("/news", "/press", "/blog", "/posts", "/podcast", "/events", "/announc")):
        score += 2.0
    if any(k in u for k in ("medium.com", "substack.com")):
        score += 1.5
    if "linkedin.com" in u:
        score += 0.5  # often walled, but can still be relevant

    # Date hints in URL
    if re.search(r"/20\d{2}/", u) or re.search(r"[-_/]20\d{2}[-_/]", u):
        score += 1.5

    # Date hints in text
    if years:
        score += 0.5 * len(years)
        # Boost if current-ish year appears
        if max(years) >= 2025:
            score += 2.0
    if sig["has_month"]:
        score += 1.0

    # Penalize thin pages
    if len((text or "").strip()) < 350:
        score -= 1.5
    return score


def _needs_reader_fallback(url: str, status_code: int | None, text: str) -> bool:
    """Heuristic: JS walls / login pages / tiny text → try reader proxy."""
    u = (url or "").lower()
    t = (text or "").lower()
    if status_code and int(status_code) >= 400:
        return True
    if "linkedin.com" in u or "twitter.com" in u or "x.com" in u:
        return True
    if len(t) < 220:
        return True
    wall_hints = (
        "sign in",
        "log in",
        "authwall",
        "captcha",
        "enable javascript",
        "access denied",
        "just a moment",
        "robot check",
    )
    return any(h in t for h in wall_hints)


def is_usable_retrieval(rec: dict[str, Any], *, min_chars: int = 260) -> bool:
    """
    Return True if a fetched record contains enough usable text to support sourcing.
    We treat hard blocks (401/403) and thin/empty pages as not usable.
    """
    if not rec:
        return False
    text = (rec.get("text_excerpt") or "").strip()
    if not text or len(text) < min_chars:
        return False

    # If we successfully extracted substantial text (often via reader proxies), treat as usable even if the
    # original origin returned 403 to direct HTTP clients.
    sc = rec.get("status_code")
    final_u = (rec.get("final_url") or "").lower()
    if sc in (401, 403) and len(text) >= min_chars and ("jina.ai" in final_u or rec.get("reader_used")):
        return True

    if sc in (401, 403):
        return False

    # Any other HTTP errors (404, 410, 5xx, etc.) should not be treated as usable evidence,
    # even if the error page has plenty of text.
    try:
        if sc is not None and int(sc) >= 400:
            return False
    except Exception:
        pass

    lower = text.lower()
    if any(x in lower for x in ("access denied", "forbidden", "enable javascript", "just a moment", "captcha")):
        return False
    return True


def fetch_url(url: str, *, timeout_s: float = 20.0, max_chars: int = 12000) -> dict[str, Any]:
    """
    Fetch a URL (live HTTP request) and return a compact record usable by the agents.
    On thin/login-walled HTML, retries via r.jina.ai reader for a text-first view.
    Returns:
      { url, status_code, final_url, retrieved_at, text_excerpt, reader_used? }
    """
    cache_enabled = (os.environ.get("FETCH_CACHE_ENABLED", "0") != "0")
    ttl_s = int(os.environ.get("FETCH_CACHE_TTL_SECONDS", "3600") or "3600")
    cache_dir = Path(os.environ.get("FETCH_CACHE_DIR", "output/cache"))
    cache_max_files = int(os.environ.get("FETCH_CACHE_MAX_FILES", "800") or "800")
    cache_path: Path | None = None

    if cache_enabled and url:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha256(url.encode("utf-8")).hexdigest()
            cache_path = cache_dir / f"{key}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text())
                cached_at = cached.get("cached_at")
                if cached_at:
                    try:
                        ts = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
                        age = (datetime.now(timezone.utc) - ts).total_seconds()
                        if age <= ttl_s:
                            payload = cached.get("payload")
                            if isinstance(payload, dict) and not is_usable_retrieval(payload, min_chars=120):
                                pass
                            else:
                                return payload
                    except Exception:
                        pass
        except Exception:
            cache_path = None

    retrieved_at = utc_now_iso()
    reader_used = False

    def _do_get(target: str, *, headers: dict[str, str]) -> tuple[int | None, str, str]:
        with httpx.Client(timeout=timeout_s, headers=headers, follow_redirects=True) as client:
            resp = client.get(target)
        content_type = resp.headers.get("content-type", "")
        raw = resp.text if "text" in content_type or "html" in content_type or content_type == "" else ""
        text = _strip_html(raw) if raw else ""
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        return resp.status_code, str(resp.url), text

    def _reader_urls(target_url: str) -> list[str]:
        """Build reader proxy URL list. Env var FETCH_READER_URL_TEMPLATES adds extras."""
        extra = (os.environ.get("FETCH_READER_URL_TEMPLATES") or "").strip()
        out: list[str] = []
        for part in (extra.split(",") if extra else []):
            p = part.strip()
            if p:
                try:
                    out.append(p.format(url=target_url))
                except Exception:
                    pass
        out.append(f"https://r.jina.ai/{target_url}")
        seen: set[str] = set()
        return [u for u in out if u not in seen and not seen.add(u)]  # type: ignore[func-returns-value]

    try:
        if "linkedin.com" in (url or "").lower():
            timeout_s = min(timeout_s, 12.0)

        header_variants: list[dict[str, str]] = []
        header_variants.append(dict(DEFAULT_HEADERS))
        header_variants.append(
            {
                **DEFAULT_HEADERS,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        status_code = None
        final_url = None
        text = ""
        for hv in header_variants:
            status_code, final_url, text = _do_get(url, headers=hv)
            if status_code not in (401, 403) and (text and len(text) >= 160):
                break

        if _needs_reader_fallback(url, status_code, text):
            reader_used = True
            best_sc, best_fu, best_t = status_code, final_url, text
            for reader_url in _reader_urls(url):
                rs, rf, t2 = _do_get(reader_url, headers=header_variants[-1])
                if len(t2) > len(best_t) + 40:
                    best_sc, best_fu, best_t = rs, rf, t2
            if len(best_t) > len(text) + 40 or status_code in (401, 403) or len(text) < 220:
                status_code, final_url, text = best_sc, best_fu, best_t

        payload = {
            "url": url,
            "status_code": status_code,
            "final_url": final_url,
            "retrieved_at": retrieved_at,
            "text_excerpt": text,
            "reader_used": reader_used,
        }

        should_cache = True
        if payload.get("status_code") in (401, 403) and len((payload.get("text_excerpt") or "").strip()) < 220:
            should_cache = False
        if not (payload.get("text_excerpt") or "").strip():
            should_cache = False

        if cache_enabled and cache_path is not None and should_cache:
            try:
                cache_path.write_text(json.dumps({"cached_at": utc_now_iso(), "payload": payload}))
                if cache_max_files > 0:
                    try:
                        files = [p for p in cache_dir.glob("*.json") if p.is_file()]
                        if len(files) > cache_max_files:
                            files.sort(key=lambda p: p.stat().st_mtime)
                            for p in files[: max(0, len(files) - cache_max_files)]:
                                try:
                                    p.unlink()
                                except Exception:
                                    pass
                    except Exception:
                        pass
            except Exception:
                pass
        return payload
    except Exception as e:
        payload = {
            "url": url,
            "status_code": None,
            "final_url": None,
            "retrieved_at": retrieved_at,
            "text_excerpt": "",
            "error": str(e),
            "reader_used": reader_used,
        }
        if cache_enabled and cache_path is not None:
            try:
                cache_path.write_text(json.dumps({"cached_at": utc_now_iso(), "payload": payload}))
            except Exception:
                pass
        return payload


def _fetch_html(url: str, *, timeout_s: float) -> str:
    """Fetch HTML; on DDG blocks/throttling, fall back through a reader proxy."""
    with httpx.Client(timeout=timeout_s, headers=DEFAULT_HEADERS, follow_redirects=True) as client:
        r = client.get(url)
        # DDG lite sometimes returns 202 (throttling) — proxy often works better.
        if r.status_code in (202, 403):
            proxy_url = f"https://r.jina.ai/{url}"
            r = client.get(proxy_url)
        r.raise_for_status()
        return r.text


def _parse_ddg_lite(html: str, *, max_results: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for m in re.finditer(r'(?is)<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html):
        href = unescape(m.group(1)).strip()
        title = _strip_html(m.group(2))
        if not href or not title:
            continue

        url: Optional[str] = None

        if href.startswith("/l/?") or "duckduckgo.com/l/?" in href:
            parsed = urlparse(href if href.startswith("http") else f"https://duckduckgo.com{href}")
            qs = parse_qs(parsed.query)
            uddg = (qs.get("uddg") or [None])[0]
            if uddg:
                url = unquote(uddg)
        elif href.startswith("//"):
            url = "https:" + href
        elif href.startswith("http://") or href.startswith("https://"):
            url = href

        if not url:
            continue
        if "duckduckgo.com" in urlparse(url).netloc:
            continue

        results.append({"title": title, "url": url})
        if len(results) >= max_results:
            break
    return results


def _parse_bing(html: str, *, max_results: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    # Typical pattern: <li class="b_algo"><h2><a href="URL">Title</a>
    for m in re.finditer(
        r'(?is)<li[^>]+class="[^"]*b_algo[^"]*"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html,
    ):
        href = unescape(m.group(1)).strip()
        title = _strip_html(m.group(2))
        if not href or not title:
            continue
        if not (href.startswith("http://") or href.startswith("https://")):
            continue
        if "bing.com" in urlparse(href).netloc:
            continue
        results.append({"title": title, "url": href})
        if len(results) >= max_results:
            break
    return results


def ddg_search(query: str, *, max_results: int = 5, timeout_s: float = 20.0) -> list[dict[str, str]]:
    """
    Lightweight web search (no API key): DuckDuckGo Lite + Bing fallback.
    Returns: { title, url }
    """
    q = quote_plus(query)
    # Prefer the "lite" endpoint which tends to be less strict than /html/.
    search_url = f"https://duckduckgo.com/lite/?q={q}"

    html = ""
    try:
        html = _fetch_html(search_url, timeout_s=timeout_s)
    except Exception:
        html = ""

    results = _parse_ddg_lite(html, max_results=max_results)
    if results:
        return results

    # Fallback: Bing HTML search
    try:
        bing_html = _fetch_html(f"https://www.bing.com/search?q={q}", timeout_s=timeout_s)
        return _parse_bing(bing_html, max_results=max_results)
    except Exception:
        return []



def jina_reader_search(query: str, *, max_results: int = 5, timeout_s: float = 20.0) -> list[dict[str, str]]:
    """
    Jina Reader search endpoint (no API key). Returns cleaned pages for top results.
    Docs: https://github.com/jina-ai/reader (search via https://s.jina.ai/<query>)
    """
    # s.jina.ai expects the query as a path segment; use full URL encoding (not form-style + for spaces).
    q = quote((query or "").strip(), safe="")
    url = f"https://s.jina.ai/{q}"

    try:
        with httpx.Client(timeout=timeout_s, headers=DEFAULT_HEADERS, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            text = resp.text
    except Exception:
        return []

    # Heuristic parse: look for markdown links [title](url) which Reader commonly emits.
    out: list[dict[str, str]] = []
    for m in re.finditer(r"\[([^\]]+)\]\((https?://[^)]+)\)", text):
        title = (m.group(1) or "").strip()
        u = (m.group(2) or "").strip()
        if title and u:
            out.append({"title": title, "url": u})
        if len(out) >= max_results:
            break

    # Fallback: raw URLs if markdown isn't present
    if not out:
        for m in re.finditer(r"(https?://[^\s\)\]]+)", text):
            u = (m.group(1) or "").strip().rstrip(").,]")
            if u.startswith("http"):
                out.append({"title": u, "url": u})
            if len(out) >= max_results:
                break

    return out


def serper_search(query: str, *, max_results: int = 5, timeout_s: float = 20.0) -> list[dict[str, str]]:
    """
    Serper.dev Google Search API (optional). Set SERPER_API_KEY in the environment.
    Free tier: 2500 queries/month. Sign up at https://serper.dev
    """
    key = (os.environ.get("SERPER_API_KEY") or "").strip()
    if not key:
        return []
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
            resp = client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": max_results},
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return []
    out: list[dict[str, str]] = []
    for r in data.get("organic") or []:
        u = (r.get("link") or "").strip()
        title = (r.get("title") or "").strip()
        if u and title:
            out.append({"title": title, "url": u})
        if len(out) >= max_results:
            break
    return out


def web_search(query: str, *, max_results: int = 5, timeout_s: float = 20.0) -> list[dict[str, str]]:
    """
    Primary search entrypoint. Priority: Serper → DDG/Bing → Jina Reader.
    """
    serper = serper_search(query, max_results=max_results, timeout_s=timeout_s)
    if serper:
        return serper
    ddg = ddg_search(query, max_results=max_results, timeout_s=timeout_s)
    if ddg:
        return ddg
    # Last resort: Jina Reader search (often works when HTML SERPs are blocked/throttled).
    return jina_reader_search(query, max_results=max_results, timeout_s=timeout_s)


def _org_slug(organization: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", (organization or "").strip()).lower()


def _host_from_url(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower().lstrip("www.")
    except Exception:
        return ""


def _org_domain_score(org: str, host: str, text: str) -> float:
    """Heuristic score for whether `host` is the org's canonical website."""
    if not host:
        return -999.0
    slug = _org_slug(org)
    ol = (org or "").lower()
    tl = (text or "").lower()
    hl = host.lower()

    score = 0.0

    # Penalize obvious non-official hosts
    bad = (
        "linkedin.com",
        "twitter.com",
        "x.com",
        "facebook.com",
        "instagram.com",
        "youtube.com",
        "wikipedia.org",
        "crunchbase.com",
        "bloomberg.com",
        "pitchbook.com",
        "zoominfo.com",
        "apollo.io",
        "duckduckgo.com",
        "bing.com",
    )
    if any(b in hl for b in bad):
        score -= 6.0

    # Reward if org tokens appear in page text
    tokens = [t for t in re.split(r"\s+", ol) if len(t) >= 4]
    for t in tokens[:3]:
        if t in tl:
            score += 1.2
    if slug and slug in tl.replace(" ", ""):
        score += 2.0

    # Reward common corporate site paths in fetched URL (handled elsewhere), but host hints:
    if any(h in hl for h in (".vc", ".com", ".io", ".co")):
        score += 0.2

    # Reward if homepage-ish content mentions team/invest/portfolio signals
    for k in ("team", "portfolio", "invest", "fund", "about", "careers", "contact"):
        if k in tl:
            score += 0.35

    return score


def discover_org_domain(
    organization: str,
    *,
    timeout_s: float,
    max_chars: int,
) -> tuple[Optional[str], list[dict[str, str]]]:
    """
    Try to discover the org's canonical domain for site-restricted retrieval.
    All URL fetching is parallelised so the total wall time is bounded by a single
    fetch timeout rather than N × timeout.

    Returns: (best_host_or_none, extra_hit_records)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    org = (organization or "").strip()
    if not org:
        return None, []

    # Use a shorter timeout for discovery — we just need a landing page, not full content.
    disc_timeout = min(10.0, timeout_s)
    disc_max_chars = min(6000, max_chars)

    extra: list[dict[str, str]] = []
    candidates: list[tuple[str, float, str]] = []  # (host, score, evidence_url)

    # ── 1) Collect candidate URLs (guesses + search) ──────────────────────────
    guess_urls = guess_org_homepage_urls(org, max_guesses=5)

    queries = [
        f"{org} official website",
        f"{org} venture fund website",
    ]
    search_hits: list[dict[str, str]] = []
    for q in queries:
        try:
            for h in web_search(q, max_results=5, timeout_s=disc_timeout):
                u = (h.get("url") or "").strip()
                if u:
                    extra.append({"title": h.get("title", ""), "url": u, "query": f"org-discovery:{q}"})
                    search_hits.append(h)
        except Exception:
            pass

    all_urls_to_fetch = list(dict.fromkeys(guess_urls + [h.get("url", "") for h in search_hits if h.get("url")]))

    # ── 2) Fetch all candidates in parallel ───────────────────────────────────
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_url, u, timeout_s=disc_timeout, max_chars=disc_max_chars): u for u in all_urls_to_fetch}
        for fut in _as_completed(futs):
            u = futs[fut]
            try:
                rec = fut.result()
            except Exception:
                continue
            if not is_usable_retrieval(rec, min_chars=220):
                continue
            host = _host_from_url(rec.get("final_url") or rec.get("url") or u)
            sc = _org_domain_score(org, host, rec.get("text_excerpt") or "")
            if host:
                candidates.append((host, sc, u))

    if not candidates:
        return None, extra

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_host, best_score, _ = candidates[0]
    if best_score < 0.5:
        return None, extra
    return best_host, extra


def site_probe_urls(host: str) -> list[str]:
    """Probe common paths on a discovered host."""
    h = (host or "").strip().lower().lstrip("www.")
    if not h:
        return []
    base = f"https://{h}"
    paths = ["/", "/team", "/people", "/about", "/company", "/leadership", "/contact", "/news", "/blog", "/press"]
    return [base + p for p in paths]


def guess_org_homepage_urls(organization: str, *, max_guesses: int = 8) -> list[str]:
    """
    Best-effort guesses for org site URLs when search is flaky.
    Example: 'Mercury' -> mercury.com, mercury.io, etc.
    """
    raw = (organization or "").strip()
    if not raw:
        return []

    slug = re.sub(r"[^a-zA-Z0-9]+", "", raw).lower()
    if not slug:
        return []

    ol = raw.lower()
    if any(k in ol for k in ("capital", "ventures", "venture", "vc", "fund", "partners", "holdings")):
        # VC-style names often use .vc domains
        tlds = ("vc", "com", "io", "co", "ai", "app", "net", "org")
    else:
        tlds = ("com", "io", "co", "vc", "ai", "app", "net", "org")
    guesses: list[str] = []
    for t in tlds:
        guesses.append(f"https://{slug}.{t}/")
        if len(guesses) >= max_guesses:
            break
    return guesses


def unique_urls(items: list[dict[str, str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        u = it.get("url") or ""
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out
