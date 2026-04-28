"""
Digest Generator
Takes Agent 1 + Agent 2 outputs and produces a human-readable markdown review digest.
"""

from datetime import datetime, timezone
from typing import Optional, Any


SEVERITY_EMOJI = {
    "BLOCK": "🔴",
    "WARN": "🟡",
}

RESULT_EMOJI = {
    "PASS": "✅",
    "WARN": "⚠️",
    "FAIL": "❌",
}

FLAG_LABELS = {
    "title_mismatch": "Title Mismatch",
    "wrong_contact": "Wrong Contact",
    "unverifiable": "Unverifiable Claim",
    "domain_mismatch": "Email Domain Mismatch",
    "hallucination": "Hallucinated Fact",
    "low_confidence": "Low Confidence",
}


def generate_digest(agent1_outputs: list, agent2_outputs: list, output_path: str = "output/review_digest.md"):
    """Generate the final markdown digest from all agent outputs."""

    lines = []
    lines.append("# KKC Outreach Pipeline — Review Digest")
    lines.append(f"\n_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")
    lines.append("---\n")

    # Build lookup by name
    a2_by_name = {r.get("name", ""): r for r in agent2_outputs}

    def _compact_access_summary(research_sources: list[dict[str, Any]]) -> Optional[str]:
        """
        If sources are mostly inaccessible (403/401/429) or empty, return a short summary string.
        This avoids spamming the digest with repetitive access errors.
        """
        if not research_sources:
            return None

        total = 0
        blocked = 0
        empty = 0
        for s in research_sources:
            total += 1
            summ = (s.get("content_summary") or "").lower()
            if "403" in summ or "forbidden" in summ or "401" in summ or "429" in summ or "rate limit" in summ:
                blocked += 1
            if "no content" in summ or "no text" in summ or "empty" in summ:
                empty += 1

        if total >= 3 and (blocked / total) >= 0.7:
            return f"Most sources were not accessible (access blocked on {blocked}/{total} URLs)."
        if total >= 3 and (empty / total) >= 0.7:
            return f"Most sources returned little/no usable text ({empty}/{total} URLs)."
        return None

    for a1 in agent1_outputs:
        name = a1.get("name", "Unknown")
        org = a1.get("organization", "Unknown")
        a2 = a2_by_name.get(name, {})

        result = a2.get("validation_result", "FAIL")
        emoji = RESULT_EMOJI.get(result, "❓")

        lines.append(f"## {emoji} {name} — {org}")
        lines.append("")

        # Basic info table
        resolved_title = a1.get("resolved_title") or "_not resolved_"
        resolved_email = a1.get("resolved_email") or "_not found_"
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| Resolved Title | {resolved_title} |")
        lines.append(f"| Resolved Email | {resolved_email} |")
        lines.append(f"| Validation Result | **{result}** |")
        lines.append("")

        # Approved email or rejection
        approved_email = a2.get("approved_email")
        if approved_email:
            lines.append("### ✉️ Approved Email — Ready to Send")
            lines.append("")
            lines.append("```")
            lines.append(approved_email.strip())
            lines.append("```")
        else:
            lines.append("### ❌ Email Not Approved")
            lines.append("")
            notes = a2.get("validation_notes", "No details available.")
            lines.append(f"> {notes}")
            if a1.get("draft_email"):
                lines.append("")
                lines.append("Draft email (blocked — do not send):")
                lines.append("")
                lines.append("```")
                lines.append(a1.get("draft_email", "").strip())
                lines.append("```")
        lines.append("")

        # Flags
        flags = a2.get("flags", [])
        if flags:
            lines.append("### 🚩 Flags Raised")
            lines.append("")
            for flag in flags:
                sev = flag.get("severity", "WARN")
                sev_em = SEVERITY_EMOJI.get(sev, "🟡")
                ftype = FLAG_LABELS.get(flag.get("flag_type", ""), flag.get("flag_type", ""))
                detail = flag.get("detail", "")
                cid = flag.get("claim_id", "")
                lines.append(f"- {sev_em} **{ftype}** (claim `{cid}`): {detail}")
            lines.append("")

        # Validation summary
        val_notes = a2.get("validation_notes", "")
        if val_notes:
            lines.append("### 🔍 Validation Summary")
            lines.append("")
            lines.append(val_notes)
            lines.append("")

        # Claims with sources
        claims = a1.get("claims", [])
        sources = a1.get("research_sources", [])
        if claims:
            lines.append("### 📎 Verified Claims & Sources")
            lines.append("")
            # Build a quick set of flagged claim IDs
            blocked_ids = {f.get("claim_id") for f in flags if f.get("severity") == "BLOCK"}
            warned_ids = {f.get("claim_id") for f in flags if f.get("severity") == "WARN"}

            for idx, claim in enumerate(claims, start=1):
                cid = (claim.get("claim_id") or "").strip() or f"c{idx}"
                if cid in blocked_ids:
                    status = "🔴 BLOCKED"
                elif cid in warned_ids:
                    status = "🟡 WARN"
                else:
                    status = "✅ OK"

                conf = claim.get("confidence", 0.0)
                stype = claim.get("source_type", "")
                src_url = claim.get("source_url", "")
                text = claim.get("claim_text", "")

                lines.append(f"**{cid}** {status} _(confidence: {conf:.2f}, type: {stype})_")
                lines.append(f"> {text}")
                if src_url:
                    lines.append(f"> Source: [{src_url}]({src_url})")
                lines.append("")

        # Research sources
        if sources:
            lines.append("### 🌐 Research Sources")
            lines.append("")
            compact = _compact_access_summary(sources)
            if compact:
                lines.append(f"- {compact}")
                lines.append("")

            # Still show up to a small number of links for debugging/traceability.
            for src in sources[:10]:
                url = src.get("url", "")
                summary = src.get("content_summary", "")
                retrieved = src.get("retrieved_at", "")
                lines.append(f"- [{url}]({url})")
                lines.append(f"  - {summary}")
                if retrieved:
                    lines.append(f"  - Retrieved: {retrieved}")
            if len(sources) > 10:
                lines.append(f"- _(plus {len(sources) - 10} more)_")
            lines.append("")

        lines.append("---\n")

    digest = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(digest)

    print(f"Digest written to {output_path}")
    return digest
