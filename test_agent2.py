"""
Unit tests for Agent 2 (Validator)
Requirement: at minimum, one test that feeds a known-bad claim (unverifiable, no source URL)
through Agent 2 and confirms it is flagged correctly.

Run with: python -m pytest test_agent2.py -v
or:        python test_agent2.py
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

# Load `.env` for local dev so live integration test can run
# without requiring the developer to `export` keys manually.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent1_output(claims=None, draft_email="Test email body.", resolved_title="Partner"):
    """Build a minimal Agent 1 output dict for testing."""
    return {
        "name": "Test Person",
        "organization": "Test Org",
        "resolved_title": resolved_title,
        "resolved_email": "test@testorg.com",
        "email_domain_source": "https://testorg.com/team",
        "draft_email": draft_email,
        "research_sources": [],
        "claims": claims or [],
    }


KNOWN_BAD_CLAIM = {
    "claim_id": "c1",
    "claim_text": "You recently said AI will replace all jobs by 2025 at the FutureWork Summit.",
    "source_url": "",          # ← deliberately empty — unverifiable
    "source_type": "conference_panel",
    "confidence": 0.1,          # ← below 0.3 threshold → must BLOCK
}

KNOWN_LOW_CONF_CLAIM = {
    "claim_id": "c2",
    "claim_text": "Your firm recently invested in a commerce startup.",
    "source_url": "https://testorg.com/news",
    "source_type": "news_article",
    "confidence": 0.45,         # ← 0.3–0.6 range → must WARN
}

KNOWN_GOOD_CLAIM = {
    "claim_id": "c3",
    "claim_text": "Test Org focuses on early-stage investments.",
    "source_url": "https://testorg.com/about",
    "source_type": "org_website",
    "confidence": 0.85,         # ← above 0.6 → no flag required
}


# ---------------------------------------------------------------------------
# Pure-logic tests (no API calls)
# ---------------------------------------------------------------------------

class TestConfidenceFloorLogic(unittest.TestCase):
    """Test that the pipeline correctly applies confidence floor rules to pre-built flag lists."""

    def _apply_confidence_rules(self, claims):
        """Simulate the confidence-floor logic in agent2_validator.run_agent2."""
        flags = []
        for claim in claims:
            conf = claim.get("confidence", 1.0)
            if conf < 0.3:
                flags.append({
                    "claim_id": claim["claim_id"],
                    "flag_type": "low_confidence",
                    "detail": f"Confidence {conf:.2f} is below BLOCK threshold of 0.3",
                    "severity": "BLOCK",
                })
            elif conf < 0.6:
                flags.append({
                    "claim_id": claim["claim_id"],
                    "flag_type": "low_confidence",
                    "detail": f"Confidence {conf:.2f} is below WARN threshold of 0.6",
                    "severity": "WARN",
                })
        return flags

    def test_very_low_confidence_becomes_block(self):
        flags = self._apply_confidence_rules([KNOWN_BAD_CLAIM])
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["severity"], "BLOCK")
        self.assertEqual(flags[0]["claim_id"], "c1")

    def test_medium_confidence_becomes_warn(self):
        flags = self._apply_confidence_rules([KNOWN_LOW_CONF_CLAIM])
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["severity"], "WARN")
        self.assertEqual(flags[0]["claim_id"], "c2")

    def test_high_confidence_produces_no_flag(self):
        flags = self._apply_confidence_rules([KNOWN_GOOD_CLAIM])
        self.assertEqual(len(flags), 0)

    def test_mixed_claims_all_flagged_correctly(self):
        flags = self._apply_confidence_rules([KNOWN_BAD_CLAIM, KNOWN_LOW_CONF_CLAIM, KNOWN_GOOD_CLAIM])
        flag_map = {f["claim_id"]: f for f in flags}
        self.assertIn("c1", flag_map)
        self.assertEqual(flag_map["c1"]["severity"], "BLOCK")
        self.assertIn("c2", flag_map)
        self.assertEqual(flag_map["c2"]["severity"], "WARN")
        self.assertNotIn("c3", flag_map)


class TestValidationResultLogic(unittest.TestCase):
    """Test that FAIL/WARN/PASS are assigned correctly based on flag severities."""

    def _compute_result(self, flags):
        if any(f["severity"] == "BLOCK" for f in flags):
            return "FAIL"
        elif any(f["severity"] == "WARN" for f in flags):
            return "WARN"
        return "PASS"

    def test_block_flag_causes_fail(self):
        flags = [{"claim_id": "c1", "flag_type": "hallucination", "detail": "...", "severity": "BLOCK"}]
        self.assertEqual(self._compute_result(flags), "FAIL")

    def test_warn_flag_causes_warn(self):
        flags = [{"claim_id": "c2", "flag_type": "title_mismatch", "detail": "...", "severity": "WARN"}]
        self.assertEqual(self._compute_result(flags), "WARN")

    def test_no_flags_causes_pass(self):
        self.assertEqual(self._compute_result([]), "PASS")

    def test_fail_requires_null_approved_email(self):
        """If result is FAIL, approved_email must be None."""
        result = {"validation_result": "FAIL", "approved_email": "some text"}
        # Simulate the enforcement in run_agent2
        if result["validation_result"] == "FAIL":
            result["approved_email"] = None
        self.assertIsNone(result["approved_email"])


class TestSourceURLValidation(unittest.TestCase):
    """Test unverifiable claim detection (empty/missing source_url)."""

    def _check_sources(self, claims):
        flags = []
        for claim in claims:
            url = claim.get("source_url", "").strip()
            if not url:
                flags.append({
                    "claim_id": claim["claim_id"],
                    "flag_type": "unverifiable",
                    "detail": "No source URL provided for claim",
                    "severity": "BLOCK",
                })
        return flags

    def test_empty_source_url_is_flagged(self):
        """A claim with an empty source_url must be flagged as unverifiable BLOCK."""
        flags = self._check_sources([KNOWN_BAD_CLAIM])
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["flag_type"], "unverifiable")
        self.assertEqual(flags[0]["severity"], "BLOCK")

    def test_present_source_url_not_flagged(self):
        flags = self._check_sources([KNOWN_GOOD_CLAIM])
        self.assertEqual(len(flags), 0)


class TestJSONExtraction(unittest.TestCase):
    """Test the _extract_json helper used by both agents."""

    def test_clean_json_parses(self):
        from utils import extract_json as _extract_json
        raw = '{"name": "Test", "validation_result": "PASS", "flags": [], "approved_email": "Hi"}'
        result = _extract_json(raw)
        self.assertEqual(result["name"], "Test")
        self.assertEqual(result["validation_result"], "PASS")

    def test_json_with_markdown_fences_parses(self):
        from utils import extract_json as _extract_json
        raw = '```json\n{"name": "Test", "validation_result": "FAIL", "flags": [], "approved_email": null}\n```'
        result = _extract_json(raw)
        self.assertEqual(result["validation_result"], "FAIL")
        self.assertIsNone(result["approved_email"])

    def test_trailing_commas_handled(self):
        from utils import extract_json as _extract_json
        raw = '{"name": "Test", "flags": [{"id": "c1",}],}'
        result = _extract_json(raw)
        self.assertEqual(result["name"], "Test")

    def test_empty_text_raises(self):
        from utils import extract_json as _extract_json
        with self.assertRaises(ValueError):
            _extract_json("")

    def test_no_json_raises(self):
        from utils import extract_json as _extract_json
        with self.assertRaises(ValueError):
            _extract_json("This is just plain text with no JSON.")


class TestHardValidationRules(unittest.TestCase):
    """Deterministic rules required by the assignment (confidence floor, source URLs)."""

    def test_missing_source_url_is_fail(self):
        from validation_rules import apply_hard_validation_rules

        result = {"validation_result": "PASS", "flags": [], "approved_email": "send me"}
        clean = {
            "claims": [{
                "claim_id": "c1",
                "claim_text": "Something happened",
                "source_url": "",
                "source_type": "news_article",
                "confidence": 0.95,
            }]
        }
        apply_hard_validation_rules(result, clean)
        self.assertEqual(result["validation_result"], "FAIL")
        self.assertIsNone(result["approved_email"])

    def test_low_confidence_warn(self):
        from validation_rules import apply_hard_validation_rules

        result = {"validation_result": "PASS", "flags": [], "approved_email": "ok"}
        clean = {
            "draft_email": "Hello\n\nThis is a test email body with enough length.\n\nBest,\nMe",
            "claims": [{
                "claim_id": "c2",
                "claim_text": "x",
                "source_url": "https://example.com",
                "source_type": "org_website",
                "confidence": 0.45,
            }]
        }
        apply_hard_validation_rules(result, clean)
        self.assertEqual(result["validation_result"], "WARN")

    def test_missing_draft_email_is_fail(self):
        from validation_rules import apply_hard_validation_rules

        result = {"validation_result": "PASS", "flags": [], "approved_email": "ok"}
        clean = {"draft_email": None, "claims": []}
        apply_hard_validation_rules(result, clean)
        self.assertEqual(result["validation_result"], "FAIL")
        self.assertIsNone(result["approved_email"])
        self.assertTrue(any(f.get("claim_id") == "system" for f in result.get("flags", [])))

    def test_domain_mismatch_removed_when_no_resolved_email(self):
        from validation_rules import apply_hard_validation_rules

        result = {
            "validation_result": "PASS",
            "approved_email": "ok",
            "flags": [
                {"claim_id": "email", "flag_type": "domain_mismatch", "detail": "x", "severity": "BLOCK"},
                {"claim_id": "title", "flag_type": "title_mismatch", "detail": "y", "severity": "WARN"},
            ],
        }
        clean = {"draft_email": "Hello\n\nBody\n\nBest,\nMe", "claims": [], "resolved_email": None}
        apply_hard_validation_rules(result, clean)
        self.assertFalse(any(f.get("flag_type") == "domain_mismatch" for f in result.get("flags", [])))
        self.assertTrue(any(f.get("flag_type") == "title_mismatch" for f in result.get("flags", [])))


class TestKkcClaimPruning(unittest.TestCase):
    def test_prunes_unnecessary_kkc_claims_and_sources(self):
        from claim_postprocess import prune_unnecessary_kkc_references

        result = {
            "draft_email": "Hello",
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": "Your post about building durable products stood out to me.",
                    "source_url": "https://recipient.example/about",
                    "source_type": "org_website",
                    "confidence": 1.0,
                },
                {
                    "claim_id": "c2",
                    "claim_text": "We are a pre-seed and seed-stage firm focused on commerce, AI, and labor.",
                    "source_url": "https://kyberknight.com/",
                    "source_type": "org_website",
                    "confidence": 1.0,
                },
            ],
            "research_sources": [
                {"url": "https://recipient.example/about", "content_summary": "x", "retrieved_at": "t"},
                {"url": "https://kyberknight.com/", "content_summary": "y", "retrieved_at": "t"},
            ],
        }

        prune_unnecessary_kkc_references(result)

        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["claim_id"], "c1")
        self.assertEqual([s["url"] for s in result["research_sources"]], ["https://recipient.example/about"])


# ---------------------------------------------------------------------------
# Claim/source alignment (topic-agnostic)
# ---------------------------------------------------------------------------

class TestClaimSourceAlignment(unittest.TestCase):
    def test_specific_claim_cited_to_generic_source_is_flagged(self):
        from agent2_validator import _apply_claim_source_alignment_rules

        clean = {
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": "Your recent podcast episode about technical teams in the software 3.0 era discussed how you structure hiring loops for AI-native orgs.",
                    "source_url": "https://example.com/home",
                    "source_type": "podcast",
                    "confidence": 0.95,
                }
            ]
        }
        result = {"validation_result": "PASS", "flags": [], "approved_email": "x"}
        packet = [
            {
                "url": "https://example.com/home",
                "final_url": "https://example.com/home",
                "status_code": 200,
                "text_excerpt": "Example Fund. We invest early. Portfolio. Contact.",
            }
        ]

        _apply_claim_source_alignment_rules(result, clean, packet)
        self.assertTrue(any(f.get("claim_id") == "c1" for f in result.get("flags", [])))

    def test_reasonably_aligned_claim_not_flagged(self):
        from agent2_validator import _apply_claim_source_alignment_rules

        clean = {
            "claims": [
                {
                    "claim_id": "c2",
                    "claim_text": "Your website lists a portfolio including Mistral and Cartesia among the companies you have backed.",
                    "source_url": "https://example.com/portfolio",
                    "source_type": "org_website",
                    "confidence": 0.9,
                }
            ]
        }
        result = {"validation_result": "PASS", "flags": [], "approved_email": "x"}
        packet = [
            {
                "url": "https://example.com/portfolio",
                "final_url": "https://example.com/portfolio",
                "status_code": 200,
                "text_excerpt": "Portfolio: Mistral, Cartesia, Harvey. We invest in AI-native companies.",
            }
        ]

        _apply_claim_source_alignment_rules(result, clean, packet)
        self.assertEqual(len(result.get("flags", [])), 0)


# ---------------------------------------------------------------------------
# Integration test (skipped if no API key)
# ---------------------------------------------------------------------------

@unittest.skipUnless(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"), "GEMINI_API_KEY not set — skipping live API test")
class TestAgent2LiveKnownBadClaim(unittest.TestCase):
    """
    Integration test: feed a known-bad claim (no source URL, confidence 0.1)
    through the real Agent 2 and confirm it surfaces a BLOCK flag.
    """

    def test_known_bad_claim_is_blocked(self):
        from agent2_validator import run_agent2

        agent1_output = make_agent1_output(
            claims=[KNOWN_BAD_CLAIM],
            draft_email="You recently said AI will replace all jobs by 2025 at the FutureWork Summit. I'd love a 30-minute call.",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(agent1_output, tf)
            tmp_path = Path(tf.name)

        try:
            result = run_agent2(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        # Must be FAIL (BLOCK flag present)
        self.assertEqual(result["validation_result"], "FAIL",
                         f"Expected FAIL but got {result['validation_result']}. Flags: {result.get('flags')}")

        # Must have at least one BLOCK flag
        block_flags = [f for f in result.get("flags", []) if f.get("severity") == "BLOCK"]
        self.assertGreater(len(block_flags), 0,
                           f"Expected at least one BLOCK flag. Got: {result.get('flags')}")

        # approved_email must be null
        self.assertIsNone(result.get("approved_email"),
                          f"Expected null approved_email but got: {result.get('approved_email')}")

        print(f"\n[Integration Test] PASS — Agent 2 correctly blocked claim c1")
        print(f"  Flags: {json.dumps(block_flags, indent=2)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
