import unittest


class TestAgent1AffiliationEvidence(unittest.TestCase):
    def test_affiliation_evidence_detected(self):
        from agent1_researcher import _has_affiliation_evidence

        packet = [
            {"text_excerpt": "Sarah Guo is the Founder and GP at Conviction."},
            {"text_excerpt": "Conviction is an early stage venture firm."},
        ]
        self.assertTrue(_has_affiliation_evidence(packet, "Sarah Guo", "Conviction"))

    def test_affiliation_evidence_not_detected_when_no_overlap(self):
        from agent1_researcher import _has_affiliation_evidence

        packet = [
            {"text_excerpt": "Cognition builds AI tooling for software development."},
            {"text_excerpt": "Sarah Guo wrote about AI agents."},
        ]
        self.assertFalse(_has_affiliation_evidence(packet, "Sarah Guo", "Cognition"))


class TestAgent1OpeningHookGuardrail(unittest.TestCase):
    def test_missing_specific_hook_returns_false(self):
        from agent1_researcher import _has_specific_opening_hook

        result = {
            "draft_email": "Hi Sarah,\n\nI enjoyed learning about your work.\n\nBest,\nAlex",
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": "I enjoyed learning about your work.",
                    "source_url": "https://example.com/",
                    "source_type": "org_website",
                    "confidence": 1.0,
                }
            ],
        }
        self.assertFalse(_has_specific_opening_hook(result))

    def test_specific_hook_in_opening_returns_true(self):
        from agent1_researcher import _has_specific_opening_hook

        result = {
            "draft_email": "Hi Sarah,\n\nYour recent podcast episode about building AI-native teams stood out.\n\nBest,\nAlex",
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": "Your recent podcast episode about building AI-native teams stood out.",
                    "source_url": "https://example.com/podcast",
                    "source_type": "podcast",
                    "confidence": 0.9,
                }
            ],
        }
        self.assertTrue(_has_specific_opening_hook(result))


class TestAgent1HitRanking(unittest.TestCase):
    def test_high_signal_press_outranks_generic_site_probe(self):
        from agent1_researcher import _score_hit_record

        press_hit = {
            "title": "CNBC: Brendan Foody on Mercor's growth",
            "url": "https://www.cnbc.com/2026/01/01/mercor-brendan-foody.html",
            "query": "third-party:Brendan Foody interview 2026",
        }
        generic_hit = {
            "title": "site-probe",
            "url": "https://www.mercor.com/team",
            "query": "site-probe:www.mercor.com",
        }

        self.assertGreater(
            _score_hit_record(press_hit, "Brendan Foody", "Mercor", "www.mercor.com"),
            _score_hit_record(generic_hit, "Brendan Foody", "Mercor", "www.mercor.com"),
        )


class TestAgent1EmailResolution(unittest.TestCase):
    def test_extracts_explicit_email_from_packet(self):
        from agent1_researcher import _extract_email_evidence

        packet = [
            {
                "url": "https://example.com/team",
                "final_url": "https://example.com/team",
                "text_excerpt": "Brendan Foody Founder Mercor Contact Brendan at brendan@mercor.com for press requests.",
            }
        ]
        email, src = _extract_email_evidence(packet, "Brendan Foody", "Mercor", "www.mercor.com")
        self.assertEqual(email, "brendan@mercor.com")
        self.assertEqual(src, "https://example.com/team")

    def test_inferrs_pattern_with_same_domain_corroboration(self):
        from agent1_researcher import _extract_email_evidence

        packet = [
            {
                "url": "https://example.com/contact",
                "final_url": "https://example.com/contact",
                "text_excerpt": "Leadership team: jane.doe@mercor.com and sam.lee@mercor.com. Mercor builds frontier AI recruiting tools.",
            }
        ]
        email, src = _extract_email_evidence(packet, "Brendan Foody", "Mercor", "mercor.com")
        self.assertEqual(email, "brendan.foody@mercor.com")
        self.assertEqual(src, "https://example.com/contact")


class TestAgent1SupportedHook(unittest.TestCase):
    def test_supported_opening_hook_requires_source_overlap(self):
        from agent1_researcher import _has_specific_opening_hook_supported

        result = {
            "draft_email": "Hi Sarah,\n\nYour recent piece on Software Abundance stood out.\n\nBest,\nAlex",
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": "Your recent piece on Software Abundance stood out.",
                    "source_url": "https://example.com/article",
                    "source_type": "news_article",
                    "confidence": 0.9,
                }
            ],
        }
        packet = [{"url": "https://example.com/article", "final_url": "https://example.com/article", "text_excerpt": "Software Abundance explores how AI changes software economics."}]
        self.assertTrue(_has_specific_opening_hook_supported(result, packet))

    def test_unsupported_opening_hook_fails_when_source_is_thin(self):
        from agent1_researcher import _has_specific_opening_hook_supported

        result = {
            "draft_email": "Hi Sarah,\n\nYour recent piece on Software Abundance stood out.\n\nBest,\nAlex",
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": "Your recent piece on Software Abundance argued that judgment and coordination matter most.",
                    "source_url": "https://example.com/article",
                    "source_type": "news_article",
                    "confidence": 0.9,
                }
            ],
        }
        packet = [{"url": "https://example.com/article", "final_url": "https://example.com/article", "text_excerpt": "Recent article: Software Abundance."}]
        self.assertFalse(_has_specific_opening_hook_supported(result, packet))


if __name__ == "__main__":
    unittest.main()

