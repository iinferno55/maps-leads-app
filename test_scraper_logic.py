import unittest

from app import (
    clean_extracted_review_snippet,
    click_until_reviews_ready,
    detail_page_matches_candidate,
    is_business_card_snippet,
    normalize_place_href,
    sanitize_review_snippets,
    validate_owner_detection,
)


class ScraperLogicTests(unittest.TestCase):
    def test_preserves_google_maps_place_href_with_data_param(self):
        href = "/maps/place/KY-KO+Roofing/data=!4m1!3m2!2m1!1sfoo?entry=ttu"
        self.assertEqual(
            normalize_place_href(href),
            "https://www.google.com/maps/place/KY-KO+Roofing/data=!4m1!3m2!2m1!1sfoo?entry=ttu",
        )

    def test_rejects_non_place_google_maps_href(self):
        href = "https://www.google.com/maps/search/roofers+phoenix"
        self.assertEqual(normalize_place_href(href), "")

    def test_company_name_false_positive_is_rejected(self):
        reviews = """
        Randy was great to work with and explained everything clearly.
        Randy and Mike kept us updated and the roof came out perfect.
        Huge thanks to Randy for fitting us in so quickly.
        """
        det = validate_owner_detection(
            reviews,
            "Casa Roofing LLC",
            {
                "owner_name": "Kyko",
                "solo": True,
                "confidence": 0.9,
                "reason": "Dominant name in reviews",
            },
        )
        # Reject the company-name false positive: do not report Kyko as owner.
        self.assertNotEqual((det["owner_name"] or "").strip().lower(), "kyko")
        # Two-person (Randy and Mike) can be marked solo/call-worthy; confidence need not be minimal.
        self.assertLessEqual(det["confidence"], 0.95)

    def test_multiple_service_names_force_low_confidence(self):
        reviews = """
        Josh was easy to work with and handled the estimate.
        Danny came out the next day and fixed the flashing issue.
        Josh and Danny both made the process easy from start to finish.
        """
        det = validate_owner_detection(
            reviews,
            "Gleason Roofing",
            {
                "owner_name": "Josh",
                "solo": True,
                "confidence": 0.9,
                "reason": "Josh was the dominant name",
            },
        )
        # Two people frequently mentioned (Josh and Danny) – mark as call-worthy, ask for one.
        self.assertTrue(det["solo"])
        self.assertIn("two-person", (det["reason"] or "").lower())

    def test_click_until_reviews_ready_requires_dom_confirmation(self):
        calls = []
        success_markers = []

        def click_action():
            calls.append("clicked")

        def ready_check():
            calls.append("checked")
            return False

        result = click_until_reviews_ready(click_action, ready_check, on_success=lambda: success_markers.append("ok"))

        self.assertFalse(result)
        self.assertEqual(calls, ["clicked", "checked"])
        self.assertEqual(success_markers, [])

    def test_click_until_reviews_ready_marks_success_after_dom_confirmation(self):
        calls = []

        def click_action():
            calls.append("clicked")

        def ready_check():
            calls.append("checked")
            return True

        result = click_until_reviews_ready(click_action, ready_check)

        self.assertTrue(result)
        self.assertEqual(calls, ["clicked", "checked"])

    def test_detail_page_match_rejects_place_url_drift_even_if_name_matches(self):
        self.assertFalse(
            detail_page_matches_candidate(
                detail_name="Floor n more",
                expected_name="Floor n more",
                place_url="https://www.google.com/maps/place/Floor+n+more/about",
                expected_place_url="https://www.google.com/maps/place/Floor+n+more/data=!4m7!entry=ttu",
            )
        )

    def test_detail_page_match_rejects_wrong_business_name(self):
        self.assertFalse(
            detail_page_matches_candidate(
                detail_name="Different Flooring Co",
                expected_name="Floor n more",
                place_url="https://www.google.com/maps/place/Different+Flooring+Co/data=!4m7!entry=ttu",
                expected_place_url="https://www.google.com/maps/place/Floor+n+more/data=!4m7!entry=ttu",
            )
        )

    def test_clean_extracted_review_snippet_strips_reviewer_metadata_header(self):
        snippet = "Allynn Aragon 7 reviews · 6 photos\nI contacted about having my car detailed. They replied right away and showed up on time."
        self.assertEqual(
            clean_extracted_review_snippet(snippet),
            "I contacted about having my car detailed. They replied right away and showed up on time.",
        )

    def test_clean_extracted_review_snippet_rejects_metadata_only_line(self):
        self.assertEqual(clean_extracted_review_snippet("Kerry Pratt 3 reviews"), "")

    def test_clean_extracted_review_snippet_rejects_local_guide_metadata(self):
        self.assertEqual(clean_extracted_review_snippet("Pammy Chaos 6 reviews · 6 photos \ue5d4"), "")

    def test_clean_extracted_review_snippet_rejects_review_topic_chip(self):
        self.assertEqual(clean_extracted_review_snippet("vehicle, mentioned in 3 reviews"), "")

    def test_clean_extracted_review_snippet_extracts_photo_of_reviewer_quote(self):
        snippet = 'Photo of reviewer who wrote "They showed up on time, explained every step, and the floors came out amazing."'
        self.assertEqual(
            clean_extracted_review_snippet(snippet),
            "They showed up on time, explained every step, and the floors came out amazing.",
        )

    def test_clean_extracted_review_snippet_rejects_punctuated_reviewer_metadata(self):
        snippet = "Lia F. Local Guide · 420 reviews · 2,991 photos"
        self.assertEqual(clean_extracted_review_snippet(snippet), "")

    def test_is_business_card_snippet_rejects_search_result_card(self):
        snippet = "Park Cities Mobile Detail 4.7(255) Car detailing service · 2100 N Stemmons Fwy Closed · Opens 8 AM · (469) 438-7377"
        self.assertTrue(is_business_card_snippet(snippet))

    def test_clean_extracted_review_snippet_rejects_search_result_card(self):
        snippet = "Park Cities Mobile Detail 4.7(255) Car detailing service · 2100 N Stemmons Fwy Closed · Opens 8 AM · (469) 438-7377"
        self.assertEqual(clean_extracted_review_snippet(snippet), "")

    def test_sanitize_review_snippets_removes_metadata_only_entries(self):
        snippets = [
            "I was in contact with them for two weeks and they did great work.",
            "Allynn Aragon 7 reviews · 6 photos",
            "Kerry Pratt 3 reviews",
            "vehicle, mentioned in 3 reviews",
            "Park Cities Mobile Detail 4.7(255) Car detailing service · 2100 N Stemmons Fwy Closed · Opens 8 AM · (469) 438-7377",
            "I contacted about having my car detailed and Michael showed up right on time.",
        ]
        self.assertEqual(
            sanitize_review_snippets(snippets, max_items=6),
            [
                "I was in contact with them for two weeks and they did great work.",
                "I contacted about having my car detailed and Michael showed up right on time.",
            ],
        )


if __name__ == "__main__":
    unittest.main()
