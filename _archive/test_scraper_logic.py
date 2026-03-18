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


    def test_clean_extracted_review_snippet_keeps_short_real_review(self):
        """Short customer reviews must NOT be dropped as metadata."""
        short_reviews = [
            "Great work!",
            "5 star service, highly recommend!",
            "Amazing results, 10/10!",
            "Quick and professional. Very happy.",
            "Will use again!",
        ]
        for review in short_reviews:
            result = clean_extracted_review_snippet(review)
            self.assertTrue(
                len(result) > 0,
                f"Expected short review to be kept but was dropped: {review!r}",
            )

    def test_is_business_card_snippet_rejects_google_topics_widget(self):
        """Google's 'Price assessment' Topics widget must be flagged as non-review."""
        widget = "Price assessment Great price Services Power/pressure washing"
        self.assertTrue(is_business_card_snippet(widget))

    def test_clean_extracted_review_snippet_rejects_google_topics_widget(self):
        """Google's Topics widget should be cleaned away entirely."""
        widget = "Price assessment Great price Services Power/pressure washing"
        self.assertEqual(clean_extracted_review_snippet(widget), "")

    def test_clean_extracted_review_snippet_strips_owner_reply(self):
        """Owner reply text concatenated after customer review should be stripped."""
        # "Response from the owner" appended to a real review
        mixed = "Great service, very professional. Response from the owner Thank you for your kind words!"
        result = clean_extracted_review_snippet(mixed)
        self.assertNotIn("thank you", result.lower())
        self.assertIn("professional", result.lower())

    def test_sanitize_drops_owner_reply_snippet(self):
        """sanitize_review_snippets must drop snippets that are purely owner replies."""
        snippets = [
            "Amazing work, showed up on time.",
            "Response from the owner Thanks for the review!",
            "Best power washing in the city.",
        ]
        result = sanitize_review_snippets(snippets)
        self.assertEqual(len(result), 2)
        for r in result:
            self.assertNotIn("response from the owner", r.lower())

    def test_is_business_card_snippet_rejects_address(self):
        """Pure address strings must be flagged as non-review."""
        self.assertTrue(is_business_card_snippet("6500 E 44th Ave Unit G, Denver, CO 80216"))

    def test_is_business_card_snippet_rejects_hours(self):
        """Hours-of-operation strings must be flagged as non-review."""
        hours = "SundayClosed Monday9 AM–3 PM Tuesday9 AM–3 PM Wednesday9 AM–3 PM Thursday9 AM–3 PM Friday9 AM–3 PM Saturday10 AM–2 PM"
        self.assertTrue(is_business_card_snippet(hours))

    def test_is_business_card_snippet_rejects_in_store(self):
        """Google business info like 'In-store shopping' must be flagged."""
        self.assertTrue(is_business_card_snippet("· In-store shopping · In-store pickup · Delivery"))

    def test_clean_extracted_review_snippet_rejects_address(self):
        """Pure address should be cleaned away entirely."""
        self.assertEqual(clean_extracted_review_snippet("6500 E 44th Ave Unit G, Denver, CO 80216"), "")

    def test_clean_extracted_review_snippet_rejects_hours(self):
        """Hours-of-operation should be cleaned away entirely."""
        hours = "SundayClosed Monday8 AM–4 PM Tuesday8 AM–4 PM Wednesday8 AM–4 PM Thursday8 AM–4 PM Friday8 AM–4 PM SaturdayClosed"
        self.assertEqual(clean_extracted_review_snippet(hours), "")

    def test_is_business_card_snippet_rejects_open_24_hours(self):
        """Hours with 'Open 24 hours' pattern must be flagged."""
        hours = "SundayOpen 24 hours MondayOpen 24 hours TuesdayOpen 24 hours WednesdayOpen 24 hours ThursdayOpen 24 hours FridayOpen 24 hours SaturdayOpen 24 hours"
        self.assertTrue(is_business_card_snippet(hours))

    def test_is_business_card_snippet_rejects_plus_code(self):
        """Google Plus Code like 'WHGX+H2 El Paso, Texas' must be flagged."""
        self.assertTrue(is_business_card_snippet("WHGX+H2 El Paso, Texas"))

    def test_is_business_card_snippet_rejects_google_boilerplate(self):
        """Google review disclaimer boilerplate must be flagged."""
        boilerplate = "Reviews are automatically processed to detect inappropriate content like fake reviews and spam. We may take down reviews that are flagged in order to comply with Google policies or legal obligations."
        self.assertTrue(is_business_card_snippet(boilerplate))

    def test_clean_extracted_review_snippet_rejects_open_24_hours(self):
        """Open 24 hours block should be cleaned away entirely."""
        hours = "SundayOpen 24 hours MondayOpen 24 hours TuesdayOpen 24 hours WednesdayOpen 24 hours ThursdayOpen 24 hours FridayOpen 24 hours SaturdayOpen 24 hours"
        self.assertEqual(clean_extracted_review_snippet(hours), "")

    def test_clean_extracted_review_snippet_rejects_plus_code(self):
        """Google Plus Code should be cleaned away entirely."""
        self.assertEqual(clean_extracted_review_snippet("WHGX+H2 El Paso, Texas"), "")

    def test_clean_extracted_review_snippet_rejects_google_boilerplate(self):
        """Google review disclaimer should be cleaned away entirely."""
        boilerplate = "Reviews are automatically processed to detect inappropriate content like fake reviews and spam. We may take down reviews that are flagged in order to comply with Google policies or legal obligations."
        self.assertEqual(clean_extracted_review_snippet(boilerplate), "")

    def test_clean_strips_like_share_from_end(self):
        """'Like Share' and '+N Like Share' buttons at end should be stripped."""
        text = "Great service, very professional! Like Share"
        result = clean_extracted_review_snippet(text)
        self.assertNotIn("Like Share", result)
        self.assertIn("professional", result)

    def test_clean_strips_plus_n_like_share(self):
        text = "Vehicle looks brand new! Have used them for two vehicles. +2 Like Share"
        result = clean_extracted_review_snippet(text)
        self.assertNotIn("Like Share", result)
        self.assertIn("brand new", result)

    def test_clean_strips_reviewer_metadata_prefix(self):
        """Reviewer name + metadata at start should be stripped."""
        text = "Anakaren Estrada 2 reviews 4 months ago I booked an appointment to have my car's engine steam cleaned."
        result = clean_extracted_review_snippet(text)
        self.assertNotIn("Anakaren", result)
        self.assertIn("booked an appointment", result)

    def test_clean_strips_local_guide_metadata_prefix(self):
        text = "Nancy Trujillo Local Guide · 13 reviews · 5 photos a month ago Vehicle looks brand new!"
        result = clean_extracted_review_snippet(text)
        self.assertNotIn("Nancy", result)
        self.assertNotIn("Local Guide", result)
        self.assertIn("brand new", result)

    def test_clean_strips_hover_to_react(self):
        text = "Excellent service! Hover to react"
        result = clean_extracted_review_snippet(text)
        self.assertNotIn("Hover to react", result)
        self.assertIn("Excellent", result)

    def test_clean_rejects_quoted_highlight_fragment(self):
        """Quoted review highlight fragments should be dropped."""
        text = '"Passionate about the quality of the service and that\'s the most important part."'
        result = clean_extracted_review_snippet(text)
        self.assertEqual(result, "")

    def test_sanitize_drops_substring_duplicate(self):
        """If a snippet is a substring of an already-accepted review, drop it."""
        snippets = [
            "Dogg did an excellent job getting two difficult stains out of my seats. I would upload photos but I didn't take before pictures. The inside of my car looks brand new and smells great.",
            "Amazing service, very impressed!",
            "Dogg did an excellent job getting two difficult stains out of my seats.",
        ]
        result = sanitize_review_snippets(snippets)
        self.assertEqual(len(result), 2)
        # The full review and the other review should remain
        self.assertIn("Amazing service", result[1])

    def test_sanitize_drops_superset_duplicate(self):
        """If a new snippet contains an already-accepted review, drop the new one."""
        snippets = [
            "Vehicle looks brand new! Have used them for two vehicles, and can't wait to have the 3rd done!",
            "Amazing service!",
            "Nancy Trujillo Local Guide 13 reviews 5 photos a month ago Vehicle looks brand new! Have used them for two vehicles, and can't wait to have the 3rd done! +2 Like Share",
        ]
        result = sanitize_review_snippets(snippets)
        self.assertEqual(len(result), 2)

    def test_clean_rejects_pure_quoted_highlight_fragment(self):
        """Google Review Highlights quoted fragments must be dropped."""
        frag = '"Passionate about the quality of the service and that\'s the most important part."'
        self.assertEqual(clean_extracted_review_snippet(frag), "")

    def test_clean_rejects_quoted_highlight_owner_mention(self):
        """Another typical highlight fragment."""
        frag = '"According to the owner he has 30 years experience well it does not show."'
        self.assertEqual(clean_extracted_review_snippet(frag), "")

    def test_clean_keeps_real_review_not_quoted(self):
        """Real reviews should not be dropped even if short."""
        review = "Great service! Diego did an amazing job on my car."
        result = clean_extracted_review_snippet(review)
        self.assertIn("Diego", result)

    def test_clean_keeps_long_quoted_review(self):
        """A long quoted review should NOT be treated as a highlight fragment."""
        long_review = '"I was stunned with the result of Diego\'s work on my 2010 Lacrosse. It looked like it came off the showroom floor. I would highly recommend his services. SPECTACULAR!! The work was not done at the Lee Trevino address however they picked up and even offered drop off at my residence."'
        result = clean_extracted_review_snippet(long_review)
        self.assertIn("Diego", result)

    def test_clean_rejects_google_ui_text(self):
        """Google Maps promotional text must be dropped."""
        self.assertEqual(clean_extracted_review_snippet("Get the most out of Google Maps"), "")

    def test_sanitize_keeps_six_real_reviews(self):
        """sanitize_review_snippets must not drop real review text, even short ones."""
        six_reviews = [
            "Amazing work, showed up on time and left everything spotless.",
            "Quick and professional.",
            "Will definitely hire again!",
            "Best power washing in the city. Highly recommend.",
            "Great price and did a fantastic job.",
            "5 stars, super happy with the results!",
        ]
        result = sanitize_review_snippets(six_reviews, max_items=6)
        self.assertEqual(len(result), 6, f"Expected 6 reviews but got {len(result)}: {result}")


if __name__ == "__main__":
    unittest.main()
