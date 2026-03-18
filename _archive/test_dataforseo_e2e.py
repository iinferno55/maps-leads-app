"""
E2E test for the DataForSEO scraping pipeline.
Mocks maps_search() and fetch_reviews_text() with realistic review data,
then runs the full owner detection pipeline (Gemini extraction + rules)
to verify it works end-to-end without spending any DataForSEO credits.

Usage:  py test_dataforseo_e2e.py
"""
import json, os, sys, time
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

# Set dummy DataForSEO credentials so scrape_dataforseo doesn't bail
os.environ.setdefault("DATAFORSEO_LOGIN", "mock_test")
os.environ.setdefault("DATAFORSEO_PASSWORD", "mock_test")

from dataforseo_client import MapsPlace

# ── Realistic test data (mimics what DataForSEO would return) ────────

WORK_HOURS_5PM = {
    "timetable": {
        day: [{"open": {"hour": 8, "minute": 0}, "close": {"hour": 17, "minute": 0}}]
        for day in ("monday", "tuesday", "wednesday", "thursday", "friday")
    }
}

TEST_BUSINESSES = [
    {
        "place": MapsPlace(
            title="Mike's Locksmith Service",
            address="123 Main St, Seattle, WA 98101",
            phone="(206) 555-0101",
            website="https://mikeslocksmith.com",
            place_id="mock_1", cid="cid_1", url=None,
            reviews_count=35, rating_value=4.8,
            work_hours=WORK_HOURS_5PM,
        ),
        "reviews": [
            "Mike came out within 30 minutes and had my car unlocked in no time. Very professional and fair pricing.",
            "Called Mike at 11pm and he showed up fast. Unlocked my door in minutes. Mike is the best locksmith in Seattle!",
            "Mike is the owner and he does all the work himself. Very trustworthy guy, been using him for years.",
            "Had Mike come out to rekey all our locks after a break-in. He was thorough and explained everything.",
            "Mike responded quickly and was very professional. He made a new key for my car on the spot.",
            "Great service from Mike! He even gave me tips on how to improve my home security.",
            "Mike has been our go-to locksmith for 5 years. Always reliable and honest pricing.",
        ],
        "owner_replies": ["Thank you for the kind words! - Mike", "Glad I could help! - Mike S."],
        "expected_qualified": True,
        "expected_owner": "Mike",
    },
    {
        "place": MapsPlace(
            title="SecureLock Pro Services",
            address="456 Pine Ave, Seattle, WA 98102",
            phone="(206) 555-0202",
            website="https://securelockpro.com",
            place_id="mock_2", cid="cid_2", url=None,
            reviews_count=52, rating_value=4.5,
            work_hours=WORK_HOURS_5PM,
        ),
        "reviews": [
            "They sent a technician named Dave who was very professional. Quick service.",
            "Called and the dispatcher sent someone within an hour. Jason did a great job on our deadbolt.",
            "Their team is very responsive. Alex came out and rekeyed all three locks efficiently.",
            "The office scheduled us for the next day. Tom arrived on time and was very knowledgeable.",
            "Great company with multiple technicians. Ryan handled our commercial lockout expertly.",
            "They have different people every time but always professional. This time it was Steve.",
            "Called their office, receptionist was helpful. Technician Mike S. arrived promptly.",
        ],
        "owner_replies": [],
        "expected_qualified": False,
        "expected_owner": None,
    },
    {
        "place": MapsPlace(
            title="Sarah's Key & Lock",
            address="789 Oak Blvd, Seattle, WA 98103",
            phone="(206) 555-0303",
            website="https://sarahskeyandlock.com",
            place_id="mock_3", cid="cid_3", url=None,
            reviews_count=18, rating_value=4.9,
            work_hours=WORK_HOURS_5PM,
        ),
        "reviews": [
            "Sarah is amazing! She came to our house at 9pm on a Sunday and got us in quickly.",
            "The owner Sarah really knows her stuff. She rekeyed our entire house in under 2 hours.",
            "Sarah and her husband run this business. She handled our car lockout perfectly.",
            "Called Sarah directly, she answered on the first ring. Arrived in 20 minutes. Highly recommend!",
            "Sarah has been doing this for years and it shows. Very fair pricing too.",
        ],
        "owner_replies": ["Thank you so much! - Sarah"],
        "expected_qualified": True,
        "expected_owner": "Sarah",
    },
    {
        "place": MapsPlace(
            title="Seattle Emergency Locksmith",
            address="321 Elm St, Seattle, WA 98104",
            phone="(206) 555-0404",
            website="https://seattleemergencylocksmith.com",
            place_id="mock_4", cid="cid_4", url=None,
            reviews_count=45, rating_value=4.2,
            work_hours=WORK_HOURS_5PM,
        ),
        "reviews": [
            "They were quick to respond to our emergency. The team arrived within 30 minutes.",
            "Professional company. Their crew handled our office lockout without any damage.",
            "Called them for an emergency lockout. The technician was professional but I don't remember his name.",
            "They charged a fair price for a late-night lockout. Would use them again.",
            "Their service is reliable. Different technician each time but always professional.",
            "Great emergency service. The team knows what they're doing.",
        ],
        "owner_replies": [],
        "expected_qualified": False,
        "expected_owner": None,
    },
    {
        "place": MapsPlace(
            title="Dan's Lock & Safe",
            address="555 Cedar Way, Seattle, WA 98105",
            phone="(206) 555-0505",
            website="Not listed",
            place_id="mock_5", cid="cid_5", url=None,
            reviews_count=8, rating_value=5.0,
            work_hours=WORK_HOURS_5PM,
        ),
        "reviews": [
            "Dan is the real deal. He opened my car in 2 minutes flat. Owner-operated and very honest.",
            "Dan came out on a holiday to help us. He's the owner and does everything himself.",
            "Fantastic service from Dan. He even installed a new deadbolt while he was here.",
        ],
        "owner_replies": ["Thanks for the review! - Dan"],
        "expected_qualified": True,
        "expected_owner": "Dan",
    },
]

# ── Build mocks ──────────────────────────────────────────────────────

mock_places = [b["place"] for b in TEST_BUSINESSES]
mock_reviews_map = {
    b["place"].place_id: {
        "reviews": b["reviews"],
        "owner_replies": b["owner_replies"],
    }
    for b in TEST_BUSINESSES
}


def mock_maps_search(**kwargs):
    print(f"  [MOCK] maps_search(keyword='{kwargs.get('keyword', '?')}')")
    return mock_places


def mock_fetch_reviews_text(**kwargs):
    pid = kwargs.get("place_id", "")
    data = mock_reviews_map.get(pid, {"reviews": [], "owner_replies": []})
    review_texts = data["reviews"]
    owner_replies = data["owner_replies"]
    debug_payload = {"_owner_answers": owner_replies}
    print(f"  [MOCK] fetch_reviews_text({pid}): {len(review_texts)} reviews, {len(owner_replies)} replies")
    return review_texts, debug_payload


# ── Run the pipeline ─────────────────────────────────────────────────

print(f"=== E2E Test: DataForSEO pipeline with Gemini owner detection ===\n")
print(f"GOOGLE_API_KEY set: {bool(os.getenv('GOOGLE_API_KEY'))}\n")

from app import scrape_dataforseo, is_qualified_lead_row, MIN_CONFIDENCE


def mock_status(msg):
    print(f"  [STATUS] {msg}")


def mock_progress(pct, msg=None):
    pass


with patch("app.maps_search", side_effect=mock_maps_search), \
     patch("app.fetch_reviews_text", side_effect=mock_fetch_reviews_text):
    start = time.time()
    rows = scrape_dataforseo(
        city="Seattle",
        niche="locksmith",
        max_businesses=10,
        progress_callback=mock_progress,
        status_callback=mock_status,
        dataforseo_location_name="Seattle,Washington,United States",
        reviews_depth=20,
        reviews_priority=2,
    )
    elapsed = time.time() - start


# ── Analyze results ──────────────────────────────────────────────────

def get_reject_reason(row):
    owner = str(row.get("owner_name") or "").strip().lower()
    conf = float(row.get("confidence_score") or 0)
    if not bool(row.get("solo")):
        return "Not marked solo by owner detection"
    if owner in ("", "unknown", "none", "null"):
        return "No valid owner name found"
    if conf < MIN_CONFIDENCE:
        return f"Confidence below threshold ({conf:.2f} < {MIN_CONFIDENCE:.2f})"
    return "Qualified lead"


print(f"\n{'='*70}")
print(f"E2E TEST RESULTS ({elapsed:.1f}s)")
print(f"{'='*70}\n")

qualified = 0
correct = 0
total = len(rows)

for i, r in enumerate(rows):
    is_q = is_qualified_lead_row(r)
    reject = get_reject_reason(r)
    owner = r.get("owner_name", "Unknown")
    conf = float(r.get("confidence_score") or 0)
    solo = r.get("solo", False)
    od = r.get("_debug", {}).get("owner_detection", {})
    reason = od.get("reason", "")[:100]

    # Match against expected
    expected = TEST_BUSINESSES[i] if i < len(TEST_BUSINESSES) else None
    exp_q = expected["expected_qualified"] if expected else None
    exp_owner = expected["expected_owner"] if expected else None

    status = "✓ QUALIFIED" if is_q else "✗ REJECTED"
    match = ""
    if exp_q is not None:
        if is_q == exp_q:
            correct += 1
            match = " [CORRECT]"
        else:
            match = " [WRONG]"

    if is_q:
        qualified += 1

    print(f"  [{i+1}] {r.get('business_name', '?')[:45]}")
    print(f"      {status}{match}")
    print(f"      owner={owner}  conf={conf:.2f}  solo={solo}")
    print(f"      reason: {reason}")
    if not is_q:
        print(f"      reject: {reject}")
    if exp_owner and is_q:
        owner_match = "✓" if exp_owner.lower() in str(owner).lower() else "✗"
        print(f"      expected_owner={exp_owner} {owner_match}")
    print()

print(f"{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  Businesses processed: {total}")
print(f"  Qualified leads:      {qualified}/{total}")
print(f"  Accuracy vs expected: {correct}/{total} ({100*correct/total:.0f}%)" if total else "  N/A")
print(f"  Time:                 {elapsed:.1f}s")
print(f"{'='*70}")

if correct == total:
    print("\n✓ ALL TESTS PASSED — Gemini owner detection is working correctly in the DataForSEO pipeline!")
elif qualified == 0:
    print("\n✗ FAIL: No qualified leads. Gemini detection may not be running.")
    print("  Check GOOGLE_API_KEY in .env")
else:
    print(f"\n⚠ {total - correct} mismatches vs expected. Review the results above.")
