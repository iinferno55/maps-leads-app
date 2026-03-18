"""
Minimal live DataForSEO test — fetches 1 business with reviews to diagnose
why owner detection shows confidence_score=0 in the real pipeline.
Uses minimal credits (1 maps search + 1 reviews fetch).

Usage:  py test_dataforseo_live.py
"""
import json, os, sys, logging

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

# Force all logs to console for this test
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from dataforseo_client import maps_search, fetch_reviews_text, MapsPlace, DataForSeoError

login = os.getenv("DATAFORSEO_LOGIN", "")
password = os.getenv("DATAFORSEO_PASSWORD", "")
if not login or not password:
    print("ERROR: DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD not set in .env")
    sys.exit(1)

print("=" * 60)
print("STEP 1: maps_search — fetch 1 page of results")
print("=" * 60)

try:
    places = maps_search(
        login=login,
        password=password,
        keyword="locksmith in Seattle",
        location_name="Seattle,Washington,United States",
        language_code="en",
        depth=20,  # minimal
    )
except Exception as e:
    print(f"maps_search FAILED: {e}")
    sys.exit(1)

print(f"\nReturned {len(places)} places\n")

# Show first 5 with details
for i, p in enumerate(places[:5]):
    print(f"  [{i}] {p.title}")
    print(f"      address={p.address}")
    print(f"      phone={p.phone}")
    print(f"      reviews_count={p.reviews_count}")
    print(f"      place_id={p.place_id}")
    print(f"      cid={p.cid}")
    print(f"      work_hours={'present' if p.work_hours else 'None'}")
    print()

# Pick the first place with reviews
target = None
for p in places:
    if (p.reviews_count or 0) >= 5:
        target = p
        break

if not target:
    print("No business with 5+ reviews found. Exiting.")
    sys.exit(1)

print("=" * 60)
print(f"STEP 2: fetch_reviews_text for '{target.title}'")
print("=" * 60)
print(f"  place_id={target.place_id}")
print(f"  cid={target.cid}")
print(f"  reviews_count={target.reviews_count}")

try:
    reviews, payload = fetch_reviews_text(
        login=login,
        password=password,
        place_id=target.place_id,
        cid=target.cid,
        keyword_fallback=target.title,
        location_name="Seattle,Washington,United States",
        language_code="en",
        depth=10,  # minimal
        priority=2,
    )
except DataForSeoError as e:
    print(f"\nfetch_reviews_text FAILED: {e}")
    reviews = []
    payload = {"error": str(e)}

print(f"\nReviews returned: {len(reviews)}")
if reviews:
    for i, r in enumerate(reviews[:3]):
        print(f"  [{i}] {r[:150]}...")
else:
    print("  *** NO REVIEWS RETURNED ***")
    print(f"  Payload keys: {list(payload.keys()) if payload else 'None'}")
    if payload and "error" in payload:
        print(f"  Error: {payload['error']}")

owner_answers = (payload or {}).get("_owner_answers", [])
print(f"\nOwner replies: {len(owner_answers)}")

reviews_text = "\n\n".join(reviews) if reviews else ""
print(f"Reviews text length: {len(reviews_text)} chars")

# ── STEP 3: Run detect_owner on the reviews ──

print("\n" + "=" * 60)
print("STEP 3: detect_owner")
print("=" * 60)

if not reviews_text or len(reviews_text.strip()) < 50:
    print(f"  *** REVIEWS TEXT TOO SHORT ({len(reviews_text)} chars) — detection will SKIP ***")
    print("  This is why confidence_score = 0 for all businesses!")
    print("  Root cause: DataForSEO is not returning review text.")
else:
    from app import detect_owner, detect_owner_with_gemini
    print(f"\n  Testing Gemini detection directly...")
    gemini_result = detect_owner_with_gemini(reviews_text, target.title)
    print(f"  Gemini: owner={gemini_result.get('owner_name')} solo={gemini_result.get('solo')} "
          f"conf={gemini_result.get('confidence', 0):.2f} reason={gemini_result.get('reason', '')[:80]}")

    print(f"\n  Testing full detect_owner pipeline...")
    full_result = detect_owner(reviews_text, target.title)
    print(f"  Full:   owner={full_result.get('owner_name')} solo={full_result.get('solo')} "
          f"conf={full_result.get('confidence', 0):.2f} reason={full_result.get('reason', '')[:80]}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
