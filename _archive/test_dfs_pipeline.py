"""
Minimal LIVE DataForSEO pipeline test — calls scrape_dataforseo directly
with max_businesses=2 to use minimal credits. Shows full logging output.
"""
import os, sys, logging
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

# Force ALL logs (including INFO) to console so we see everything
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s  %(message)s")

from app import scrape_dataforseo, is_qualified_lead_row, MIN_CONFIDENCE

def status_cb(msg):
    print(f"[STATUS] {msg}")

def progress_cb(pct, msg=None):
    pass

print("=" * 60)
print("LIVE DataForSEO pipeline test (max 2 businesses)")
print("=" * 60)
print(f"GOOGLE_API_KEY set: {bool(os.getenv('GOOGLE_API_KEY'))}")
print(f"DATAFORSEO_LOGIN set: {bool(os.getenv('DATAFORSEO_LOGIN'))}")
print()

rows = scrape_dataforseo(
    city="Seattle",
    niche="locksmith",
    max_businesses=2,
    progress_callback=progress_cb,
    status_callback=status_cb,
    dataforseo_location_name="Seattle,Washington,United States",
    reviews_depth=10,
    reviews_priority=2,
)

print(f"\n{'='*60}")
print(f"RESULTS: {len(rows)} businesses")
print(f"{'='*60}\n")

for i, r in enumerate(rows):
    is_q = is_qualified_lead_row(r)
    od = r.get("_debug", {}).get("owner_detection", {})
    print(f"[{i+1}] {r.get('business_name','?')[:45]}")
    print(f"    owner={r.get('owner_name')} conf={float(r.get('confidence_score') or 0):.2f} solo={r.get('solo')}")
    print(f"    qualified={is_q}")
    print(f"    reason: {od.get('reason','')[:100]}")
    print(f"    reviews_in_debug: {r.get('_debug',{}).get('review_snippet_count',0)}")
    print()

qualified = sum(1 for r in rows if is_qualified_lead_row(r))
print(f"Summary: {qualified}/{len(rows)} qualified")
