import json
import os
import time
import traceback
import re

import app


cases = [
    ("plumber", "phoenix"),
    ("HVAC", "dallas"),
    ("electrician", "seattle"),
]
results = []
start = time.time()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_validation_results.json")

for niche, city in cases:
    t0 = time.time()
    try:
        rows = app.scrape_google_maps(
            city=city,
            niche=niche,
            max_pages=2,
            headless=True,
            max_businesses=6,
            run_owner_detection=False,
            review_snippets_target=app.TRAINER_REVIEW_SNIPPETS,
        )
        passed = False
        picked = None
        diagnostics = []
        for r in rows:
            snippets = [
                str(s).strip()
                for s in list((r.get("_debug") or {}).get("sample_review_snippets") or [])
                if str(s).strip()
            ]
            first6 = snippets[:6]
            has_owner = any("response from the owner" in s.lower() for s in first6)
            has_more_marker = any(re.search(r"(?:\.\.\.\s*more|…\s*more|\bmore)$", s.lower()) for s in first6)
            diagnostics.append(
                {
                    "business": str(r.get("business_name") or ""),
                    "snippet_count": len(snippets),
                    "first6_count": len(first6),
                    "owner_response": has_owner,
                    "has_more_marker": has_more_marker,
                }
            )
            if len(first6) < 6:
                continue
            if has_owner:
                continue
            if has_more_marker:
                continue
            passed = True
            picked = r
            break

        elapsed = round(time.time() - t0, 2)
        results.append(
            {
                "niche": niche,
                "city": city,
                "rows": len(rows),
                "pass": passed,
                "elapsed_s": elapsed,
                "picked_business": (picked or {}).get("business_name", ""),
                "picked_snippets": len(list(((picked or {}).get("_debug") or {}).get("sample_review_snippets") or [])) if picked else 0,
                "diagnostics": diagnostics,
            }
        )
    except Exception as e:
        results.append(
            {
                "niche": niche,
                "city": city,
                "pass": False,
                "error": str(e),
                "trace": traceback.format_exc(),
            }
        )

payload = {"total_elapsed_s": round(time.time() - start, 2), "results": results}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(json.dumps(payload, indent=2))
