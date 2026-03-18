import json
import time

import app


CASES = [
    ("mobile detailer", "san jose"),
    ("plumber", "phoenix"),
    ("electrician", "seattle"),
]


def snippets_from_row(row: dict) -> list[str]:
    txt = str(row.get("_reviews_text") or "")
    return [s.strip() for s in txt.split("\n\n") if s.strip()]


def row_passes(row: dict) -> tuple[bool, str]:
    snips = snippets_from_row(row)
    first6 = snips[:6]
    if len(first6) < 6:
        return False, f"only_{len(first6)}"
    if any("response from the owner" in s.lower() for s in first6):
        return False, "owner_response_present"
    if any(s.strip().endswith("...") or s.strip().endswith("…") for s in first6):
        return False, "unexpanded_ellipsis"
    return True, "ok"


def run_case(niche: str, city: str) -> dict:
    t0 = time.time()
    rows = app.scrape_google_maps(
        city=city,
        niche=niche,
        max_pages=2,
        headless=True,
        max_businesses=8,
        run_owner_detection=False,
        review_snippets_target=app.TRAINER_REVIEW_SNIPPETS,
    )

    best = None
    fail_reasons = {}
    for row in rows:
        ok, reason = row_passes(row)
        if ok:
            best = row
            break
        fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    elapsed = round(time.time() - t0, 2)
    if best is None:
        return {
            "niche": niche,
            "city": city,
            "pass": False,
            "elapsed_s": elapsed,
            "rows": len(rows),
            "fail_reasons": fail_reasons,
            "top_rows": [
                {
                    "business_name": r.get("business_name", ""),
                    "snippet_count": len(snippets_from_row(r)),
                    "debug": r.get("_debug", {}),
                }
                for r in rows[:3]
            ],
        }

    chosen_snips = snippets_from_row(best)
    return {
        "niche": niche,
        "city": city,
        "pass": True,
        "elapsed_s": elapsed,
        "rows": len(rows),
        "picked_business": best.get("business_name", ""),
        "picked_snippets": len(chosen_snips),
        "first6": chosen_snips[:6],
        "debug": best.get("_debug", {}),
    }


def main() -> None:
    start = time.time()
    results = [run_case(niche, city) for niche, city in CASES]
    payload = {
        "total_elapsed_s": round(time.time() - start, 2),
        "passed": sum(1 for r in results if r.get("pass")) == len(results),
        "results": results,
    }
    with open("trainer_reliability_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
