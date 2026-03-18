import json
import re
import time

from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8501"
CASES = [
    ("plumber", "phoenix"),
    ("HVAC", "dallas"),
    ("electrician", "seattle"),
]


def wait_for_text(page, text: str, timeout_ms: int = 240_000) -> None:
    page.wait_for_function(
        "(t) => !!document.body && (document.body.innerText || '').toLowerCase().includes(String(t).toLowerCase())",
        arg=text,
        timeout=timeout_ms,
    )


def ensure_sidebar_open(page) -> None:
    try:
        open_btn = page.locator('button[aria-label="Open sidebar"]').first
        if open_btn.count() > 0:
            open_btn.click(timeout=1200)
            time.sleep(0.2)
            return
    except Exception:
        pass


def _set_slider_value(page, label: str, target: int, min_value: int = 1) -> None:
    slider = page.get_by_label(label).first
    slider.click()
    page.keyboard.press("Home")
    for _ in range(max(0, target - min_value)):
        page.keyboard.press("ArrowRight")


def _get_review_snippets(page) -> list[str]:
    snippets = page.evaluate(
                r"""
        () => {
          const out = [];
          const nodes = Array.from(document.querySelectorAll("div[style*='max-width:65ch']"));
          for (const node of nodes) {
            let text = (node.innerText || node.textContent || '').replace(/\s+/g, ' ').trim();
            text = text.replace(/^\d+\.\s*/, '').trim();
            if (!text) continue;
            out.push(text);
            if (out.length >= 6) break;
          }
          return out;
        }
        """
    ) or []
    return [str(s).strip() for s in snippets if str(s).strip()]


def run_case(page, niche: str, city: str) -> dict:
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=120_000)
    wait_for_text(page, "Cold Calling Leads", timeout_ms=120_000)

    page.get_by_text("Review Trainer", exact=False).first.click()
    ensure_sidebar_open(page)

    sidebar = page.locator('section[data-testid="stSidebar"]')
    sidebar.get_by_label("City (trainer)").fill(city)

    sidebar.get_by_label("Niche (trainer)").click()
    page.keyboard.press("Control+a")
    page.keyboard.type(niche)
    page.keyboard.press("Enter")

    _set_slider_value(page, "Max pages (trainer)", target=2, min_value=1)
    _set_slider_value(page, "Listings per chunk", target=5, min_value=1)

    triage_toggle = sidebar.get_by_text("Local AI triage (slower, optional)", exact=False).first
    try:
        if triage_toggle.count() > 0:
            triage_toggle.click()
    except Exception:
        pass

    sidebar.get_by_role("button", name="Pull first chunk").click()

    try:
        wait_for_text(page, "Pulled", timeout_ms=300_000)
    except Exception:
        wait_for_text(page, "Added", timeout_ms=300_000)

    wait_for_text(page, "Review snippets (up to 6):", timeout_ms=120_000)
    snippets = _get_review_snippets(page)
    first6 = snippets[:6]

    has_six = len(first6) >= 6
    has_owner_response = any("response from the owner" in s.lower() for s in first6)
    has_unexpanded = any(re.search(r"(?:\.\.\.\s*more|…\s*more|\bmore)$", s.lower()) for s in first6)
    substantial_count = sum(1 for s in first6 if len(s) >= 60)

    passed = has_six and not has_owner_response and not has_unexpanded and substantial_count >= 5
    return {
        "niche": niche,
        "city": city,
        "pass": passed,
        "snippet_count": len(snippets),
        "first6_substantial": substantial_count,
        "owner_response_in_first6": has_owner_response,
        "unexpanded_marker_in_first6": has_unexpanded,
        "sample": first6,
    }


def main() -> int:
    results = []
    started = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 980})
        page = context.new_page()

        for niche, city in CASES:
            t0 = time.time()
            try:
                result = run_case(page, niche=niche, city=city)
                result["elapsed_s"] = round(time.time() - t0, 2)
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "niche": niche,
                        "city": city,
                        "pass": False,
                        "error": str(e),
                        "elapsed_s": round(time.time() - t0, 2),
                    }
                )

        context.close()
        browser.close()

    payload = {
        "total_elapsed_s": round(time.time() - started, 2),
        "results": results,
        "all_passed": all(bool(r.get("pass")) for r in results),
    }
    with open("trainer_live_reliability_results_latest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
