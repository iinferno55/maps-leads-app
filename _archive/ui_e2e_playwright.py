import os
import sys
import time
import re


def _get_base_url() -> str:
    return os.environ.get("APP_URL", "http://localhost:8501").rstrip("/")


def _wait_for_text(page, text: str, timeout_ms: int = 120_000) -> None:
    # Streamlit uses async rendering; run the check in-page so Playwright's timeout applies.
    page.wait_for_function(
        "(t) => !!document.body && (document.body.innerText || '').toLowerCase().includes(String(t).toLowerCase())",
        arg=text,
        timeout=timeout_ms,
    )


def _ensure_sidebar_open(page) -> None:
    # Streamlit can auto-collapse the sidebar on narrow viewports.
    # Try to open it if the "Open sidebar" control exists.
    try:
        btn = page.locator('button[aria-label*="sidebar" i]').first
        if btn.count() > 0:
            # Prefer the explicit "Open sidebar" control if present.
            try:
                open_btn = page.locator('button[aria-label="Open sidebar"]').first
                if open_btn.count() > 0:
                    open_btn.click(timeout=1500)
                    time.sleep(0.2)
                    return
            except Exception:
                pass
            btn.click(timeout=1500)
            time.sleep(0.2)
    except Exception:
        pass


def test_lead_finder_fast_smoke(page) -> None:
    # Lead Finder is the default page.
    _wait_for_text(page, "Cold Calling Leads", timeout_ms=60_000)

    # Sidebar controls live in the left; Streamlit renders them in DOM order.
    # Turn on debug mode and cap businesses.
    _ensure_sidebar_open(page)
    sidebar = page.locator('section[data-testid="stSidebar"]')

    # Set max pages to minimum for speed.
    print("[STEP] Lead Finder: set max pages min", flush=True)
    max_pages = sidebar.get_by_role("slider", name=re.compile(r"Max result pages", re.IGNORECASE)).first
    max_pages.click()
    page.keyboard.press("Home")

    # Turn on debug mode (click label text; the underlying input is 0x0 sized).
    print("[STEP] Lead Finder: enable debug mode", flush=True)
    sidebar.get_by_text("Debug mode (limit businesses for testing)", exact=False).first.click()

    # Keep debug max businesses at minimum.
    print("[STEP] Lead Finder: set debug max businesses min", flush=True)
    dbg_max = sidebar.get_by_role("slider", name=re.compile(r"Debug max businesses", re.IGNORECASE)).first
    dbg_max.click()
    page.keyboard.press("Home")

    # Target leads to minimum.
    print("[STEP] Lead Finder: set target leads min", flush=True)
    tgt = sidebar.get_by_role("slider", name=re.compile(r"Target qualified leads", re.IGNORECASE)).first
    tgt.click()
    page.keyboard.press("Home")

    # UI smoke only (do not run a full scrape; that can take minutes).
    print("[STEP] Lead Finder: verify Start Scraping visible", flush=True)
    sidebar.get_by_role("button", name="Start Scraping").first.wait_for(timeout=30_000)


def test_review_trainer_pull_chunk(page) -> None:
    # Streamlit renders the actual <input type="radio"> as 0x0; click the label text instead.
    page.get_by_text("Review Trainer", exact=False).first.click()
    _ensure_sidebar_open(page)

    sidebar = page.locator('section[data-testid="stSidebar"]')
    # Set trainer inputs for the reported slow case.
    sidebar.get_by_label("City (trainer)").fill("phoenix")
    # Select niche
    # Streamlit selectbox: open dropdown, type to filter, hit Enter.
    sidebar.get_by_label("Niche (trainer)").click()
    page.keyboard.type("mobile")
    page.keyboard.press("Enter")
    # Ensure max pages = 2
    mp = sidebar.get_by_label("Max pages (trainer)").first
    mp.click()
    # Move to 2 (Home -> 1, ArrowRight -> 2)
    page.keyboard.press("Home")
    page.keyboard.press("ArrowRight")
    # Listings per chunk = 5
    chunk = sidebar.get_by_label("Listings per chunk").first
    chunk.click()
    # Try to set to 5 (Home then 4 right steps)
    page.keyboard.press("Home")
    for _ in range(4):
        page.keyboard.press("ArrowRight")
    # Ensure triage is OFF if visible.
    try:
        tri = sidebar.get_by_text("Local AI triage (slower, optional)", exact=False).first
        if tri.count() > 0:
            # If checked, click to toggle off (best-effort).
            tri.click()
    except Exception:
        pass
    print("[STEP] Trainer: click Pull first chunk", flush=True)
    sidebar.get_by_role("button", name="Pull first chunk").click()
    # Pull status: "Pulled X listings" / "Added X listings"
    print("[STEP] Trainer: wait for pull result", flush=True)
    try:
        _wait_for_text(page, "Pulled", timeout_ms=240_000)
    except Exception:
        _wait_for_text(page, "Added", timeout_ms=240_000)


def test_review_trainer_label_and_next(page) -> None:
    page.get_by_text("Review Trainer", exact=False).first.click()
    _ensure_sidebar_open(page)

    # If no rows yet, pull a tiny chunk (uses free scrape path)
    if page.get_by_text("Pull first chunk", exact=False).count() > 0:
        # If the empty-state info is visible, pull
        if page.get_by_text("Click", exact=False).count() > 0:
            page.get_by_role("button", name="Pull first chunk").click()
            try:
                _wait_for_text(page, "Pulled", timeout_ms=240_000)
            except Exception:
                _wait_for_text(page, "Added", timeout_ms=240_000)

    # Select "yes" for the first listing (do NOT save; avoid touching label CSV).
    print("[STEP] Trainer: label yes (no save)", flush=True)
    _wait_for_text(page, "Would you call this lead?", timeout_ms=60_000)
    decision = page.locator('div[role="radiogroup"][aria-label="Would you call this lead?"]').first
    decision.scroll_into_view_if_needed(timeout=8000)
    decision.get_by_text(re.compile(r"^yes$", re.IGNORECASE)).first.click()
    # We mainly assert no crash; Streamlit will stay responsive.
    _wait_for_text(page, "Label Listings", timeout_ms=30_000)


def main() -> int:
    from playwright.sync_api import sync_playwright

    base = _get_base_url()
    failures = []
    print(f"[INFO] Base URL: {base}", flush=True)
    with sync_playwright() as p:
        print("[INFO] Launching browser...", flush=True)
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 980})
        page = context.new_page()
        print("[INFO] Navigating...", flush=True)
        page.goto(base, wait_until="domcontentloaded", timeout=60_000)

        # Wait for app title render.
        print("[INFO] Waiting for app to render...", flush=True)
        _wait_for_text(page, "Cold Calling Leads", timeout_ms=60_000)
        print("[INFO] App render OK", flush=True)

        tests = [
            ("lead_finder_ui_smoke", test_lead_finder_fast_smoke),
            ("review_trainer_pull_phoenix_mobile", test_review_trainer_pull_chunk),
            ("review_trainer_label_no_save", test_review_trainer_label_and_next),
        ]
        for name, fn in tests:
            start = time.time()
            try:
                fn(page)
                dur = time.time() - start
                print(f"[PASS] {name} in {dur:.1f}s")
            except Exception as e:
                dur = time.time() - start
                failures.append((name, dur, repr(e)))
                print(f"[FAIL] {name} in {dur:.1f}s: {e}")

        context.close()
        browser.close()

    if failures:
        print("\nFailures:")
        for name, dur, err in failures:
            print(f"- {name} ({dur:.1f}s): {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

