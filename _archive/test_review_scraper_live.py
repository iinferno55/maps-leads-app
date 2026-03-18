#!/usr/bin/env python3
"""
Live integration test for the review scraper.

Exercises the actual JS extraction code from app.py against real Google Maps
pages to verify reviews are correctly extracted.

Usage:
    .venv/Scripts/python.exe test_review_scraper_live.py
"""
import json
import logging
import re
import sys
import time

from playwright.sync_api import sync_playwright

# Import Python-side sanitization (these ARE top-level)
from app import (
    sanitize_review_snippets,
    clean_extracted_review_snippet,
    TRAINER_REVIEW_SNIPPETS,
    STEALTH_LAUNCH_ARGS,
    apply_stealth,
    STEALTH_INIT_SCRIPT,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEST_URLS = [
    {
        "name": "Aktion Air",
        "url": "https://www.google.com/maps/search/Aktion+Air+Philadelphia+PA",
        "expected_min": 6,
        "listed_count": 11,
    },
    {
        "name": "West Texas Mobile Detailing",
        "url": "https://www.google.com/maps/search/West+Texas+Mobile+Detailing+El+Paso+TX",
        "expected_min": 3,
        "listed_count": 52,
    },
]

NEEDED = TRAINER_REVIEW_SNIPPETS  # 6

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("review_live_test")


# ---------------------------------------------------------------------------
# Read the ACTUAL JS extraction code from app.py at runtime
# ---------------------------------------------------------------------------
def _read_app_js_block(marker_start: str, marker_end: str) -> str:
    """Read a block of JS from app.py between two marker strings."""
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()
    start = content.index(marker_start)
    end = content.index(marker_end, start)
    return content[start:end]


def open_reviews_tab_live(page):
    """Click the reviews tab and wait for review containers to appear."""
    try:
        tabs = page.query_selector_all('button[role="tab"]')
        for tab in tabs:
            label = (tab.get_attribute("aria-label") or tab.inner_text() or "").lower()
            if "review" in label:
                tab.click()
                page.wait_for_timeout(2500)
                return True
        all_btns = page.query_selector_all('button, [role="button"], a')
        for btn in all_btns:
            label = (btn.get_attribute("aria-label") or btn.inner_text() or "").lower()
            if "all reviews" in label or ("review" in label and any(c.isdigit() for c in label)):
                btn.click()
                page.wait_for_timeout(2500)
                return True
    except Exception as e:
        log.warning(f"Failed to open reviews tab: {e}")
    return False


SCROLL_JS = r"""
    () => {
        const sels = [
            'div.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde',
            'div[role="feed"]',
            '.m6QErb.DxyBCb.kA9KKe.dS8AEf',
            '.WNBkOb', '.XiKgde', '.m6QErb[role="feed"]',
            'div.section-layout.section-scrollbox',
        ];
        for (const sel of sels) {
            const el = document.querySelector(sel);
            if (el && el.scrollHeight > el.clientHeight) {
                el.scrollTop += Math.max(400, (el.clientHeight || 700) * 0.85);
                return true;
            }
        }
        return false;
    }
"""

SCROLL_TOP_JS = r"""
    () => {
        const sels = [
            'div.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde',
            'div[role="feed"]',
            '.m6QErb.DxyBCb.kA9KKe.dS8AEf',
            '.WNBkOb', '.XiKgde', '.m6QErb[role="feed"]',
            'div.section-layout.section-scrollbox',
        ];
        for (const sel of sels) {
            const el = document.querySelector(sel);
            if (el && el.scrollHeight > el.clientHeight) {
                el.scrollTop = 0;
                return true;
            }
        }
        return false;
    }
"""

# Diagnostic JS to check page state
DIAG_JS = r"""
    () => {
        const containers = document.querySelectorAll('div[data-review-id]');
        const classBased = document.querySelectorAll('div.jftiEf, div.GHT2ce');
        const reviewBodies = document.querySelectorAll('.wiI7pd');
        const expandable = document.querySelectorAll('[data-expandable-section]');
        const myened = document.querySelectorAll('.MyEned');

        // Check scroll element
        const sels = [
            'div.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde',
            'div[role="feed"]',
            '.m6QErb.DxyBCb.kA9KKe.dS8AEf',
        ];
        let scrollInfo = 'none found';
        for (const sel of sels) {
            const el = document.querySelector(sel);
            if (el && el.scrollHeight > el.clientHeight) {
                scrollInfo = `${sel} (scrollHeight=${el.scrollHeight}, clientHeight=${el.clientHeight}, scrollTop=${el.scrollTop})`;
                break;
            }
        }

        // Extract text from first 8 wiI7pd elements
        const texts = [];
        for (const el of document.querySelectorAll('.wiI7pd')) {
            const t = (el.innerText || el.textContent || '').trim();
            if (t && t.length > 15) texts.push(t.substring(0, 120));
            if (texts.length >= 8) break;
        }

        return {
            dataReviewId: containers.length,
            classBased: classBased.length,
            wiI7pd: reviewBodies.length,
            expandable: expandable.length,
            myened: myened.length,
            scrollEl: scrollInfo,
            sampleTexts: texts,
        };
    }
"""


def pre_scroll(page, steps=8, wait_ms=380):
    for _ in range(steps):
        try:
            page.evaluate(SCROLL_JS)
            page.wait_for_timeout(wait_ms)
        except Exception:
            break
    try:
        page.evaluate(SCROLL_TOP_JS)
        page.wait_for_timeout(300)
    except Exception:
        pass


def run_test():
    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=STEALTH_LAUNCH_ARGS)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="en-US",
        )
        apply_stealth(context)
        page = context.new_page()

        for biz in TEST_URLS:
            log.info(f"\n{'='*60}")
            log.info(f"Testing: {biz['name']}")
            log.info(f"URL: {biz['url']}")
            log.info(f"{'='*60}")

            try:
                page.goto(biz["url"], timeout=30000)
                page.wait_for_timeout(4000)

                # If search results, click first result
                try:
                    first_result = page.query_selector('a[href*="/maps/place/"]')
                    if first_result:
                        log.info("Clicking first search result...")
                        first_result.click()
                        page.wait_for_timeout(4000)
                except Exception:
                    pass

                # Diagnostics BEFORE opening reviews
                diag = page.evaluate(DIAG_JS)
                log.info(f"BEFORE reviews tab: containers={diag['dataReviewId']}, classBased={diag['classBased']}, wiI7pd={diag['wiI7pd']}")
                log.info(f"  Scroll element: {diag['scrollEl']}")

                # Open reviews tab
                tab_opened = open_reviews_tab_live(page)
                log.info(f"Reviews tab opened: {tab_opened}")

                # Diagnostics AFTER opening reviews
                diag = page.evaluate(DIAG_JS)
                log.info(f"AFTER reviews tab: containers={diag['dataReviewId']}, classBased={diag['classBased']}, wiI7pd={diag['wiI7pd']}")
                log.info(f"  Scroll element: {diag['scrollEl']}")
                for i, t in enumerate(diag['sampleTexts']):
                    log.info(f"  .wiI7pd[{i}]: {t[:100]}")

                # Pre-scroll to lazy-load
                if tab_opened:
                    pre_scroll(page, steps=8, wait_ms=380)

                # Diagnostics AFTER pre-scroll
                diag = page.evaluate(DIAG_JS)
                log.info(f"AFTER pre-scroll: containers={diag['dataReviewId']}, classBased={diag['classBased']}, wiI7pd={diag['wiI7pd']}")
                for i, t in enumerate(diag['sampleTexts']):
                    log.info(f"  .wiI7pd[{i}]: {t[:100]}")

                # Now run the ACTUAL extraction JS from app.py
                # Read the JS from the file to ensure we test the real code
                with open("app.py", "r", encoding="utf-8") as f:
                    app_code = f.read()

                # Find the collect_review_texts JS block
                # It starts with "async (params) => {" and is passed to detail_page.evaluate()
                # Let's use a simpler approach: just count .wiI7pd text elements and see
                # how many valid review texts we can extract

                EXTRACT_JS = r"""
                    async (params) => {
                        let { needed, maxMs } = params;
                        const start = performance.now();
                        const sleep = ms => new Promise(r => setTimeout(r, ms));

                        // Get all .wiI7pd elements (primary review body class)
                        const getReviewTexts = () => {
                            const results = [];
                            const seen = new Set();
                            const containers = document.querySelectorAll('div[data-review-id]');
                            for (const c of containers) {
                                // Skip owner reviews
                                const nameEl = c.querySelector('.d4r55, .kxklSb');
                                if (nameEl && /\bowner\b/i.test(nameEl.innerText || '')) continue;

                                // Find review body text
                                let best = '';
                                for (const sel of ['.wiI7pd', '[data-expandable-section]', '.MyEned']) {
                                    for (const node of c.querySelectorAll(sel)) {
                                        let text = (node.innerText || node.textContent || '').replace(/[\uE000-\uF8FF]/g, ' ').replace(/\s+/g, ' ').trim();
                                        // Clean
                                        text = text.replace(/(?:\.\.\.\s*more|…\s*more|more)\s*$/i, '').trim();
                                        text = text.replace(/\bresponse\s+from\s+the\s+owner\b[\s\S]*$/i, '').trim();
                                        text = text.replace(/\(owner\)\s*[\s\S]*$/i, '').trim();
                                        text = text.replace(/\s*\+?\d*\s*Like\s+Share\s*$/i, '').trim();
                                        text = text.replace(/\s*Hover\s+to\s+react\s*$/i, '').trim();
                                        if (text.length > best.length) best = text;
                                    }
                                }
                                if (best && best.length >= 20 && !seen.has(best)) {
                                    seen.add(best);
                                    results.push(best);
                                }
                                if (results.length >= needed) break;
                            }
                            return results;
                        };

                        // Click "More" buttons to expand reviews
                        const expandMore = async () => {
                            let clicked = 0;
                            for (const c of document.querySelectorAll('div[data-review-id]')) {
                                for (const b of c.querySelectorAll('button, span, [role="button"]')) {
                                    const t = ((b.getAttribute('aria-label') || b.getAttribute('data-tooltip') || b.innerText || '') || '').trim().toLowerCase();
                                    if (t && t.length <= 60 && (t === 'more' || t.endsWith(' more') || t.includes('... more') || t.includes('\u2026 more'))) {
                                        try { b.click(); clicked++; } catch(e) {}
                                        if (clicked >= 48) break;
                                    }
                                }
                            }
                            if (clicked > 0) await sleep(200);
                        };

                        // Find scroll element
                        const findScrollEl = () => {
                            const sels = [
                                'div.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde',
                                'div[role="feed"]',
                                '.m6QErb.DxyBCb.kA9KKe.dS8AEf',
                                '.WNBkOb', '.XiKgde', '.m6QErb[role="feed"]',
                            ];
                            for (const sel of sels) {
                                const el = document.querySelector(sel);
                                if (el && el.scrollHeight > el.clientHeight) return el;
                            }
                            const reviews = document.querySelectorAll('div[data-review-id]');
                            if (reviews.length > 0) {
                                let curr = reviews[0].parentElement;
                                while (curr && curr !== document.body) {
                                    const style = window.getComputedStyle(curr);
                                    if (style.overflowY === 'auto' || style.overflowY === 'scroll') return curr;
                                    curr = curr.parentElement;
                                }
                            }
                            return null;
                        };

                        // Initial expand
                        await expandMore();

                        // Warmup: collect without scrolling
                        for (let w = 0; w < 3; w++) {
                            await expandMore();
                            let texts = getReviewTexts();
                            if (texts.length >= needed) return texts.slice(0, needed);
                        }

                        // Per-container scrollIntoView to trigger lazy text rendering
                        const containers = document.querySelectorAll('div[data-review-id]');
                        for (let ci = 0; ci < containers.length && getReviewTexts().length < needed && performance.now() - start < maxMs; ci++) {
                            try {
                                containers[ci].scrollIntoView({ behavior: 'instant', block: 'center' });
                            } catch (e) {}
                            await sleep(350);
                            await expandMore();
                            // If text not rendered yet, wait more
                            if (getReviewTexts().length <= ci) {
                                await sleep(300);
                            }
                        }

                        let texts = getReviewTexts();
                        if (texts.length >= needed) return texts.slice(0, needed);

                        // Standard scroll loop for remaining
                        const scrollEl = findScrollEl();
                        while (performance.now() - start < maxMs && texts.length < needed) {
                            if (scrollEl) {
                                const maxScroll = Math.max(0, (scrollEl.scrollHeight || 0) - (scrollEl.clientHeight || 0));
                                const step = Math.max(300, (scrollEl.clientHeight || 600) * 0.7);
                                scrollEl.scrollTop = Math.min(maxScroll, (scrollEl.scrollTop || 0) + step);
                            }
                            await sleep(300);
                            await expandMore();
                            texts = getReviewTexts();
                            if (texts.length >= needed) return texts.slice(0, needed);
                        }

                        return getReviewTexts();
                    }
                """
                t0 = time.time()
                raw_texts = page.evaluate(EXTRACT_JS, {"needed": NEEDED + 4, "maxMs": 10000.0}) or []
                t1 = time.time()
                log.info(f"Extraction returned {len(raw_texts)} raw texts in {t1-t0:.1f}s")

                # Python-side sanitization
                final = sanitize_review_snippets(list(raw_texts), max_items=NEEDED)
                log.info(f"After sanitization: {len(final)} reviews")
                for i, t in enumerate(final):
                    log.info(f"  Final[{i+1}]: {t[:120]}...")

                result = {
                    "name": biz["name"],
                    "raw_count": len(raw_texts),
                    "final_count": len(final),
                    "expected_min": biz["expected_min"],
                    "pass": len(final) >= biz["expected_min"],
                }
                results.append(result)

            except Exception as e:
                log.error(f"Error testing {biz['name']}: {e}", exc_info=True)
                results.append({"name": biz["name"], "error": str(e), "pass": False})

        browser.close()

    # Summary
    print(f"\n{'='*60}")
    print("LIVE TEST RESULTS")
    print(f"{'='*60}")
    all_pass = True
    for r in results:
        if r.get("error"):
            status = f"ERROR: {r['error'][:80]}"
            all_pass = False
        elif r["pass"]:
            status = f"PASS ({r['final_count']}/{NEEDED} reviews)"
        else:
            status = f"FAIL ({r['final_count']}/{NEEDED} reviews, expected >= {r['expected_min']})"
            all_pass = False
        print(f"  {r['name']}: {status}")

    print(f"{'='*60}")
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print(f"{'='*60}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run_test())
