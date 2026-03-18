#!/usr/bin/env python3
"""Quick headed-mode test for Aktion Air reviews."""
import sys, time, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aktion_test")

from playwright.sync_api import sync_playwright
from app import (
    sanitize_review_snippets, STEALTH_LAUNCH_ARGS, apply_stealth,
    TRAINER_REVIEW_SNIPPETS,
)

URL = "https://www.google.com/maps/search/HVAC+Philadelphia+PA"
TARGET = 6

def run():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, args=STEALTH_LAUNCH_ARGS)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="en-US",
        )
        apply_stealth(ctx)
        page = ctx.new_page()

        log.info(f"Navigating to {URL}")
        page.goto(URL, timeout=30000)
        page.wait_for_timeout(5000)

        # Scroll search results to find Aktion Air - use Playwright native click on the link
        found_aktion = False
        for scroll_attempt in range(40):
            # Find all result links in the feed
            links = page.query_selector_all('a[href*="/maps/place/"]')
            for link in links:
                aria = (link.get_attribute("aria-label") or "").strip()
                if "aktion" in aria.lower():
                    log.info(f"Found Aktion Air link: aria-label='{aria}'")
                    # Scroll it into view first
                    link.scroll_into_view_if_needed()
                    page.wait_for_timeout(500)
                    # Use Playwright native click (not JS click) - this properly navigates
                    link.click()
                    page.wait_for_timeout(6000)
                    found_aktion = True
                    break

            if found_aktion:
                break

            # Also check heading text as fallback for clicking
            if not found_aktion:
                found_heading = page.evaluate("""
                    () => {
                        const elements = document.querySelectorAll('.fontHeadlineSmall, .qBF1Pd, .NrDZNb, h3, [role="heading"]');
                        for (const el of elements) {
                            const text = (el.innerText || el.textContent || '').trim();
                            if (/aktion/i.test(text)) {
                                return text;
                            }
                        }
                        return null;
                    }
                """)
                if found_heading:
                    log.info(f"Found Aktion Air heading text: {found_heading} - trying to find parent link")
                    # Find the closest ancestor link for this heading
                    clicked = page.evaluate("""
                        () => {
                            const elements = document.querySelectorAll('.fontHeadlineSmall, .qBF1Pd, .NrDZNb, h3, [role="heading"]');
                            for (const el of elements) {
                                const text = (el.innerText || el.textContent || '').trim();
                                if (/aktion/i.test(text)) {
                                    // Walk up to find a clickable container
                                    let curr = el;
                                    for (let i = 0; i < 8; i++) {
                                        curr = curr.parentElement;
                                        if (!curr) break;
                                        if (curr.tagName === 'A' && curr.href && curr.href.includes('/maps/place/')) {
                                            curr.click();
                                            return 'clicked-link-' + curr.href.substring(0, 80);
                                        }
                                        if (curr.getAttribute && curr.getAttribute('role') === 'article') {
                                            curr.click();
                                            return 'clicked-article';
                                        }
                                    }
                                    // Last resort: click the heading element itself
                                    el.click();
                                    return 'clicked-heading';
                                }
                            }
                            return null;
                        }
                    """)
                    log.info(f"Click result: {clicked}")
                    if clicked:
                        page.wait_for_timeout(6000)
                        found_aktion = True
                        break

            # Scroll the results feed
            page.evaluate("""
                () => {
                    const feed = document.querySelector('div[role="feed"]') ||
                                 document.querySelector('.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde') ||
                                 document.querySelector('.m6QErb');
                    if (feed) feed.scrollTop += 600;
                }
            """)
            page.wait_for_timeout(1500)
            if scroll_attempt % 5 == 4:
                results = page.query_selector_all('a[href*="/maps/place/"]')
                log.info(f"Scroll attempt {scroll_attempt+1}, found {len(results)} place links")

        if not found_aktion:
            log.error("Could not find Aktion Air in search results")
            browser.close()
            return 1

        # Verify we're on the Aktion Air detail panel by checking h1/title
        # If after clicking the URL still shows search results, try navigating directly
        page.wait_for_timeout(2000)
        current_url = page.url
        log.info(f"Current URL after click: {current_url[:100]}")

        title = ""
        try:
            title = page.locator("h1").first.inner_text(timeout=5000)
        except Exception:
            pass
        log.info(f"Page h1 title: '{title}'")

        # If we're not on the Aktion Air page, check if we navigated to a place URL
        if "aktion" not in title.lower() and "/maps/place/" not in current_url:
            log.warning("May not be on Aktion Air page - trying direct search")
            # Try direct URL approach
            page.goto("https://www.google.com/maps/search/Aktion+Air+Philadelphia+PA", timeout=30000)
            page.wait_for_timeout(5000)
            # Click first result
            links = page.query_selector_all('a[href*="/maps/place/"]')
            if links:
                links[0].click()
                page.wait_for_timeout(5000)
                try:
                    title = page.locator("h1").first.inner_text(timeout=5000)
                    log.info(f"After direct search, h1 title: '{title}'")
                except Exception:
                    pass

        # Dismiss any consent dialog
        try:
            consent = page.query_selector('button[aria-label*="Accept"], button[aria-label*="Reject"], form[action*="consent"] button')
            if consent:
                consent.click()
                page.wait_for_timeout(2000)
                log.info("Dismissed consent dialog")
        except Exception:
            pass

        # Wait for panel to settle
        page.wait_for_timeout(3000)

        # Debug: log page body preview
        try:
            body_text = page.evaluate("() => document.body.innerText.substring(0, 500)")
            log.info(f"Page body preview: {body_text[:300]}")
        except Exception:
            pass

        # Open reviews tab - try multiple approaches
        tab_opened = False

        # Approach 1: Click tab buttons
        try:
            tabs = page.query_selector_all('button[role="tab"]')
            log.info(f"Found {len(tabs)} tabs")
            for tab in tabs:
                label = (tab.get_attribute("aria-label") or tab.inner_text() or "").lower()
                if "review" in label:
                    tab.click()
                    page.wait_for_timeout(4000)
                    tab_opened = True
                    log.info(f"Clicked reviews tab: {label}")
                    break
        except Exception:
            pass

        # Approach 2: Click the star rating / review count link
        if not tab_opened:
            try:
                rating_btns = page.query_selector_all('button, [role="button"], a, span[role="img"]')
                for btn in rating_btns:
                    label = (btn.get_attribute("aria-label") or btn.inner_text() or "").lower()
                    if ("review" in label and any(c.isdigit() for c in label)):
                        btn.click()
                        page.wait_for_timeout(4000)
                        tab_opened = True
                        log.info(f"Clicked review count: {label}")
                        break
            except Exception:
                pass

        # Approach 3: Scroll panel down to reveal reviews section
        if not tab_opened:
            log.info("No reviews tab found, scrolling down to find reviews...")
            for scroll_pass in range(3):
                try:
                    scrolled = page.evaluate("""
                        () => {
                            const sels = [
                                'div.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde',
                                'div.m6QErb.DxyBCb.kA9KKe.dS8AEf',
                                '.m6QErb', '.DxyBCb',
                                'div[role="main"]',
                            ];
                            for (const sel of sels) {
                                const el = document.querySelector(sel);
                                if (el && el.scrollHeight > el.clientHeight) {
                                    el.scrollTop += 2000;
                                    return sel;
                                }
                            }
                            return null;
                        }
                    """)
                    log.info(f"Scroll pass {scroll_pass+1}: scrolled '{scrolled}'")
                    page.wait_for_timeout(2000)
                except Exception as e:
                    log.warning(f"Scroll failed: {e}")

            # Check for tabs after scrolling
            try:
                tabs = page.query_selector_all('button[role="tab"]')
                log.info(f"After scroll: found {len(tabs)} tabs")
                for tab in tabs:
                    label = (tab.get_attribute("aria-label") or tab.inner_text() or "").lower()
                    if "review" in label:
                        tab.click()
                        page.wait_for_timeout(4000)
                        tab_opened = True
                        log.info(f"Clicked reviews tab after scroll: {label}")
                        break
            except Exception:
                pass

        log.info(f"Reviews tab opened: {tab_opened}")

        # Diagnostic: count elements before extraction
        diag = page.evaluate("""
            () => {
                const containers = document.querySelectorAll('div[data-review-id]').length;
                const wiI7pd = document.querySelectorAll('.wiI7pd').length;
                const texts = [];
                for (const el of document.querySelectorAll('.wiI7pd')) {
                    const t = (el.innerText || el.textContent || '').trim();
                    if (t && t.length > 15) texts.push(t.substring(0, 100));
                    if (texts.length >= 10) break;
                }
                return { containers, wiI7pd, texts };
            }
        """)
        log.info(f"Before extraction: containers={diag['containers']}, wiI7pd={diag['wiI7pd']}")
        for i, t in enumerate(diag['texts']):
            log.info(f"  .wiI7pd[{i}]: {t}")

        # Run the EXACT extraction JS from app.py
        EXTRACT_JS = r"""
            async (params) => {
                let { needed, maxMs } = params;
                const start = performance.now();
                const sleep = ms => new Promise(r => setTimeout(r, ms));
                const orderedKeys = [];
                const textByKey = new Map();

                const upsertText = (key, text) => {
                    if (!textByKey.has(key)) orderedKeys.push(key);
                    textByKey.set(key, text);
                };

                const snapshot = () => {
                    const out = [];
                    for (const k of orderedKeys) {
                        const t = (textByKey.get(k) || '').trim();
                        if (!t) continue;
                        out.push(t);
                        if (out.length >= needed) break;
                    }
                    return out;
                };

                const isOwnerReply = (node) => {
                    let curr = node;
                    while (curr) {
                        const cls = (curr.className || '');
                        if (typeof cls === 'string' && /\bCDe7pd\b/.test(cls)) return true;
                        const aria = curr.getAttribute && curr.getAttribute('aria-label');
                        if (aria && /response from/i.test(aria)) return true;
                        curr = curr.parentElement;
                    }
                    return false;
                };

                const cleanText = (t) => {
                    return (t || '').replace(/[\uE000-\uF8FF]/g, ' ')
                        .replace(/\s+/g, ' ').trim()
                        .replace(/(?:\.\.\.\s*more|…\s*more|more)\s*$/i, '').trim()
                        .replace(/\bresponse\s+from\s+the\s+owner\b[\s\S]*$/i, '').trim()
                        .replace(/\(owner\)\s*[\s\S]*$/i, '').trim()
                        .replace(/\s*\+?\d*\s*Like\s+Share\s*$/i, '').trim()
                        .replace(/\s*Hover\s+to\s+react\s*$/i, '').trim();
                };

                const expandMore = async () => {
                    let clicked = 0;
                    for (const c of document.querySelectorAll('div[data-review-id]')) {
                        for (const b of c.querySelectorAll('button, span, [role="button"]')) {
                            // Skip elements inside anchor tags (would open reviewer profile)
                            let anc = b, insideAnchor = false;
                            while (anc && anc !== c) {
                                if (anc.tagName === 'A') { insideAnchor = true; break; }
                                anc = anc.parentElement;
                            }
                            if (insideAnchor) continue;
                            const t = ((b.getAttribute('aria-label') || b.innerText || '') || '').trim().toLowerCase();
                            if (t && t.length <= 60 && (t === 'more' || t.endsWith(' more') || t.includes('... more'))) {
                                try { b.click(); clicked++; } catch(e) {}
                                if (clicked >= 48) break;
                            }
                        }
                    }
                    if (clicked > 0) await sleep(200);
                };

                const collect = () => {
                    let idx = 0;
                    for (const c of document.querySelectorAll('div[data-review-id]')) {
                        const nameEl = c.querySelector('.d4r55, .kxklSb');
                        if (nameEl && /\bowner\b/i.test(nameEl.innerText || '')) continue;

                        let best = '';
                        for (const sel of ['.wiI7pd', '[data-expandable-section]', '.MyEned']) {
                            for (const node of c.querySelectorAll(sel)) {
                                if (isOwnerReply(node)) continue;
                                const text = cleanText(node.innerText || node.textContent || '');
                                if (text.length > best.length) best = text;
                            }
                        }
                        if (best && best.length >= 20) {
                            let key = c.getAttribute('data-review-id');
                            if (!key) key = `idx:${idx}:${best.substring(0, 80)}`;
                            idx++;
                            upsertText(key, best);
                            if (orderedKeys.length >= needed) break;
                        }
                    }
                    return snapshot();
                };

                // Warmup
                await expandMore();
                for (let w = 0; w < 3 && snapshot().length < needed; w++) {
                    await expandMore();
                    collect();
                }

                let texts = snapshot();
                if (texts.length >= needed) return texts.slice(0, needed);

                // Per-container scrollIntoView with generous waits
                const containers = document.querySelectorAll('div[data-review-id]');
                for (let ci = 0; ci < containers.length && snapshot().length < needed && performance.now() - start < maxMs; ci++) {
                    try {
                        containers[ci].scrollIntoView({ behavior: 'instant', block: 'center' });
                    } catch (e) {}
                    await sleep(350);
                    await expandMore();
                    collect();
                    if (snapshot().length <= ci) {
                        await sleep(300);
                        collect();
                    }
                }

                texts = snapshot();
                if (texts.length >= needed) return texts.slice(0, needed);

                // Second-chance: re-scroll with longer waits
                const containers2 = document.querySelectorAll('div[data-review-id]');
                for (let ci = 0; ci < containers2.length && snapshot().length < needed && performance.now() - start < maxMs; ci++) {
                    try {
                        containers2[ci].scrollIntoView({ behavior: 'instant', block: 'center' });
                    } catch (e) {}
                    await sleep(500);
                    await expandMore();
                    collect();
                }

                return snapshot().slice(0, needed);
            }
        """

        t0 = time.time()
        raw = page.evaluate(EXTRACT_JS, {"needed": TARGET + 4, "maxMs": 15000.0}) or []
        t1 = time.time()
        log.info(f"Extraction returned {len(raw)} raw texts in {t1-t0:.1f}s")

        final = sanitize_review_snippets(list(raw), max_items=TARGET)
        log.info(f"After sanitization: {len(final)} reviews")
        for i, t in enumerate(final):
            log.info(f"  Review[{i+1}]: {t[:120]}...")

        browser.close()

        passed = len(final) >= TARGET
        print(f"\n{'='*60}")
        print(f"Aktion Air: {'PASS' if passed else 'FAIL'} ({len(final)}/{TARGET} reviews)")
        print(f"{'='*60}")
        return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(run())
