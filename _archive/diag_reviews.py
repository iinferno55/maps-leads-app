#!/usr/bin/env python3
"""Quick diagnostic: Scrape one listing's reviews and show raw vs cleaned."""
import logging, os, sys, json, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from playwright.sync_api import sync_playwright

# Use the exact same JS from app.py
COLLECT_JS = r"""
async (params) => {
    const { needed, maxMs } = params;
    const start = performance.now();
    const orderedKeys = [];
    const textByKey = new Map();
    const sleep = ms => new Promise(r => setTimeout(r, ms));

    const upsertText = (key, text) => {
        const k = String(key || '').trim();
        const t = String(text || '').trim();
        if (!k || !t) return;
        if (!textByKey.has(k)) {
            orderedKeys.push(k);
            textByKey.set(k, t);
            return;
        }
        const prev = textByKey.get(k) || '';
        const next = t;
        if (!prev || next.length >= prev.length + 12 ||
            (/(?:\.\.\.|…)\s*$/.test(prev) && next.length > prev.length)) {
            textByKey.set(k, t);
        }
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

    const reviewContainerSelector = 'div[data-review-id], div.jftiEf, div.GHT2ce, div[class*="jftiEf"], div[class*="GHT2ce"]';
    const primaryReviewBodySelectors = ['.wiI7pd', '[data-expandable-section]', '.MyEned'];
    const secondaryReviewBodySelectors = ['.fontBodyMedium', 'span[dir="auto"]', 'div[dir="auto"]'];

    const cleanReviewText = (value) => {
        let text = String(value || '').replace(/[\uE000-\uF8FF]/g, ' ').replace(/\s+/g, ' ').trim();
        if (!text) return '';
        text = text.replace(/(?:\.\.\.?\s*more|…\s*more|more)\s*$/i, '').trim();
        text = text.replace(/\bresponse\s+from\s+the\s+owner\b[\s\S]*$/i, '').trim();
        text = text.replace(/\s*(?:Price|Service|Quality)\s+assessment\b[\s\S]*$/i, '').trim();
        return text;
    };

    const isLikelyReviewerMeta = (value) => {
        const text = cleanReviewText(value);
        if (!text) return true;
        if (/(local guide|\b\d+\s+reviews?\b|\b\d+\s+photos?\b|\b\d+\s+videos?\b)/i.test(text)
            && !/[.!?]/.test(text) && text.length <= 120) return true;
        if (/^\d+\s+(day|days|week|weeks|month|months|year|years)\s+ago$/i.test(text)) return true;
        return false;
    };

    const isLikelyBusinessCard = (value) => {
        const text = cleanReviewText(value);
        if (!text) return true;
        if (/\b[0-5]\.\d\s*\(\d{1,3}(?:,\d{3})*\)/.test(text)) return true;
        if (/\(\d{3}\)\s*\d{3}-\d{4}/.test(text)) return true;
        if (/\b(open|closed)\b.*\b\d{1,2}\s?(am|pm)\b/.test(text.toLowerCase())) return true;
        return false;
    };

    const isLikelyUiNoise = (value) => {
        const text = cleanReviewText(value).toLowerCase();
        if (!text) return true;
        if (text.length < 20) return true;
        if (/^photo of reviewer who wrote\b/.test(text)) return true;
        if (/^actions for .+ review$/.test(text)) return true;
        if (/^\d+\s+reviews?$/.test(text)) return true;
        if (/\bmentioned\s+in\s+\d+\s+reviews?\b/.test(text)) return true;
        if (/^(reviews?|all reviews|write a review|sort|newest|highest|lowest|most relevant|search reviews?|google maps|directions|website|call|share)$/.test(text)) return true;
        if (/^(open|closed)\b/.test(text) && text.length < 40) return true;
        if (isLikelyReviewerMeta(text) || isLikelyBusinessCard(text)) return true;
        return false;
    };

    const reviewContainers = () => {
        let containers = Array.from(document.querySelectorAll('div[data-review-id]'));
        if (containers.length === 0) {
            const raw = Array.from(document.querySelectorAll('div.jftiEf, div.GHT2ce, div[class*="jftiEf"], div[class*="GHT2ce"]'));
            const filtered = [];
            for (const el of raw) {
                let isNested = false;
                for (const other of raw) {
                    if (other !== el && other.contains(el)) { isNested = true; break; }
                }
                if (!isNested) filtered.push(el);
            }
            containers = filtered;
        }
        return containers;
    };

    const findScrollEl = () => {
        const sels = [
            'div.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde',
            'div[role="feed"]', '.m6QErb.DxyBCb.kA9KKe.dS8AEf',
            '.WNBkOb', '.XiKgde', '.m6QErb[role="feed"]',
            'div.section-layout.section-scrollbox'
        ];
        for (const sel of sels) {
            const el = document.querySelector(sel);
            if (el && el.scrollHeight > el.clientHeight) return el;
        }
        const reviews = reviewContainers();
        if (reviews.length > 0) {
            let curr = reviews[0].parentElement;
            while (curr && curr !== document.body) {
                const style = window.getComputedStyle(curr);
                if (style.overflowY === 'auto' || style.overflowY === 'scroll') return curr;
                curr = curr.parentElement;
            }
        }
        return document.scrollingElement || document.documentElement || document.body;
    };

    const scrollEl = findScrollEl();
    const scrollOnce = () => {
        if (!scrollEl) return;
        scrollEl.scrollTop = Math.min(
            (scrollEl.scrollHeight || 0) - (scrollEl.clientHeight || 0),
            (scrollEl.scrollTop || 0) + Math.max(300, (scrollEl.clientHeight || 600) * 0.7)
        );
    };

    const expandMore = async () => {
        let clicked = 0;
        for (const c of reviewContainers()) {
            for (const b of c.querySelectorAll('button, span, div, a, [role="button"]')) {
                const t = (((b.getAttribute && (b.getAttribute('aria-label') || b.getAttribute('data-tooltip'))) || b.innerText || b.textContent) || '').trim().toLowerCase();
                if (!t || !t.includes('more') || t.length > 60) continue;
                if (t === 'more' || t === 'see more' || t === 'read more' || t.endsWith(' more') || t.includes('... more') || t.includes('\u2026 more')) {
                    try { b.click(); clicked += 1; } catch (e) {}
                    if (clicked >= 48) return;
                }
            }
        }
        if (clicked > 0) await sleep(150);
    };

    const extractBestText = (container) => {
        for (const sels of [primaryReviewBodySelectors, secondaryReviewBodySelectors]) {
            let best = '';
            for (const sel of sels) {
                for (const node of container.querySelectorAll(sel)) {
                    const text = cleanReviewText(node.innerText || node.textContent || '');
                    if (!isLikelyUiNoise(text) && text.length > best.length) best = text;
                }
            }
            if (best) return best;
        }
        return '';
    };

    const collect = () => {
        let idx = 0;
        for (const c of reviewContainers()) {
            const nameEl = c.querySelector('.d4r55, .kxklSb');
            if (nameEl && (nameEl.innerText || '').toLowerCase().includes('owner')) continue;
            let text = extractBestText(c);
            if (!text) continue;
            let key = c.getAttribute('data-review-id') || `idx:${idx}:${text.substring(0, 160)}`;
            idx += 1;
            upsertText(key, text);
        }
        return snapshot();
    };

    await expandMore();
    for (let warm = 0; warm < 3 && snapshot().length < needed; warm++) {
        await expandMore(); collect();
    }
    let texts = snapshot();
    if (texts.length >= needed) return texts.slice(0, needed);

    while (performance.now() - start < maxMs && texts.length < needed) {
        scrollOnce();
        await expandMore();
        collect();
        texts = snapshot();
        if (texts.length >= needed) break;
    }
    return snapshot().slice(0, needed);
}
"""

SCROLL_JS = r"""
() => {
    const reviewContainerSelector = 'div[data-review-id], div.jftiEf, div.GHT2ce, div[class*="jftiEf"], div[class*="GHT2ce"]';
    const sels = [
        'div.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde',
        'div[role="feed"]', '.m6QErb.DxyBCb.kA9KKe.dS8AEf',
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
    const reviews = Array.from(document.querySelectorAll(reviewContainerSelector));
    if (reviews.length > 0) {
        let curr = reviews[0].parentElement;
        while (curr && curr !== document.body) {
            const style = window.getComputedStyle(curr);
            if (style.overflowY === 'auto' || style.overflowY === 'scroll') {
                curr.scrollTop += Math.max(400, (curr.clientHeight || 700) * 0.85);
                return true;
            }
            curr = curr.parentElement;
        }
    }
    return false;
}
"""

OPEN_REVIEWS_JS = r"""
() => {
    const candidates = Array.from(document.querySelectorAll('button,[role="button"],a,[role="tab"]'));
    const scored = [];
    for (const el of candidates) {
        const label = ((el.getAttribute && el.getAttribute('aria-label')) || el.textContent || '').trim();
        if (!label) continue;
        if (!/reviews?/i.test(label)) continue;
        if (/write/i.test(label)) continue;
        const m = label.match(/(\d{1,3}(?:,\d{3})*)/);
        const count = m ? parseInt(m[1].replace(/,/g,''), 10) : 0;
        scored.push({ el, count, label: label.slice(0, 120) });
    }
    scored.sort((a,b) => (b.count||0) - (a.count||0));
    const pick = scored[0] && scored[0].el;
    if (pick) { try { pick.scrollIntoView({block:'center'}); } catch(e) {} pick.click(); return scored[0].label; }
    return null;
}
"""

def main():
    log = logging.getLogger("diag")
    # Import the app.py functions for Python-side cleaning
    from app import clean_extracted_review_snippet, sanitize_review_snippets

    url = "https://www.google.com/maps/search/power+washer+san+antonio"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        page.wait_for_timeout(3000)

        # Click first result
        first = page.query_selector('a[href*="/maps/place/"]')
        if first:
            first.click()
            page.wait_for_timeout(3000)

        h1 = page.query_selector("h1.DUwDvf, h1.fontHeadlineLarge")
        biz_name = h1.inner_text() if h1 else "unknown"
        log.info(f"Business: {biz_name}")

        # Open reviews tab
        tab = page.evaluate(OPEN_REVIEWS_JS)
        log.info(f"Reviews tab: {tab}")
        page.wait_for_timeout(2000)

        # Pre-scroll
        for i in range(8):
            page.evaluate(SCROLL_JS)
            page.wait_for_timeout(380)

        # Collect RAW reviews from JS
        raw_reviews = page.evaluate(COLLECT_JS, {"needed": 6, "maxMs": 4000.0})
        log.info(f"JS returned {len(raw_reviews)} raw reviews")
        for i, r in enumerate(raw_reviews):
            log.info(f"  RAW[{i}] ({len(r)} chars): {repr(r)}")

        # Now run through Python cleaning
        log.info("--- Python-side cleaning ---")
        cleaned = sanitize_review_snippets(raw_reviews, max_items=6)
        log.info(f"After sanitize: {len(cleaned)} reviews")
        for i, c in enumerate(cleaned):
            log.info(f"  CLEAN[{i}] ({len(c)} chars): {c[:200]}")

        # Show what was dropped
        log.info("--- Per-snippet diagnostic ---")
        for i, raw in enumerate(raw_reviews):
            c = clean_extracted_review_snippet(raw)
            if c:
                log.info(f"  [{i}] KEPT ({len(c)} chars): {c[:120]}")
            else:
                log.info(f"  [{i}] DROPPED raw='{raw[:200]}'")

        ctx.close()
        browser.close()

if __name__ == "__main__":
    main()
