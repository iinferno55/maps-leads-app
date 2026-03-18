#!/usr/bin/env python3
"""Multi-round diagnostic test: verifies 6 reviews per listing across multiple businesses."""
import logging, os, sys, json, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("multi_diag")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from playwright.sync_api import sync_playwright
from app import clean_extracted_review_snippet, sanitize_review_snippets

BUSINESSES = [
    {"name": "San Antonio Pressure Washing", "url": "https://www.google.com/maps/search/power+washer+san+antonio"},
    {"name": "All-Star Power Wash", "url": "https://www.google.com/maps/search/power+washer+san+diego"},
]
NEEDED = 6
ROUNDS = 5

COLLECT_JS = r"""
async (params) => {
    const { needed, maxMs } = params;
    const start = performance.now();
    const orderedKeys = [];
    const textByKey = new Map();
    const sleep = ms => new Promise(r => setTimeout(r, ms));

    const shouldReplace = (o, n) => {
        const p = String(o || '').trim(), q = String(n || '').trim();
        if (!p) return true; if (!q) return false;
        if (q.length >= p.length + 12) return true;
        if (/(?:\.\.\.|…)\s*$/.test(p) && q.length > p.length) return true;
        return false;
    };

    const upsertText = (key, text) => {
        const k = String(key || '').trim(), t = String(text || '').trim();
        if (!k || !t) return;
        if (!textByKey.has(k)) { orderedKeys.push(k); textByKey.set(k, t); return; }
        const prev = textByKey.get(k) || '';
        if (shouldReplace(prev, t)) textByKey.set(k, t);
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

    const cleanReviewText = (value) => {
        let text = String(value || '').replace(/[\uE000-\uF8FF]/g, ' ').replace(/\s+/g, ' ').trim();
        if (!text) return '';
        text = text.replace(/(?:\.\.\.?\s*more|…\s*more|more)\s*$/i, '').trim();
        text = text.replace(/\bresponse\s+from\s+the\s+owner\b[\s\S]*$/i, '').trim();
        text = text.replace(/\s*(?:Price|Service|Quality)\s+assessment\b[\s\S]*$/i, '').trim();
        return text;
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
        if (/(local guide|\b\d+\s+reviews?\b|\b\d+\s+photos?\b)/i.test(text) && !/[.!?]/.test(text) && text.length <= 120) return true;
        if (/^\d+\s+(day|days|week|weeks|month|months|year|years)\s+ago$/i.test(text)) return true;
        if (/\b[0-5]\.\d\s*\(\d{1,3}(?:,\d{3})*\)/.test(text)) return true;
        if (/\(\d{3}\)\s*\d{3}-\d{4}/.test(text)) return true;
        if (/\b(open|closed)\b.*\b\d{1,2}\s?(am|pm)\b/.test(text)) return true;
        if (/\b(car detailing service|flooring store|contractor|plumber|electrician|hvac|roofing contractor|garage door supplier|pressure washing service)\b/.test(text) && text.length <= 200) return true;
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
        const sels = ['div.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde', 'div[role="feed"]', '.m6QErb.DxyBCb.kA9KKe.dS8AEf', '.WNBkOb', '.XiKgde', '.m6QErb[role="feed"]', 'div.section-layout.section-scrollbox'];
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
        const primarySels = ['.wiI7pd', '[data-expandable-section]', '.MyEned'];
        const secondarySels = ['.fontBodyMedium', 'span[dir="auto"]', 'div[dir="auto"]'];
        for (const sels of [primarySels, secondarySels]) {
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
    const sels = ['div.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde', 'div[role="feed"]', '.m6QErb.DxyBCb.kA9KKe.dS8AEf', '.WNBkOb', '.XiKgde', '.m6QErb[role="feed"]', 'div.section-layout.section-scrollbox'];
    for (const sel of sels) {
        const el = document.querySelector(sel);
        if (el && el.scrollHeight > el.clientHeight) { el.scrollTop += Math.max(400, (el.clientHeight || 700) * 0.85); return true; }
    }
    const reviews = Array.from(document.querySelectorAll('div[data-review-id]'));
    if (reviews.length > 0) {
        let curr = reviews[0].parentElement;
        while (curr && curr !== document.body) {
            const style = window.getComputedStyle(curr);
            if (style.overflowY === 'auto' || style.overflowY === 'scroll') { curr.scrollTop += Math.max(400, (curr.clientHeight || 700) * 0.85); return true; }
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
        if (!label || !/reviews?/i.test(label) || /write/i.test(label)) continue;
        const m = label.match(/(\d{1,3}(?:,\d{3})*)/);
        scored.push({ el, count: m ? parseInt(m[1].replace(/,/g,''), 10) : 0, label: label.slice(0, 120) });
    }
    scored.sort((a,b) => (b.count||0) - (a.count||0));
    if (scored[0]) { try { scored[0].el.scrollIntoView({block:'center'}); } catch(e) {} scored[0].el.click(); return scored[0].label; }
    return null;
}
"""

def test_business(page, biz_url, biz_name, round_num):
    page.goto(biz_url, wait_until="domcontentloaded", timeout=20000)
    page.wait_for_timeout(3000)

    first = page.query_selector('a[href*="/maps/place/"]')
    if first:
        first.click()
        page.wait_for_timeout(3000)

    h1 = page.query_selector("h1.DUwDvf, h1.fontHeadlineLarge")
    actual_name = h1.inner_text() if h1 else "unknown"

    tab = page.evaluate(OPEN_REVIEWS_JS)
    page.wait_for_timeout(2000)

    for i in range(8):
        page.evaluate(SCROLL_JS)
        page.wait_for_timeout(380)

    raw_reviews = page.evaluate(COLLECT_JS, {"needed": NEEDED, "maxMs": 4000.0})
    cleaned = sanitize_review_snippets(raw_reviews, max_items=NEEDED)

    log.info(f"  Round {round_num} | {actual_name}: JS={len(raw_reviews)} raw, final={len(cleaned)} cleaned")
    return len(cleaned)

def main():
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )
        page = ctx.new_page()

        for biz in BUSINESSES:
            name = biz["name"]
            results[name] = []
            for rd in range(1, ROUNDS + 1):
                count = test_business(page, biz["url"], name, rd)
                results[name].append(count)

        ctx.close()
        browser.close()

    log.info("=" * 50)
    log.info("RESULTS")
    log.info("=" * 50)
    total_pass = 0
    total_tests = 0
    for name, counts in results.items():
        passes = sum(1 for c in counts if c >= NEEDED)
        total_pass += passes
        total_tests += len(counts)
        log.info(f"  {name}: {passes}/{len(counts)} rounds with {NEEDED}+ reviews")
        for i, c in enumerate(counts):
            status = "PASS" if c >= NEEDED else "FAIL"
            log.info(f"    Round {i+1}: {status} ({c} reviews)")

    log.info(f"\nOverall: {total_pass}/{total_tests}")
    if total_pass < total_tests:
        log.error(f"FAILED: {total_tests - total_pass} tests failed")
        sys.exit(1)
    else:
        log.info("ALL PASSED!")

if __name__ == "__main__":
    main()
