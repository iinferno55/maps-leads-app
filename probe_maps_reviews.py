from playwright.sync_api import sync_playwright


URL = "https://www.google.com/maps/place/Valley+Roofing+and+Repair/data=!4m7!3m6!1s0x520d13b22af25c9:0xd4eaf160b1874ece!8m2!3d33.493635!4d-112.1176613!16s%2Fg%2F11wvsxk6_3!19sChIJySWvIjvRIAURzk6HsWDx6tQ?authuser=0&hl=en&rclk=1"


def safe_printable(value: str) -> str:
    return value.encode("cp1252", errors="backslashreplace").decode("cp1252", errors="ignore")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="en-US",
        )
        page = ctx.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=30000, referer="https://www.google.com/maps")
        page.wait_for_timeout(4000)

        body_text = page.evaluate("() => (document.body && (document.body.innerText || '')) || ''") or ""
        low = body_text.lower()
        print("body_len:", len(body_text))
        print("contains 'review':", ("review" in low))
        idx = low.find("review")
        if idx >= 0:
            sample = body_text[max(0, idx - 100) : idx + 150].replace("\n", " ")
            print("sample:", safe_printable(sample))

        # Gather candidate clickable elements mentioning reviews.
        candidates = page.evaluate(
            """
            () => {
              const out = [];
              const els = Array.from(document.querySelectorAll('button,[role="button"],a'));
              for (const el of els) {
                const label = ((el.getAttribute && el.getAttribute('aria-label')) || el.textContent || '').trim();
                if (!label) continue;
                if (/review/i.test(label) || /\\b\\d+[\\s,]*\\+?\\s*reviews?\\b/i.test(label)) {
                  out.push(label.slice(0, 140));
                }
                if (out.length >= 40) break;
              }
              return out;
            }
            """
        ) or []
        print(
            "clickable candidates:",
            [safe_printable(c) for c in candidates[:25]],
        )

        # Also search all short visible texts containing 'review'.
        short_texts = page.evaluate(
            """
            () => {
              const out = [];
              const els = Array.from(document.querySelectorAll('div,span,button,a'));
              for (const el of els) {
                const t = (el.innerText || el.textContent || '').trim();
                if (!t) continue;
                if (/review/i.test(t) && t.length < 80) out.push(t);
                if (out.length >= 40) break;
              }
              return out;
            }
            """
        ) or []
        print(
            "short texts:",
            [safe_printable(t) for t in short_texts[:25]],
        )
        browser.close()


if __name__ == "__main__":
    main()

