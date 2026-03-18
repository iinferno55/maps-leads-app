from playwright.sync_api import sync_playwright


def main() -> None:
    base = "http://localhost:8502"
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        c = b.new_context(viewport={"width": 1600, "height": 980})
        pg = c.new_page()
        pg.goto(base, wait_until="domcontentloaded", timeout=60_000)
        pg.wait_for_timeout(2000)

        btns = pg.locator("button")
        print("buttons_count", btns.count())
        for i in range(btns.count()):
            al = btns.nth(i).get_attribute("aria-label") or ""
            tx = (btns.nth(i).inner_text() or "").strip()
            if any(k in (al + " " + tx).lower() for k in ("sidebar", "menu", "navigation")):
                print("button", i, "aria=", al, "text=", tx[:60])

        sb = pg.locator('section[data-testid="stSidebar"]')
        print("sidebar_count", sb.count())
        if sb.count():
            print("sidebar_visible", sb.is_visible())
            print("sidebar_box", sb.bounding_box())

        cb = pg.locator('input[type="checkbox"][aria-label^="Debug mode"]')
        print("debug_checkbox_count", cb.count())
        if cb.count():
            print("debug_checkbox_visible", cb.first.is_visible())
            print("debug_checkbox_box", cb.first.bounding_box())

        c.close()
        b.close()


if __name__ == "__main__":
    main()

