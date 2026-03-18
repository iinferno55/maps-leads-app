from playwright.sync_api import sync_playwright


def main() -> None:
    base = "http://localhost:8502"
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        c = b.new_context(viewport={"width": 1600, "height": 980})
        pg = c.new_page()
        pg.goto(base, wait_until="domcontentloaded", timeout=60_000)
        pg.wait_for_function(
            "(t) => !!document.body && (document.body.innerText || '').includes(String(t))",
            arg="Cold Calling Leads",
            timeout=60_000,
        )
        print("wait_ok")
        c.close()
        b.close()


if __name__ == "__main__":
    main()

