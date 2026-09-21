#!/usr/bin/env python3
"""Playwright: open site/index.html from disk, assert the map, the topic filter and the search work,
and write docs/preview.png (desktop) and docs/preview-mobile.png. Needs `pip install playwright`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "site" / "index.html").as_uri()


def check(page, name: str) -> None:
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(PAGE)
    page.wait_for_selector(".tile")
    assert page.locator(".tile").count() == 24, page.locator(".tile").count()
    total = page.locator("#scope .count").inner_text()
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    assert overflow <= 0, f"{name}: page overflows sideways by {overflow}px"
    # tiles must stay inside the map and their labels inside the tiles
    clipped = page.evaluate("""() => [...document.querySelectorAll('.tile > span')].filter(s => s.scrollHeight > s.clientHeight + 1 || s.scrollWidth > s.clientWidth + 1).map(s => s.innerText.split('\\n')[0])""")
    assert not clipped, f"{name}: tile text clipped: {clipped}"

    page.locator('.tile[data-id="robotics"]').click()
    assert "Robotics" in page.locator("#scope h2").inner_text()
    n = int(page.locator('.tile[data-id="robotics"] .n').inner_text().split()[0])
    assert page.locator("#scope .count").inner_text().startswith(f"{n} paper"), page.locator("#scope .count").inner_text()
    assert page.locator("#list details").count() == min(n, 60)
    page.locator("#reset").click()
    assert page.locator("#scope .count").inner_text() == total

    page.fill("#q", "diffusion")
    hits = int(page.locator("#scope .count").inner_text().split()[0].replace(",", ""))
    assert 0 < hits < 1000, hits
    assert "diffusion" in page.locator("#list details").first.inner_text().lower() or page.locator("#list details").first.evaluate("d => d.querySelector('.abs').textContent.toLowerCase().includes('diffusion')")
    page.fill("#q", "")
    assert not errors, errors
    print(f"{name}: 24 tiles, topic filter ok ({n} robotics papers), search ok ({hits} hits for 'diffusion'), no overflow, no console errors")


def main() -> int:
    (ROOT / "docs").mkdir(exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        desktop = browser.new_page(viewport={"width": 1600, "height": 1040})
        check(desktop, "desktop")
        desktop.goto(PAGE)
        desktop.wait_for_selector(".tile")
        desktop.screenshot(path=str(ROOT / "docs" / "preview.png"))
        mobile = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True).new_page()
        check(mobile, "mobile")
        mobile.goto(PAGE)
        mobile.wait_for_selector(".tile")
        mobile.screenshot(path=str(ROOT / "docs" / "preview-mobile.png"), full_page=False)
        browser.close()
    print("wrote docs/preview.png and docs/preview-mobile.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
