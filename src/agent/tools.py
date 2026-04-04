"""External tool integrations."""

from __future__ import annotations

from collections.abc import Mapping
import re


def fetch_live_page(
    url: str,
    *,
    selector: str = "body",
    timeout_ms: int = 20_000,
    max_chars: int = 12_000,
    headless: bool = True,
) -> str:
    """Fetch page text via Playwright and return a citation-friendly snapshot.

    Args:
        url: HTTP(S) URL to open.
        selector: CSS selector to extract text from. Defaults to ``body``.
        timeout_ms: Navigation and selector wait timeout in milliseconds.
        max_chars: Maximum number of returned text characters.
        headless: Whether to run Chromium in headless mode.
    """

    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        raise ValueError("URL must start with http:// or https://")

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "Playwright is required. Install with `pip install playwright` and run `playwright install chromium`."
        ) from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        try:
            page = browser.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_selector(selector, timeout=timeout_ms)
            except PlaywrightTimeoutError as exc:
                raise TimeoutError(f"Timed out while loading page: {url}") from exc

            title = page.title().strip()
            text = page.inner_text(selector)
            normalized = re.sub(r"\s+", " ", text).strip()
            if max_chars > 0:
                normalized = normalized[:max_chars]
            return f"Title: {title}\nURL: {url}\n\n{normalized}".strip()
        finally:
            browser.close()


def fetch_hkbu_updates(
    *,
    timetable_url: str,
    news_url: str,
    timeout_ms: int = 20_000,
) -> Mapping[str, str]:
    """Fetch live HKBU timetable and news pages in a single call."""

    return {
        "timetable": fetch_live_page(timetable_url, timeout_ms=timeout_ms),
        "news": fetch_live_page(news_url, timeout_ms=timeout_ms),
    }

