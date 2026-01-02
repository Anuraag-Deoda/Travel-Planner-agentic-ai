"""Playwright browser lifecycle management."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

from src.config.constants import (
    BROWSER_TIMEOUT_MS,
    DEFAULT_VIEWPORT,
    PAGE_LOAD_WAIT,
    USER_AGENT,
)


class BrowserManager:
    """Manages Playwright browser lifecycle.

    Provides singleton browser instance and context management for
    efficient resource usage across multiple browsing operations.
    """

    _playwright: Optional[Playwright] = None
    _browser: Optional[Browser] = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_browser(cls) -> Browser:
        """Get or create the browser instance.

        Thread-safe singleton pattern for browser creation.

        Returns:
            Playwright Browser instance.
        """
        async with cls._lock:
            if cls._browser is None or not cls._browser.is_connected():
                cls._playwright = await async_playwright().start()
                cls._browser = await cls._playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                    ],
                )
            return cls._browser

    @classmethod
    @asynccontextmanager
    async def get_context(cls) -> AsyncGenerator[BrowserContext, None]:
        """Get a new browser context.

        Creates a fresh context for isolation between browsing sessions.
        Automatically closes when done.

        Yields:
            BrowserContext for making page requests.
        """
        browser = await cls.get_browser()
        context = await browser.new_context(
            viewport=DEFAULT_VIEWPORT,
            user_agent=USER_AGENT,
            java_script_enabled=True,
            # Prevent some bot detection
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            yield context
        finally:
            await context.close()

    @classmethod
    @asynccontextmanager
    async def get_page(cls) -> AsyncGenerator[Page, None]:
        """Get a new page within a context.

        Convenience method for single-page operations.
        Creates context, gets page, and cleans up.

        Yields:
            Page for browsing operations.
        """
        async with cls.get_context() as context:
            page = await context.new_page()
            page.set_default_timeout(BROWSER_TIMEOUT_MS)
            try:
                yield page
            finally:
                await page.close()

    @classmethod
    async def shutdown(cls) -> None:
        """Shutdown the browser and Playwright.

        Should be called when the application exits.
        """
        async with cls._lock:
            if cls._browser is not None:
                await cls._browser.close()
                cls._browser = None
            if cls._playwright is not None:
                await cls._playwright.stop()
                cls._playwright = None


async def navigate_and_wait(
    page: Page,
    url: str,
    wait_until: str = PAGE_LOAD_WAIT,
    timeout: int = BROWSER_TIMEOUT_MS,
) -> bool:
    """Navigate to a URL and wait for load.

    Args:
        page: Playwright Page instance.
        url: URL to navigate to.
        wait_until: Load state to wait for (load, domcontentloaded, networkidle).
        timeout: Maximum wait time in milliseconds.

    Returns:
        True if navigation succeeded, False otherwise.
    """
    try:
        response = await page.goto(url, wait_until=wait_until, timeout=timeout)
        return response is not None and response.ok
    except Exception:
        return False


async def extract_text_content(
    page: Page,
    selector: str | None = None,
    max_length: int = 10000,
) -> str:
    """Extract text content from a page.

    Args:
        page: Playwright Page instance.
        selector: Optional CSS selector to limit extraction.
        max_length: Maximum characters to return.

    Returns:
        Extracted text content.
    """
    try:
        if selector:
            element = await page.query_selector(selector)
            if element:
                content = await element.text_content()
            else:
                content = ""
        else:
            content = await page.evaluate("() => document.body.innerText")

        content = content or ""
        return content[:max_length] if len(content) > max_length else content
    except Exception as e:
        return f"Error extracting content: {str(e)}"


async def extract_structured_data(page: Page) -> dict | None:
    """Extract JSON-LD structured data from a page.

    Useful for getting rich data from pages that use schema.org markup.

    Args:
        page: Playwright Page instance.

    Returns:
        Parsed structured data dict, or None if not found.
    """
    try:
        script_content = await page.evaluate("""() => {
            const scripts = document.querySelectorAll('script[type="application/ld+json"]');
            const data = [];
            scripts.forEach(script => {
                try {
                    data.push(JSON.parse(script.textContent));
                } catch {}
            });
            return data.length > 0 ? data : null;
        }""")
        return script_content
    except Exception:
        return None
