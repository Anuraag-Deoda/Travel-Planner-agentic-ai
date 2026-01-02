"""Custom Playwright browser tools for the Research Agent."""

import json
from typing import Optional
from urllib.parse import quote_plus

from langchain_core.tools import tool

from src.cache.browser_cache import BrowserCache, attraction_search_key, page_content_key
from src.config.constants import MAX_ATTRACTIONS_PER_CITY, MAX_CONTENT_LENGTH
from src.tools.browser.browser_manager import (
    BrowserManager,
    extract_text_content,
    navigate_and_wait,
)


@tool
async def search_attractions(
    city: str,
    country: str,
    max_results: int = MAX_ATTRACTIONS_PER_CITY,
) -> str:
    """Search for tourist attractions in a city.

    Uses web search to find popular attractions, landmarks, and things to do.

    Args:
        city: Name of the city to search.
        country: Country the city is in.
        max_results: Maximum number of results to return.

    Returns:
        JSON string with attraction information.
    """
    cache = BrowserCache.get_instance()
    cache_key = attraction_search_key(city, "things_to_do")

    # Check cache first
    cached = cache.get(cache_key)
    if cached:
        return cached

    search_query = f"best things to do in {city} {country} tourist attractions"
    search_url = f"https://www.google.com/search?q={quote_plus(search_query)}"

    attractions = []

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, search_url)

            if not success:
                return json.dumps({"error": "Failed to load search results", "attractions": []})

            # Extract search result snippets
            # This is a simplified extraction - production would need more robust parsing
            results = await page.evaluate("""() => {
                const items = [];
                const searchResults = document.querySelectorAll('.g');

                searchResults.forEach((result, index) => {
                    if (index >= 10) return;

                    const titleEl = result.querySelector('h3');
                    const snippetEl = result.querySelector('.VwiC3b, .s3v9rd');
                    const linkEl = result.querySelector('a');

                    if (titleEl && snippetEl) {
                        items.push({
                            title: titleEl.textContent,
                            snippet: snippetEl.textContent,
                            url: linkEl ? linkEl.href : null
                        });
                    }
                });

                return items;
            }""")

            # Parse results into attractions
            for result in results[:max_results]:
                title = result.get("title", "")
                # Clean up titles that might include site names
                if " - " in title:
                    title = title.split(" - ")[0]

                attractions.append({
                    "name": title,
                    "city": city,
                    "description": result.get("snippet", ""),
                    "source_url": result.get("url"),
                    "category": "attraction",  # Will be refined by LLM
                    "estimated_duration_hours": 2.0,  # Default, will be refined
                })

    except Exception as e:
        return json.dumps({"error": str(e), "attractions": []})

    result = json.dumps({"attractions": attractions, "city": city, "country": country})
    cache.set(cache_key, result)

    return result


@tool
async def get_attraction_details(
    url: str,
    attraction_name: str,
) -> str:
    """Get detailed information about a specific attraction.

    Visits the attraction's webpage to extract details like opening hours,
    ticket prices, and visitor tips.

    Args:
        url: URL of the attraction's page.
        attraction_name: Name of the attraction for context.

    Returns:
        JSON string with attraction details.
    """
    cache = BrowserCache.get_instance()
    cache_key = page_content_key(url)

    cached = cache.get(cache_key)
    if cached:
        return cached

    details = {
        "name": attraction_name,
        "url": url,
        "opening_hours": None,
        "ticket_price": None,
        "description": None,
        "tips": [],
    }

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, url)

            if not success:
                return json.dumps({"error": "Failed to load page", **details})

            # Try to extract structured data first
            structured_data = await page.evaluate("""() => {
                const script = document.querySelector('script[type="application/ld+json"]');
                if (script) {
                    try {
                        return JSON.parse(script.textContent);
                    } catch {}
                }
                return null;
            }""")

            if structured_data:
                if isinstance(structured_data, list):
                    structured_data = structured_data[0] if structured_data else {}

                details["opening_hours"] = structured_data.get("openingHours")
                details["description"] = structured_data.get("description")

            # Fallback to text extraction
            if not details["description"]:
                content = await extract_text_content(page, max_length=2000)
                details["description"] = content[:500]

            # Look for common patterns
            price_text = await page.evaluate("""() => {
                const priceEl = document.querySelector('[class*="price"], [class*="ticket"], [class*="fee"]');
                return priceEl ? priceEl.textContent : null;
            }""")
            if price_text:
                details["ticket_price"] = price_text.strip()[:100]

    except Exception as e:
        details["error"] = str(e)

    result = json.dumps(details)
    cache.set(cache_key, result)

    return result


@tool
async def search_restaurants(
    city: str,
    country: str,
    cuisine_type: Optional[str] = None,
    budget: str = "mid_range",
    max_results: int = 5,
) -> str:
    """Search for restaurants in a city.

    Args:
        city: Name of the city.
        country: Country the city is in.
        cuisine_type: Type of cuisine (optional, e.g., "local", "italian").
        budget: Budget level (budget, mid_range, luxury).
        max_results: Maximum number of results.

    Returns:
        JSON string with restaurant information.
    """
    cache = BrowserCache.get_instance()

    cuisine_query = cuisine_type or "local traditional"
    budget_terms = {
        "budget": "cheap affordable",
        "mid_range": "good value",
        "luxury": "fine dining upscale",
    }

    search_query = f"best {cuisine_query} restaurants {city} {country} {budget_terms.get(budget, '')}"
    search_url = f"https://www.google.com/search?q={quote_plus(search_query)}"

    restaurants = []

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, search_url)

            if not success:
                return json.dumps({"error": "Failed to load search results", "restaurants": []})

            results = await page.evaluate("""() => {
                const items = [];
                const searchResults = document.querySelectorAll('.g');

                searchResults.forEach((result, index) => {
                    if (index >= 8) return;

                    const titleEl = result.querySelector('h3');
                    const snippetEl = result.querySelector('.VwiC3b, .s3v9rd');

                    if (titleEl && snippetEl) {
                        items.push({
                            title: titleEl.textContent,
                            snippet: snippetEl.textContent
                        });
                    }
                });

                return items;
            }""")

            for result in results[:max_results]:
                title = result.get("title", "")
                if " - " in title:
                    title = title.split(" - ")[0]

                restaurants.append({
                    "name": title,
                    "city": city,
                    "description": result.get("snippet", ""),
                    "cuisine_type": cuisine_type or "local",
                    "budget_level": budget,
                })

    except Exception as e:
        return json.dumps({"error": str(e), "restaurants": []})

    return json.dumps({"restaurants": restaurants, "city": city})


@tool
async def get_transport_info(
    from_city: str,
    to_city: str,
    country: str,
) -> str:
    """Get transport options between two cities.

    Args:
        from_city: Origin city.
        to_city: Destination city.
        country: Country (for context).

    Returns:
        JSON string with transport options.
    """
    search_query = f"how to travel from {from_city} to {to_city} {country} train bus flight"
    search_url = f"https://www.google.com/search?q={quote_plus(search_query)}"

    transport_info = {
        "from_city": from_city,
        "to_city": to_city,
        "options": [],
        "raw_info": "",
    }

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, search_url)

            if not success:
                return json.dumps({"error": "Failed to load search results", **transport_info})

            # Get featured snippet or answer box if available
            featured = await page.evaluate("""() => {
                const featured = document.querySelector('.hgKElc, .IZ6rdc, .kp-header');
                return featured ? featured.textContent : null;
            }""")

            if featured:
                transport_info["raw_info"] = featured[:1000]

            # Get search result snippets
            snippets = await page.evaluate("""() => {
                const items = [];
                const results = document.querySelectorAll('.g .VwiC3b');
                results.forEach((el, i) => {
                    if (i < 3) items.push(el.textContent);
                });
                return items.join(' ');
            }""")

            if snippets:
                transport_info["raw_info"] += " " + snippets[:1500]

    except Exception as e:
        transport_info["error"] = str(e)

    return json.dumps(transport_info)


@tool
async def extract_page_content(
    url: str,
    selector: Optional[str] = None,
) -> str:
    """Extract text content from a webpage.

    General-purpose tool for extracting information from any webpage.

    Args:
        url: URL to fetch.
        selector: Optional CSS selector to limit extraction.

    Returns:
        Extracted text content.
    """
    cache = BrowserCache.get_instance()
    cache_key = page_content_key(url, selector)

    cached = cache.get(cache_key)
    if cached:
        return cached

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, url)

            if not success:
                return f"Error: Failed to load {url}"

            content = await extract_text_content(page, selector, MAX_CONTENT_LENGTH)

            if content:
                cache.set(cache_key, content)

            return content

    except Exception as e:
        return f"Error extracting content: {str(e)}"
