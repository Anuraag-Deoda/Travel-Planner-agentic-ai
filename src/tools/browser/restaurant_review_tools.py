"""Restaurant review scraping tools using Playwright."""

import json
from typing import Optional
from urllib.parse import quote_plus

from langchain_core.tools import tool

from src.cache.browser_cache import BrowserCache
from src.cache.transport_cache import RESTAURANT_REVIEW_CACHE_TTL, restaurant_review_key
from src.tools.browser.browser_manager import BrowserManager, navigate_and_wait


@tool
async def scrape_google_maps_restaurants(
    city: str,
    cuisine: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Scrape restaurant listings from Google Maps.

    Args:
        city: City to search for restaurants.
        cuisine: Optional cuisine type filter (e.g., "Indian", "Italian").
        max_results: Maximum number of restaurants to return.

    Returns:
        JSON string with restaurant data including ratings and reviews.
    """
    cache = BrowserCache.get_instance()
    cache_key = restaurant_review_key(city, "google_maps", cuisine)

    cached = cache.get(cache_key)
    if cached:
        return cached

    # Build search query
    search_query = f"best {cuisine + ' ' if cuisine else ''}restaurants in {city}"
    url = f"https://www.google.com/maps/search/{quote_plus(search_query)}"

    result = {
        "source": "google_maps",
        "city": city,
        "cuisine": cuisine,
        "restaurants": [],
        "error": None,
    }

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, url, timeout=45000)

            if not success:
                result["error"] = "Failed to load Google Maps"
                return json.dumps(result)

            # Wait for results and scroll to load more
            await page.wait_for_timeout(3000)

            # Scroll the results panel to load more
            for _ in range(3):
                await page.evaluate("""() => {
                    const panel = document.querySelector('[role="feed"]');
                    if (panel) panel.scrollTop = panel.scrollHeight;
                }""")
                await page.wait_for_timeout(1500)

            # Extract restaurant data
            restaurants_data = await page.evaluate("""(maxResults) => {
                const restaurants = [];

                // Google Maps restaurant cards
                const cards = document.querySelectorAll('[data-result-index], .Nv2PK');

                cards.forEach((card, idx) => {
                    if (idx >= maxResults) return;

                    try {
                        // Name
                        const nameEl = card.querySelector('.qBF1Pd, .fontHeadlineSmall');
                        const name = nameEl ? nameEl.textContent.trim() : null;

                        if (!name) return;

                        // Rating
                        const ratingEl = card.querySelector('.MW4etd, .ZkP5Je');
                        const rating = ratingEl ? parseFloat(ratingEl.textContent) : null;

                        // Review count
                        const reviewEl = card.querySelector('.UY7F9, .e4rVHe');
                        const reviewText = reviewEl ? reviewEl.textContent : '';
                        const reviewMatch = reviewText.match(/\\(([\\d,]+)\\)/);
                        const reviewCount = reviewMatch ? parseInt(reviewMatch[1].replace(',', '')) : null;

                        // Price level
                        const priceEl = card.querySelector('.mgr77e, .W8BRNb');
                        const priceLevel = priceEl ? priceEl.textContent.trim() : null;

                        // Cuisine/Type
                        const typeEl = card.querySelector('.W4Efsd:nth-child(2) > span:nth-child(1), .DkEaL');
                        const cuisineType = typeEl ? typeEl.textContent.trim() : null;

                        // Address
                        const addrEl = card.querySelector('.W4Efsd:nth-child(2) > span:nth-child(3), .AcYnuc');
                        const address = addrEl ? addrEl.textContent.trim() : null;

                        // URL
                        const linkEl = card.querySelector('a[href*="maps"]');
                        const url = linkEl ? linkEl.href : null;

                        restaurants.push({
                            name,
                            rating,
                            review_count: reviewCount,
                            price_level: priceLevel,
                            cuisine_type: cuisineType,
                            address,
                            url,
                        });
                    } catch (e) {
                        // Skip malformed entries
                    }
                });

                return restaurants;
            }""", max_results)

            # Normalize and add to result
            for r in restaurants_data:
                normalized = {
                    "name": r.get("name"),
                    "city": city,
                    "rating": r.get("rating"),
                    "review_count": r.get("review_count"),
                    "price_level": _normalize_price_level(r.get("price_level")),
                    "cuisine_types": [r.get("cuisine_type")] if r.get("cuisine_type") else [],
                    "address": r.get("address"),
                    "source_url": r.get("url"),
                    "review_highlights": [],
                    "popular_dishes": [],
                }
                result["restaurants"].append(normalized)

    except Exception as e:
        result["error"] = str(e)

    json_result = json.dumps(result)

    if not result.get("error") and result["restaurants"]:
        cache.set(cache_key, json_result, ttl=RESTAURANT_REVIEW_CACHE_TTL)

    return json_result


@tool
async def scrape_zomato_restaurants(
    city: str,
    cuisine: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Scrape restaurant listings from Zomato (India-focused).

    Args:
        city: City to search for restaurants.
        cuisine: Optional cuisine type filter.
        max_results: Maximum number of restaurants to return.

    Returns:
        JSON string with restaurant data including ratings and reviews.
    """
    cache = BrowserCache.get_instance()
    cache_key = restaurant_review_key(city, "zomato", cuisine)

    cached = cache.get(cache_key)
    if cached:
        return cached

    # Normalize city for Zomato URL
    city_slug = city.lower().replace(" ", "-")
    base_url = f"https://www.zomato.com/{city_slug}/restaurants"
    if cuisine:
        base_url += f"?cuisine={quote_plus(cuisine.lower())}"

    result = {
        "source": "zomato",
        "city": city,
        "cuisine": cuisine,
        "restaurants": [],
        "error": None,
    }

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, base_url, timeout=45000)

            if not success:
                result["error"] = "Failed to load Zomato"
                return json.dumps(result)

            await page.wait_for_timeout(3000)

            # Scroll to load more results
            for _ in range(2):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(2000)

            # Extract restaurant data
            restaurants_data = await page.evaluate("""(maxResults) => {
                const restaurants = [];

                // Zomato restaurant cards
                const cards = document.querySelectorAll('[data-testid="resturant-card"], .sc-jdHILj');

                cards.forEach((card, idx) => {
                    if (idx >= maxResults) return;

                    try {
                        // Name
                        const nameEl = card.querySelector('h4, .sc-1hp8d8a-0');
                        const name = nameEl ? nameEl.textContent.trim() : null;

                        if (!name) return;

                        // Rating
                        const ratingEl = card.querySelector('[class*="rating"], .sc-1q7bklc-1');
                        const rating = ratingEl ? parseFloat(ratingEl.textContent) : null;

                        // Cuisines
                        const cuisineEl = card.querySelector('[class*="cuisines"], .sc-1hez2tp-0');
                        const cuisines = cuisineEl ? cuisineEl.textContent.split(',').map(c => c.trim()) : [];

                        // Price for two
                        const priceEl = card.querySelector('[class*="cost"], .sc-1hez2tp-0:last-child');
                        const priceText = priceEl ? priceEl.textContent : null;

                        // Location
                        const locEl = card.querySelector('[class*="locality"], .sc-clNaTc');
                        const location = locEl ? locEl.textContent.trim() : null;

                        // URL
                        const linkEl = card.querySelector('a');
                        const url = linkEl ? linkEl.href : null;

                        // Review highlights
                        const reviewEl = card.querySelector('[class*="review"]');
                        const highlight = reviewEl ? reviewEl.textContent.trim() : null;

                        restaurants.push({
                            name,
                            rating,
                            cuisines,
                            price_text: priceText,
                            location,
                            url,
                            highlight,
                        });
                    } catch (e) {
                        // Skip malformed entries
                    }
                });

                return restaurants;
            }""", max_results)

            # Normalize results
            for r in restaurants_data:
                normalized = {
                    "name": r.get("name"),
                    "city": city,
                    "rating": r.get("rating"),
                    "review_count": None,  # Zomato doesn't always show this in listings
                    "price_level": _parse_zomato_price(r.get("price_text")),
                    "cuisine_types": r.get("cuisines", []),
                    "address": r.get("location"),
                    "source_url": r.get("url"),
                    "review_highlights": [r.get("highlight")] if r.get("highlight") else [],
                    "popular_dishes": [],
                }
                result["restaurants"].append(normalized)

    except Exception as e:
        result["error"] = str(e)

    json_result = json.dumps(result)

    if not result.get("error") and result["restaurants"]:
        cache.set(cache_key, json_result, ttl=RESTAURANT_REVIEW_CACHE_TTL)

    return json_result


@tool
async def scrape_swiggy_restaurants(
    city: str,
    cuisine: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Scrape restaurant listings from Swiggy (India-focused).

    Args:
        city: City to search for restaurants.
        cuisine: Optional cuisine type filter.
        max_results: Maximum number of restaurants to return.

    Returns:
        JSON string with restaurant data including ratings and popular dishes.
    """
    cache = BrowserCache.get_instance()
    cache_key = restaurant_review_key(city, "swiggy", cuisine)

    cached = cache.get(cache_key)
    if cached:
        return cached

    # Swiggy URL format
    city_slug = city.lower().replace(" ", "-")
    url = f"https://www.swiggy.com/restaurants?{quote_plus(city_slug)}"

    result = {
        "source": "swiggy",
        "city": city,
        "cuisine": cuisine,
        "restaurants": [],
        "error": None,
    }

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, url, timeout=45000)

            if not success:
                result["error"] = "Failed to load Swiggy"
                return json.dumps(result)

            await page.wait_for_timeout(3000)

            # Handle location popup if present
            try:
                location_input = await page.query_selector('input[placeholder*="location"]')
                if location_input:
                    await location_input.fill(city)
                    await page.wait_for_timeout(1500)
                    suggestion = await page.query_selector('[class*="AutoComplete"]')
                    if suggestion:
                        await suggestion.click()
                        await page.wait_for_timeout(2000)
            except Exception:
                pass

            # Scroll to load more
            for _ in range(2):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1500)

            # Extract restaurant data
            restaurants_data = await page.evaluate("""(maxResults) => {
                const restaurants = [];

                // Swiggy restaurant cards
                const cards = document.querySelectorAll('[data-testid="restaurant-card"], .styles_container__20fKi');

                cards.forEach((card, idx) => {
                    if (idx >= maxResults) return;

                    try {
                        // Name
                        const nameEl = card.querySelector('[class*="restaurantName"], .styles_restaurantTitle__2ZqpN');
                        const name = nameEl ? nameEl.textContent.trim() : null;

                        if (!name) return;

                        // Rating
                        const ratingEl = card.querySelector('[class*="rating"], .styles_avgRating__3F-JH');
                        const rating = ratingEl ? parseFloat(ratingEl.textContent) : null;

                        // Cuisines
                        const cuisineEl = card.querySelector('[class*="cuisines"], .styles_cuisines__jiTc0');
                        const cuisines = cuisineEl ? cuisineEl.textContent.split(',').map(c => c.trim()) : [];

                        // Price for two
                        const priceEl = card.querySelector('[class*="costForTwo"]');
                        const priceText = priceEl ? priceEl.textContent : null;

                        // Delivery time (indicates popularity)
                        const timeEl = card.querySelector('[class*="deliveryTime"]');
                        const deliveryTime = timeEl ? timeEl.textContent.trim() : null;

                        // Location
                        const locEl = card.querySelector('[class*="areaName"]');
                        const location = locEl ? locEl.textContent.trim() : null;

                        restaurants.push({
                            name,
                            rating,
                            cuisines,
                            price_text: priceText,
                            delivery_time: deliveryTime,
                            location,
                        });
                    } catch (e) {
                        // Skip malformed entries
                    }
                });

                return restaurants;
            }""", max_results)

            # Normalize results
            for r in restaurants_data:
                normalized = {
                    "name": r.get("name"),
                    "city": city,
                    "rating": r.get("rating"),
                    "review_count": None,
                    "price_level": _parse_swiggy_price(r.get("price_text")),
                    "cuisine_types": r.get("cuisines", []),
                    "address": r.get("location"),
                    "source_url": None,
                    "review_highlights": [],
                    "popular_dishes": [],
                    "delivery_time": r.get("delivery_time"),
                }
                result["restaurants"].append(normalized)

    except Exception as e:
        result["error"] = str(e)

    json_result = json.dumps(result)

    if not result.get("error") and result["restaurants"]:
        cache.set(cache_key, json_result, ttl=RESTAURANT_REVIEW_CACHE_TTL)

    return json_result


@tool
async def get_restaurant_details_google(
    restaurant_name: str,
    city: str,
) -> str:
    """Get detailed reviews and popular dishes for a specific restaurant.

    Args:
        restaurant_name: Name of the restaurant.
        city: City where the restaurant is located.

    Returns:
        JSON string with detailed review data and popular dishes.
    """
    search_query = f"{restaurant_name} {city} restaurant reviews"
    url = f"https://www.google.com/maps/search/{quote_plus(search_query)}"

    result = {
        "source": "google_maps",
        "restaurant_name": restaurant_name,
        "city": city,
        "rating": None,
        "review_count": None,
        "review_highlights": [],
        "popular_dishes": [],
        "open_hours": None,
        "phone": None,
        "address": None,
        "error": None,
    }

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, url, timeout=45000)

            if not success:
                result["error"] = "Failed to load Google Maps"
                return json.dumps(result)

            await page.wait_for_timeout(3000)

            # Click on first result to open details
            first_result = await page.query_selector('[data-result-index="0"], .Nv2PK')
            if first_result:
                await first_result.click()
                await page.wait_for_timeout(2500)

            # Extract detailed info
            details = await page.evaluate("""() => {
                const details = {
                    rating: null,
                    review_count: null,
                    review_highlights: [],
                    popular_dishes: [],
                    open_hours: null,
                    phone: null,
                    address: null,
                };

                // Rating
                const ratingEl = document.querySelector('.F7nice span[aria-hidden="true"], .MW4etd');
                if (ratingEl) details.rating = parseFloat(ratingEl.textContent);

                // Review count
                const countEl = document.querySelector('.F7nice span:nth-child(2), .UY7F9');
                if (countEl) {
                    const match = countEl.textContent.match(/([\\d,]+)/);
                    if (match) details.review_count = parseInt(match[1].replace(',', ''));
                }

                // Address
                const addrEl = document.querySelector('[data-item-id="address"], .rogA2c');
                if (addrEl) details.address = addrEl.textContent.trim();

                // Phone
                const phoneEl = document.querySelector('[data-item-id^="phone"], .rogA2c');
                if (phoneEl) details.phone = phoneEl.textContent.trim();

                // Hours
                const hoursEl = document.querySelector('[data-item-id="oh"], .o0Svhf');
                if (hoursEl) details.open_hours = hoursEl.textContent.trim();

                // Reviews - extract highlights
                const reviewEls = document.querySelectorAll('.MyEned span, .wiI7pd');
                reviewEls.forEach((el, idx) => {
                    if (idx < 5) {
                        const text = el.textContent.trim();
                        if (text.length > 20 && text.length < 200) {
                            details.review_highlights.push(text);
                        }
                    }
                });

                // Popular dishes from "Popular dishes" section
                const dishEls = document.querySelectorAll('[data-item-id*="dish"] span, .suEOdc');
                dishEls.forEach((el, idx) => {
                    if (idx < 10) {
                        const dish = el.textContent.trim();
                        if (dish && dish.length < 50) {
                            details.popular_dishes.push(dish);
                        }
                    }
                });

                return details;
            }""")

            result.update({
                "rating": details.get("rating"),
                "review_count": details.get("review_count"),
                "review_highlights": details.get("review_highlights", []),
                "popular_dishes": details.get("popular_dishes", []),
                "open_hours": details.get("open_hours"),
                "phone": details.get("phone"),
                "address": details.get("address"),
            })

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result)


def _normalize_price_level(price_str: Optional[str]) -> str:
    """Normalize Google Maps price level to standard format."""
    if not price_str:
        return "unknown"

    dollar_count = price_str.count("$") + price_str.count("₹")

    if dollar_count == 1:
        return "budget"
    elif dollar_count == 2:
        return "moderate"
    elif dollar_count == 3:
        return "expensive"
    elif dollar_count >= 4:
        return "luxury"

    return "unknown"


def _parse_zomato_price(price_text: Optional[str]) -> str:
    """Parse Zomato price text to standard level."""
    if not price_text:
        return "unknown"

    # Zomato shows "₹X00 for two"
    try:
        # Extract number
        import re
        match = re.search(r"₹?\s*(\d+)", price_text.replace(",", ""))
        if match:
            price = int(match.group(1))
            if price < 400:
                return "budget"
            elif price < 800:
                return "moderate"
            elif price < 1500:
                return "expensive"
            else:
                return "luxury"
    except Exception:
        pass

    return "unknown"


def _parse_swiggy_price(price_text: Optional[str]) -> str:
    """Parse Swiggy price text to standard level."""
    if not price_text:
        return "unknown"

    try:
        import re
        match = re.search(r"₹?\s*(\d+)", price_text.replace(",", ""))
        if match:
            price = int(match.group(1))
            if price < 300:
                return "budget"
            elif price < 600:
                return "moderate"
            elif price < 1200:
                return "expensive"
            else:
                return "luxury"
    except Exception:
        pass

    return "unknown"
