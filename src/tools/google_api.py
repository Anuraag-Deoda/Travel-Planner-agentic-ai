"""Google Places API (New) tools for detailed restaurant and attraction data."""

import asyncio
import os
import json
import httpx
import logging
from typing import Optional

from langchain_core.tools import tool

from src.cache.browser_cache import BrowserCache
from src.cache.transport_cache import RESTAURANT_REVIEW_CACHE_TTL


logger = logging.getLogger(__name__)

# Load API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # Try loading from .env file directly
    from pathlib import Path
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("GOOGLE_API_KEY"):
                    GOOGLE_API_KEY = line.split("=", 1)[1].strip().strip("'\"")
                    break

# New Places API (v1) endpoints
PLACES_API_BASE = "https://places.googleapis.com/v1"

# Cache TTLs
ATTRACTION_CACHE_TTL = 604800  # 7 days for attractions
PHOTO_CACHE_TTL = 604800  # 7 days for photos


def _get_headers() -> dict:
    """Get headers for Places API (New) requests."""
    return {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "*",
    }


def _get_photo_url(photo_name: str, max_width: int = 800) -> str:
    """Generate a Google Places photo URL from the new API format.

    The new API returns photo names like 'places/PLACE_ID/photos/PHOTO_REF'
    """
    return (
        f"{PLACES_API_BASE}/{photo_name}/media"
        f"?maxWidthPx={max_width}&key={GOOGLE_API_KEY}"
    )


def _convert_price_level(level: Optional[str]) -> str:
    """Convert Google's new price level format to descriptive string."""
    if not level:
        return "unknown"

    mapping = {
        "PRICE_LEVEL_FREE": "free",
        "PRICE_LEVEL_INEXPENSIVE": "budget",
        "PRICE_LEVEL_MODERATE": "moderate",
        "PRICE_LEVEL_EXPENSIVE": "expensive",
        "PRICE_LEVEL_VERY_EXPENSIVE": "luxury",
    }
    return mapping.get(level, "unknown")


def _categorize_attraction(types: list[str]) -> str:
    """Categorize attraction based on Google place types."""
    type_set = set(types)

    if type_set & {"museum", "art_gallery"}:
        return "museum"
    if type_set & {"hindu_temple", "church", "mosque", "synagogue", "place_of_worship"}:
        return "religious_site"
    if type_set & {"park", "natural_feature", "campground", "national_park"}:
        return "nature"
    if type_set & {"amusement_park", "zoo", "aquarium"}:
        return "entertainment"
    if type_set & {"shopping_mall", "market"}:
        return "shopping"
    if type_set & {"tourist_attraction", "point_of_interest", "historical_landmark"}:
        return "landmark"
    if type_set & {"restaurant", "cafe", "bar"}:
        return "food_drink"

    return "attraction"


def _extract_dishes_from_reviews(reviews: list[dict]) -> list[str]:
    """Extract food dishes mentioned in reviews."""
    food_indicators = [
        "try the", "must try", "recommend the", "order the", "loved the",
        "amazing", "delicious", "best", "incredible", "fantastic",
    ]

    dishes = []
    seen = set()

    for review in reviews:
        text = (review.get("text") or review.get("originalText", {}).get("text", "")).lower()

        for indicator in food_indicators:
            if indicator in text:
                idx = text.find(indicator)
                if idx != -1:
                    after = text[idx + len(indicator):idx + len(indicator) + 50]
                    words = after.strip().split()[:4]
                    if words:
                        dish = " ".join(words).strip(".,!?")
                        if dish and dish not in seen and len(dish) > 3:
                            seen.add(dish)
                            dishes.append(dish.title())

    return dishes[:8]


async def _text_search(
    client: httpx.AsyncClient,
    query: str,
    included_type: Optional[str] = None,
    max_results: int = 20,
) -> dict:
    """Perform a text search using the new Places API.

    Args:
        client: HTTP client.
        query: Search query.
        included_type: Optional place type filter.
        max_results: Maximum results to return.

    Returns:
        API response dict.
    """
    url = f"{PLACES_API_BASE}/places:searchText"

    body = {
        "textQuery": query,
        "maxResultCount": min(max_results, 20),  # API max is 20
        "languageCode": "en",
    }

    if included_type:
        body["includedType"] = included_type

    # Request specific fields to reduce response size
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": ",".join([
            "places.id",
            "places.displayName",
            "places.formattedAddress",
            "places.rating",
            "places.userRatingCount",
            "places.priceLevel",
            "places.types",
            "places.photos",
            "places.websiteUri",
            "places.nationalPhoneNumber",
            "places.googleMapsUri",
            "places.regularOpeningHours",
            "places.reviews",
            "places.editorialSummary",
        ]),
    }

    try:
        response = await client.post(url, json=body, headers=headers)
        data = response.json()

        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", str(data))
            logger.error(f"Places API error: {error_msg}")
            return {"error": error_msg, "places": []}

        return data
    except Exception as e:
        logger.error(f"Places API request failed: {e}")
        return {"error": str(e), "places": []}


async def _get_place_details(
    client: httpx.AsyncClient,
    place_id: str,
) -> dict:
    """Get detailed place information from the new Places API.

    Args:
        client: HTTP client.
        place_id: Google Place ID.

    Returns:
        Place details dict.
    """
    url = f"{PLACES_API_BASE}/places/{place_id}"

    headers = {
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": ",".join([
            "id",
            "displayName",
            "formattedAddress",
            "rating",
            "userRatingCount",
            "priceLevel",
            "types",
            "photos",
            "websiteUri",
            "nationalPhoneNumber",
            "googleMapsUri",
            "regularOpeningHours",
            "reviews",
            "editorialSummary",
        ]),
    }

    try:
        response = await client.get(url, headers=headers)

        if response.status_code != 200:
            return {}

        return response.json()
    except Exception:
        return {}


def _parse_place(place: dict, city: str) -> dict:
    """Parse a place from the new API format into our standard format."""
    place_id = place.get("id", "")

    # Get photos
    photo_urls = []
    photos = place.get("photos", [])
    for photo in photos[:4]:
        photo_name = photo.get("name")
        if photo_name:
            photo_urls.append(_get_photo_url(photo_name))

    # Get reviews
    reviews = place.get("reviews", [])
    review_highlights = []
    for review in reviews[:5]:
        text = review.get("text", {}).get("text", "") or review.get("originalText", {}).get("text", "")
        if text:
            review_highlights.append({
                "text": text[:200],
                "rating": review.get("rating"),
                "author": review.get("authorAttribution", {}).get("displayName"),
            })

    # Get opening hours
    opening_hours = []
    hours_data = place.get("regularOpeningHours", {})
    if hours_data:
        opening_hours = hours_data.get("weekdayDescriptions", [])

    return {
        "name": place.get("displayName", {}).get("text", "Unknown"),
        "city": city,
        "address": place.get("formattedAddress", ""),
        "rating": place.get("rating"),
        "review_count": place.get("userRatingCount"),
        "price_level": _convert_price_level(place.get("priceLevel")),
        "types": place.get("types", []),
        "source": "google_places_api",
        "place_id": place_id,
        "phone": place.get("nationalPhoneNumber"),
        "website": place.get("websiteUri"),
        "google_maps_url": place.get("googleMapsUri"),
        "opening_hours": opening_hours,
        "review_highlights": review_highlights,
        "photo_urls": photo_urls,
        "editorial_summary": place.get("editorialSummary", {}).get("text", ""),
    }


@tool
async def search_restaurants_places_api(
    city: str,
    cuisine: Optional[str] = None,
    max_results: int = 15,
) -> str:
    """Search for restaurants using Google Places API with detailed info and photos.

    Args:
        city: City to search in.
        cuisine: Optional cuisine type (e.g., "Indian", "Italian", "seafood").
        max_results: Maximum number of results.

    Returns:
        JSON with detailed restaurant data including ratings, reviews, photos, price levels.
    """
    if not GOOGLE_API_KEY:
        return json.dumps({"error": "Google API key not configured", "restaurants": []})

    cache = BrowserCache.get_instance()
    cache_key = f"places_restaurants_v2:{city}:{cuisine or 'all'}"

    cached = cache.get(cache_key)
    if cached:
        logger.info(f"Using cached restaurant data for {city}")
        return cached

    result = {
        "source": "google_places_api",
        "city": city,
        "cuisine": cuisine,
        "restaurants": [],
        "error": None,
    }

    try:
        query = f"best {cuisine + ' ' if cuisine else ''}restaurants in {city}"
        logger.info(f"Searching restaurants: {query}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            data = await _text_search(client, query, included_type="restaurant", max_results=max_results)

            if data.get("error"):
                result["error"] = data["error"]
                return json.dumps(result)

            places = data.get("places", [])
            logger.info(f"Found {len(places)} restaurants in {city}")

            for place in places:
                restaurant = _parse_place(place, city)
                restaurant["cuisine_types"] = [t for t in place.get("types", []) if "restaurant" not in t.lower()]
                restaurant["popular_dishes"] = _extract_dishes_from_reviews(place.get("reviews", []))
                result["restaurants"].append(restaurant)

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Restaurant search exception: {e}")

    json_result = json.dumps(result)

    if not result.get("error") and result["restaurants"]:
        cache.set(cache_key, json_result, ttl=RESTAURANT_REVIEW_CACHE_TTL)

    return json_result


@tool
async def get_restaurant_details_places_api(
    restaurant_name: str,
    city: str,
) -> str:
    """Get detailed information for a specific restaurant including reviews and photos.

    Args:
        restaurant_name: Name of the restaurant.
        city: City where the restaurant is located.

    Returns:
        JSON with detailed restaurant data.
    """
    if not GOOGLE_API_KEY:
        return json.dumps({"error": "Google API key not configured"})

    result = {
        "source": "google_places_api",
        "restaurant_name": restaurant_name,
        "city": city,
        "error": None,
    }

    try:
        query = f"{restaurant_name} restaurant {city}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            data = await _text_search(client, query, included_type="restaurant", max_results=1)

            if data.get("error") or not data.get("places"):
                result["error"] = data.get("error") or "Restaurant not found"
                return json.dumps(result)

            place = data["places"][0]
            result.update(_parse_place(place, city))
            result["popular_dishes"] = _extract_dishes_from_reviews(place.get("reviews", []))

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result)


@tool
async def search_attractions_places_api(
    city: str,
    attraction_type: Optional[str] = None,
    max_results: int = 20,
) -> str:
    """Search for tourist attractions using Google Places API with photos.

    Args:
        city: City to search in.
        attraction_type: Optional type (e.g., "museum", "temple", "landmark", "park").
        max_results: Maximum number of results.

    Returns:
        JSON with detailed attraction data including photos, ratings, reviews.
    """
    if not GOOGLE_API_KEY:
        return json.dumps({"error": "Google API key not configured", "attractions": []})

    cache = BrowserCache.get_instance()
    cache_key = f"places_attractions_v2:{city}:{attraction_type or 'all'}"

    cached = cache.get(cache_key)
    if cached:
        logger.info(f"Using cached attraction data for {city}")
        return cached

    result = {
        "source": "google_places_api",
        "city": city,
        "attraction_type": attraction_type,
        "attractions": [],
        "error": None,
    }

    try:
        if attraction_type:
            query = f"{attraction_type} in {city}"
        else:
            query = f"top tourist attractions things to do in {city}"

        logger.info(f"Searching attractions: {query}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            data = await _text_search(client, query, max_results=max_results)

            if data.get("error"):
                result["error"] = data["error"]
                return json.dumps(result)

            places = data.get("places", [])
            logger.info(f"Found {len(places)} attractions in {city}")

            for place in places:
                attraction = _parse_place(place, city)
                attraction["category"] = _categorize_attraction(place.get("types", []))
                result["attractions"].append(attraction)

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Attraction search exception: {e}")

    json_result = json.dumps(result)

    if not result.get("error") and result["attractions"]:
        cache.set(cache_key, json_result, ttl=ATTRACTION_CACHE_TTL)

    return json_result


@tool
async def get_attraction_details_places_api(
    attraction_name: str,
    city: str,
) -> str:
    """Get detailed information for a specific attraction including photos and reviews.

    Args:
        attraction_name: Name of the attraction.
        city: City where the attraction is located.

    Returns:
        JSON with detailed attraction data including photos.
    """
    if not GOOGLE_API_KEY:
        return json.dumps({"error": "Google API key not configured"})

    result = {
        "source": "google_places_api",
        "attraction_name": attraction_name,
        "city": city,
        "error": None,
    }

    try:
        query = f"{attraction_name} {city}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            data = await _text_search(client, query, max_results=1)

            if data.get("error") or not data.get("places"):
                result["error"] = data.get("error") or "Attraction not found"
                return json.dumps(result)

            place = data["places"][0]
            result.update(_parse_place(place, city))
            result["category"] = _categorize_attraction(place.get("types", []))

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result)


@tool
async def search_hotels_places_api(
    city: str,
    budget_level: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Search for hotels using Google Places API with detailed info and photos.

    Args:
        city: City to search in.
        budget_level: Optional budget level ("budget", "mid_range", "luxury").
        max_results: Maximum number of results.

    Returns:
        JSON with detailed hotel data including ratings, reviews, photos, amenities.
    """
    if not GOOGLE_API_KEY:
        logger.error("Google API key not configured for hotel search")
        return json.dumps({"error": "Google API key not configured", "hotels": []})

    cache = BrowserCache.get_instance()
    cache_key = f"places_hotels_v2:{city}:{budget_level or 'all'}"

    cached = cache.get(cache_key)
    if cached:
        logger.info(f"Using cached hotel data for {city}")
        return cached

    result = {
        "source": "google_places_api",
        "city": city,
        "budget_level": budget_level,
        "hotels": [],
        "error": None,
    }

    try:
        # Build search query based on budget
        if budget_level == "budget":
            query = f"budget hotels hostels guesthouses in {city}"
        elif budget_level == "luxury":
            query = f"luxury 5 star hotels resorts in {city}"
        else:
            query = f"best hotels in {city}"

        logger.info(f"Searching hotels: {query}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            data = await _text_search(client, query, included_type="hotel", max_results=max_results)

            if data.get("error"):
                result["error"] = data["error"]
                logger.error(f"Hotel search failed: {result['error']}")
                return json.dumps(result)

            places = data.get("places", [])
            logger.info(f"Found {len(places)} hotels in {city}")

            for place in places:
                hotel = _parse_place(place, city)
                result["hotels"].append(hotel)

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Hotel search exception: {e}")

    json_result = json.dumps(result)

    if not result.get("error") and result["hotels"]:
        cache.set(cache_key, json_result, ttl=ATTRACTION_CACHE_TTL)

    return json_result


async def search_all_city_data(city: str, budget_level: str = "mid_range") -> dict:
    """Search for all city data (attractions, restaurants, hotels) in parallel.

    Args:
        city: City name.
        budget_level: Budget level for filtering.

    Returns:
        Dict with attractions, restaurants, and hotels.
    """
    logger.info(f"Fetching all data for {city} in parallel...")

    # Run all searches in parallel
    results = await asyncio.gather(
        search_attractions_places_api.ainvoke({"city": city, "max_results": 15}),
        search_restaurants_places_api.ainvoke({"city": city, "max_results": 15}),
        search_hotels_places_api.ainvoke({"city": city, "budget_level": budget_level, "max_results": 8}),
        return_exceptions=True,
    )

    attractions_data = json.loads(results[0]) if not isinstance(results[0], Exception) else {"attractions": [], "error": str(results[0])}
    restaurants_data = json.loads(results[1]) if not isinstance(results[1], Exception) else {"restaurants": [], "error": str(results[1])}
    hotels_data = json.loads(results[2]) if not isinstance(results[2], Exception) else {"hotels": [], "error": str(results[2])}

    return {
        "city": city,
        "attractions": attractions_data.get("attractions", []),
        "restaurants": restaurants_data.get("restaurants", []),
        "hotels": hotels_data.get("hotels", []),
        "errors": {
            "attractions": attractions_data.get("error"),
            "restaurants": restaurants_data.get("error"),
            "hotels": hotels_data.get("error"),
        }
    }
