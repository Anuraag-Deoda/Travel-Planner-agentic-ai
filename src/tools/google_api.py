"""Google Places API tools for detailed restaurant and attraction data."""

import os
import json
import httpx
from typing import Optional
from urllib.parse import quote_plus

from langchain_core.tools import tool

from src.cache.browser_cache import BrowserCache
from src.cache.transport_cache import RESTAURANT_REVIEW_CACHE_TTL


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PLACES_BASE_URL = "https://maps.googleapis.com/maps/api/place"
PHOTOS_BASE_URL = "https://maps.googleapis.com/maps/api/place/photo"

# Cache TTLs
ATTRACTION_CACHE_TTL = 604800  # 7 days for attractions
PHOTO_CACHE_TTL = 604800  # 7 days for photos


def _get_photo_url(photo_reference: str, max_width: int = 800) -> str:
    """Generate a Google Places photo URL."""
    return (
        f"{PHOTOS_BASE_URL}?maxwidth={max_width}"
        f"&photo_reference={photo_reference}"
        f"&key={GOOGLE_API_KEY}"
    )


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
    cache_key = f"places_restaurants:{city}:{cuisine or 'all'}"

    cached = cache.get(cache_key)
    if cached:
        return cached

    result = {
        "source": "google_places_api",
        "city": city,
        "cuisine": cuisine,
        "restaurants": [],
        "error": None,
    }

    try:
        # Build search query
        query = f"best {cuisine + ' ' if cuisine else ''}restaurants in {city}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Text search for restaurants
            search_url = f"{PLACES_BASE_URL}/textsearch/json"
            search_params = {
                "query": query,
                "type": "restaurant",
                "key": GOOGLE_API_KEY,
            }

            response = await client.get(search_url, params=search_params)
            data = response.json()

            if data.get("status") != "OK":
                result["error"] = data.get("status")
                return json.dumps(result)

            places = data.get("results", [])[:max_results]

            # Get detailed info for each place
            for place in places:
                place_id = place.get("place_id")

                # Get place details
                details = await _get_place_details(client, place_id)

                restaurant = {
                    "name": place.get("name"),
                    "city": city,
                    "address": place.get("formatted_address"),
                    "rating": place.get("rating"),
                    "review_count": place.get("user_ratings_total"),
                    "price_level": _convert_price_level(place.get("price_level")),
                    "cuisine_types": details.get("types", []),
                    "source": "google_places_api",
                    "place_id": place_id,

                    # Detailed info from Place Details API
                    "phone": details.get("phone"),
                    "website": details.get("website"),
                    "open_now": details.get("open_now"),
                    "opening_hours": details.get("opening_hours"),
                    "google_maps_url": details.get("url"),

                    # Reviews with actual text
                    "review_highlights": details.get("reviews", []),
                    "popular_dishes": [],  # Will be extracted from reviews

                    # Photos
                    "photos": [],
                    "photo_urls": [],
                }

                # Get photo URLs
                photos = place.get("photos", [])
                for photo in photos[:3]:  # Get up to 3 photos
                    photo_ref = photo.get("photo_reference")
                    if photo_ref:
                        restaurant["photos"].append(photo_ref)
                        restaurant["photo_urls"].append(_get_photo_url(photo_ref))

                # Extract dishes mentioned in reviews
                restaurant["popular_dishes"] = _extract_dishes_from_reviews(
                    details.get("raw_reviews", [])
                )

                result["restaurants"].append(restaurant)

    except Exception as e:
        result["error"] = str(e)

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
            # Find the place
            search_url = f"{PLACES_BASE_URL}/findplacefromtext/json"
            search_params = {
                "input": query,
                "inputtype": "textquery",
                "fields": "place_id,name,formatted_address,rating,user_ratings_total,price_level,photos",
                "key": GOOGLE_API_KEY,
            }

            response = await client.get(search_url, params=search_params)
            data = response.json()

            if data.get("status") != "OK" or not data.get("candidates"):
                result["error"] = "Restaurant not found"
                return json.dumps(result)

            place = data["candidates"][0]
            place_id = place.get("place_id")

            # Get full details
            details = await _get_place_details(client, place_id, include_reviews=True)

            result.update({
                "name": place.get("name"),
                "address": place.get("formatted_address"),
                "rating": place.get("rating"),
                "review_count": place.get("user_ratings_total"),
                "price_level": _convert_price_level(place.get("price_level")),
                "phone": details.get("phone"),
                "website": details.get("website"),
                "open_now": details.get("open_now"),
                "opening_hours": details.get("opening_hours"),
                "google_maps_url": details.get("url"),
                "reviews": details.get("reviews", []),
                "review_highlights": details.get("reviews", [])[:5],
                "popular_dishes": _extract_dishes_from_reviews(details.get("raw_reviews", [])),
                "photo_urls": [],
            })

            # Get photos
            photos = place.get("photos", [])
            for photo in photos[:5]:
                photo_ref = photo.get("photo_reference")
                if photo_ref:
                    result["photo_urls"].append(_get_photo_url(photo_ref, max_width=1200))

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
    cache_key = f"places_attractions:{city}:{attraction_type or 'all'}"

    cached = cache.get(cache_key)
    if cached:
        return cached

    result = {
        "source": "google_places_api",
        "city": city,
        "attraction_type": attraction_type,
        "attractions": [],
        "error": None,
    }

    try:
        # Build search query
        if attraction_type:
            query = f"{attraction_type} in {city}"
        else:
            query = f"top tourist attractions in {city}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            search_url = f"{PLACES_BASE_URL}/textsearch/json"
            search_params = {
                "query": query,
                "key": GOOGLE_API_KEY,
            }

            response = await client.get(search_url, params=search_params)
            data = response.json()

            if data.get("status") != "OK":
                result["error"] = data.get("status")
                return json.dumps(result)

            places = data.get("results", [])[:max_results]

            for place in places:
                place_id = place.get("place_id")

                # Get additional details
                details = await _get_place_details(client, place_id)

                attraction = {
                    "name": place.get("name"),
                    "city": city,
                    "address": place.get("formatted_address"),
                    "rating": place.get("rating"),
                    "review_count": place.get("user_ratings_total"),
                    "types": place.get("types", []),
                    "category": _categorize_attraction(place.get("types", [])),
                    "source": "google_places_api",
                    "place_id": place_id,

                    # Details
                    "phone": details.get("phone"),
                    "website": details.get("website"),
                    "open_now": details.get("open_now"),
                    "opening_hours": details.get("opening_hours"),
                    "google_maps_url": details.get("url"),

                    # Reviews
                    "review_highlights": details.get("reviews", [])[:3],

                    # Photos
                    "photos": [],
                    "photo_urls": [],
                }

                # Get photo URLs
                photos = place.get("photos", [])
                for photo in photos[:4]:
                    photo_ref = photo.get("photo_reference")
                    if photo_ref:
                        attraction["photos"].append(photo_ref)
                        attraction["photo_urls"].append(_get_photo_url(photo_ref, max_width=1200))

                result["attractions"].append(attraction)

    except Exception as e:
        result["error"] = str(e)

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
            search_url = f"{PLACES_BASE_URL}/findplacefromtext/json"
            search_params = {
                "input": query,
                "inputtype": "textquery",
                "fields": "place_id,name,formatted_address,rating,user_ratings_total,types,photos,geometry",
                "key": GOOGLE_API_KEY,
            }

            response = await client.get(search_url, params=search_params)
            data = response.json()

            if data.get("status") != "OK" or not data.get("candidates"):
                result["error"] = "Attraction not found"
                return json.dumps(result)

            place = data["candidates"][0]
            place_id = place.get("place_id")

            # Get full details
            details = await _get_place_details(client, place_id, include_reviews=True)

            result.update({
                "name": place.get("name"),
                "address": place.get("formatted_address"),
                "rating": place.get("rating"),
                "review_count": place.get("user_ratings_total"),
                "types": place.get("types", []),
                "category": _categorize_attraction(place.get("types", [])),
                "phone": details.get("phone"),
                "website": details.get("website"),
                "open_now": details.get("open_now"),
                "opening_hours": details.get("opening_hours"),
                "google_maps_url": details.get("url"),
                "reviews": details.get("reviews", []),
                "location": place.get("geometry", {}).get("location"),
                "photo_urls": [],
            })

            # Get photos
            photos = place.get("photos", [])
            for photo in photos[:6]:
                photo_ref = photo.get("photo_reference")
                if photo_ref:
                    result["photo_urls"].append(_get_photo_url(photo_ref, max_width=1200))

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result)


async def _get_place_details(
    client: httpx.AsyncClient,
    place_id: str,
    include_reviews: bool = True,
) -> dict:
    """Get detailed place information from Places API.

    Args:
        client: HTTP client.
        place_id: Google Place ID.
        include_reviews: Whether to fetch reviews.

    Returns:
        Dictionary with place details.
    """
    details_url = f"{PLACES_BASE_URL}/details/json"

    fields = [
        "name",
        "formatted_phone_number",
        "website",
        "opening_hours",
        "url",
        "types",
    ]
    if include_reviews:
        fields.append("reviews")

    params = {
        "place_id": place_id,
        "fields": ",".join(fields),
        "key": GOOGLE_API_KEY,
    }

    try:
        response = await client.get(details_url, params=params)
        data = response.json()

        if data.get("status") != "OK":
            return {}

        result_data = data.get("result", {})

        details = {
            "phone": result_data.get("formatted_phone_number"),
            "website": result_data.get("website"),
            "url": result_data.get("url"),
            "types": result_data.get("types", []),
        }

        # Parse opening hours
        hours = result_data.get("opening_hours", {})
        if hours:
            details["open_now"] = hours.get("open_now")
            details["opening_hours"] = hours.get("weekday_text", [])

        # Parse reviews
        if include_reviews:
            reviews = result_data.get("reviews", [])
            details["raw_reviews"] = reviews
            details["reviews"] = [
                {
                    "text": r.get("text", "")[:200],  # Truncate long reviews
                    "rating": r.get("rating"),
                    "author": r.get("author_name"),
                    "time_description": r.get("relative_time_description"),
                }
                for r in reviews[:5]
            ]

        return details

    except Exception:
        return {}


def _convert_price_level(level: Optional[int]) -> str:
    """Convert Google's price level (0-4) to descriptive string."""
    if level is None:
        return "unknown"

    mapping = {
        0: "free",
        1: "budget",
        2: "moderate",
        3: "expensive",
        4: "luxury",
    }
    return mapping.get(level, "unknown")


def _categorize_attraction(types: list[str]) -> str:
    """Categorize attraction based on Google place types."""
    type_set = set(types)

    if type_set & {"museum", "art_gallery"}:
        return "museum"
    if type_set & {"hindu_temple", "church", "mosque", "synagogue", "place_of_worship"}:
        return "religious_site"
    if type_set & {"park", "natural_feature", "campground"}:
        return "nature"
    if type_set & {"amusement_park", "zoo", "aquarium"}:
        return "entertainment"
    if type_set & {"shopping_mall", "market"}:
        return "shopping"
    if type_set & {"tourist_attraction", "point_of_interest"}:
        return "landmark"
    if type_set & {"restaurant", "cafe", "bar"}:
        return "food_drink"

    return "attraction"


def _extract_dishes_from_reviews(reviews: list[dict]) -> list[str]:
    """Extract food dishes mentioned in reviews."""
    # Common food-related words to look for
    food_indicators = [
        "try the", "must try", "recommend the", "order the", "loved the",
        "amazing", "delicious", "best", "incredible", "fantastic",
    ]

    dishes = []
    seen = set()

    for review in reviews:
        text = (review.get("text") or "").lower()

        for indicator in food_indicators:
            if indicator in text:
                # Find words after the indicator
                idx = text.find(indicator)
                if idx != -1:
                    # Get the next few words
                    after = text[idx + len(indicator):idx + len(indicator) + 50]
                    words = after.strip().split()[:4]
                    if words:
                        dish = " ".join(words).strip(".,!?")
                        if dish and dish not in seen and len(dish) > 3:
                            seen.add(dish)
                            dishes.append(dish.title())

    return dishes[:8]  # Return top 8 dishes
