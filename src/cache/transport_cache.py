"""Transport-specific caching with shorter TTL for volatile prices."""

import hashlib
from typing import Optional


# Transport prices are more volatile than attractions - use shorter TTL
TRANSPORT_CACHE_TTL = 14400  # 4 hours in seconds

# For high-frequency routes (prices change often), use even shorter TTL
DYNAMIC_ROUTE_TTL = 7200  # 2 hours

# Station/airport info is stable - use longer TTL
STATION_CACHE_TTL = 604800  # 7 days

# Restaurant reviews don't change frequently
RESTAURANT_REVIEW_CACHE_TTL = 86400  # 24 hours


def transport_price_key(
    mode: str,
    from_location: str,
    to_location: str,
    travel_date: str,
    class_type: Optional[str] = None,
) -> str:
    """Generate cache key for transport price queries.

    Args:
        mode: Transport mode (flight, train, bus).
        from_location: Origin city/station.
        to_location: Destination city/station.
        travel_date: Travel date in ISO format.
        class_type: Optional class filter.

    Returns:
        Cache key string.
    """
    normalized = f"{mode}:{from_location}:{to_location}:{travel_date}".lower()
    normalized = normalized.replace(" ", "_")

    if class_type:
        normalized += f":{class_type.lower()}"

    # Hash for consistent length
    key_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
    return f"transport:{key_hash}"


def station_info_key(city: str, country: str) -> str:
    """Generate cache key for station/airport info.

    Args:
        city: City name.
        country: Country name.

    Returns:
        Cache key string.
    """
    normalized = f"{city}:{country}".lower().replace(" ", "_")
    key_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
    return f"stations:{key_hash}"


def restaurant_review_key(
    city: str,
    source: str,
    cuisine: Optional[str] = None,
) -> str:
    """Generate cache key for restaurant review searches.

    Args:
        city: City name.
        source: Review source (google_maps, zomato, swiggy).
        cuisine: Optional cuisine type filter.

    Returns:
        Cache key string.
    """
    normalized_city = city.lower().strip().replace(" ", "_")
    cuisine_part = cuisine.lower().strip().replace(" ", "_") if cuisine else "all"
    return f"restaurant_reviews:{source}:{normalized_city}:{cuisine_part}"


def is_high_frequency_route(from_location: str, to_location: str) -> bool:
    """Check if a route is high-frequency (prices change often).

    High-frequency routes include major city pairs with many daily departures.
    These should use DYNAMIC_ROUTE_TTL instead of TRANSPORT_CACHE_TTL.

    Args:
        from_location: Origin city.
        to_location: Destination city.

    Returns:
        True if high-frequency route, False otherwise.
    """
    high_freq_routes = {
        # India domestic
        ("delhi", "mumbai"), ("mumbai", "delhi"),
        ("delhi", "bangalore"), ("bangalore", "delhi"),
        ("delhi", "bengaluru"), ("bengaluru", "delhi"),
        ("mumbai", "bangalore"), ("bangalore", "mumbai"),
        ("mumbai", "bengaluru"), ("bengaluru", "mumbai"),
        ("mumbai", "goa"), ("goa", "mumbai"),
        ("delhi", "kolkata"), ("kolkata", "delhi"),
        ("delhi", "chennai"), ("chennai", "delhi"),
        ("mumbai", "chennai"), ("chennai", "mumbai"),
        # International
        ("new york", "london"), ("london", "new york"),
        ("tokyo", "osaka"), ("osaka", "tokyo"),
        ("singapore", "kuala lumpur"), ("kuala lumpur", "singapore"),
        ("hong kong", "singapore"), ("singapore", "hong kong"),
        ("dubai", "mumbai"), ("mumbai", "dubai"),
        ("dubai", "delhi"), ("delhi", "dubai"),
    }

    route = (from_location.lower().strip(), to_location.lower().strip())
    return route in high_freq_routes


def get_transport_cache_ttl(from_location: str, to_location: str) -> int:
    """Get appropriate cache TTL based on route characteristics.

    Args:
        from_location: Origin city.
        to_location: Destination city.

    Returns:
        TTL in seconds.
    """
    if is_high_frequency_route(from_location, to_location):
        return DYNAMIC_ROUTE_TTL
    return TRANSPORT_CACHE_TTL
