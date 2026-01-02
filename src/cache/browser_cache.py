"""DiskCache wrapper for browser content caching."""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from diskcache import Cache

from src.config.settings import get_settings


class BrowserCache:
    """Cache for browsed content to avoid redundant web requests.

    Uses DiskCache for persistent, file-based caching with automatic
    expiration based on TTL.
    """

    _instance: Optional["BrowserCache"] = None

    def __init__(self, cache_dir: str | None = None):
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache storage. Defaults to data/cache.
        """
        settings = get_settings()

        if cache_dir is None:
            cache_dir = "data/cache"

        # Ensure cache directory exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        self.cache = Cache(cache_dir)
        self.default_ttl = settings.cache_ttl_seconds

    @classmethod
    def get_instance(cls) -> "BrowserCache":
        """Get or create the singleton cache instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        if cls._instance is not None:
            cls._instance.cache.close()
            cls._instance = None

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """Create a consistent cache key from arguments.

        Args:
            *args: Positional arguments to include in key.
            **kwargs: Keyword arguments to include in key.

        Returns:
            SHA256 hash of the arguments (first 32 chars).
        """
        key_data = {
            "args": args,
            "kwargs": kwargs,
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def get(self, key: str) -> Optional[str]:
        """Get a value from the cache.

        Args:
            key: Cache key.

        Returns:
            Cached value if exists and not expired, None otherwise.
        """
        return self.cache.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds. Defaults to cache default TTL.
        """
        self.cache.set(key, value, expire=ttl or self.default_ttl)

    def delete(self, key: str) -> bool:
        """Delete a key from the cache.

        Args:
            key: Cache key.

        Returns:
            True if key existed and was deleted, False otherwise.
        """
        return self.cache.delete(key)

    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache size and item count.
        """
        return {
            "size_bytes": self.cache.volume(),
            "item_count": len(self.cache),
            "cache_dir": str(self.cache.directory),
        }

    def close(self) -> None:
        """Close the cache connection."""
        self.cache.close()


# Key generation helpers for common cache patterns

def attraction_search_key(city: str, query: str = "attractions") -> str:
    """Generate cache key for attraction searches.

    Args:
        city: City name.
        query: Search query type.

    Returns:
        Cache key string.
    """
    normalized_city = city.lower().strip().replace(" ", "_")
    normalized_query = query.lower().strip().replace(" ", "_")
    return f"attractions:{normalized_city}:{normalized_query}"


def page_content_key(url: str, selector: str | None = None) -> str:
    """Generate cache key for page content.

    Args:
        url: Page URL.
        selector: Optional CSS selector used for extraction.

    Returns:
        Cache key string.
    """
    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
    selector_hash = hashlib.md5((selector or "full").encode()).hexdigest()[:8]
    return f"page:{url_hash}:{selector_hash}"


def food_search_key(city: str, cuisine: str | None = None) -> str:
    """Generate cache key for food/restaurant searches.

    Args:
        city: City name.
        cuisine: Optional cuisine type filter.

    Returns:
        Cache key string.
    """
    normalized_city = city.lower().strip().replace(" ", "_")
    cuisine_part = cuisine.lower().strip().replace(" ", "_") if cuisine else "all"
    return f"food:{normalized_city}:{cuisine_part}"
