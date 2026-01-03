"""Models for scraped restaurant review data."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ReviewSource(str, Enum):
    """Source of restaurant review data."""

    GOOGLE_MAPS = "google_maps"
    ZOMATO = "zomato"
    SWIGGY = "swiggy"
    LLM_GENERATED = "llm_generated"


class PriceLevel(str, Enum):
    """Restaurant price level indicators."""

    BUDGET = "budget"  # $ - cheap eats
    MODERATE = "moderate"  # $$ - mid-range
    EXPENSIVE = "expensive"  # $$$ - upscale
    LUXURY = "luxury"  # $$$$ - fine dining
    UNKNOWN = "unknown"


class RestaurantReview(BaseModel):
    """Review data scraped from a restaurant listing."""

    name: str
    city: str
    country: Optional[str] = None
    address: Optional[str] = None
    rating: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="Rating out of 5 stars",
    )
    review_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of reviews",
    )
    price_level: PriceLevel = PriceLevel.UNKNOWN
    price_range: Optional[str] = None  # "$10-20 per person"
    cuisine_types: list[str] = Field(default_factory=list)
    review_highlights: list[str] = Field(
        default_factory=list,
        description="Key phrases extracted from reviews",
    )
    popular_dishes: list[str] = Field(
        default_factory=list,
        description="Most mentioned dishes in reviews",
    )
    source: ReviewSource
    source_url: Optional[str] = None
    open_hours: Optional[str] = None
    phone: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class RestaurantSearchResult(BaseModel):
    """Result from a restaurant review scraping operation."""

    city: str
    country: Optional[str] = None
    source: ReviewSource
    cuisine_filter: Optional[str] = None
    restaurants: list[RestaurantReview] = Field(default_factory=list)
    total_found: int = 0
    error: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class AggregatedRestaurant(BaseModel):
    """Restaurant with aggregated data from multiple review sources."""

    name: str
    city: str
    country: Optional[str] = None
    address: Optional[str] = None

    # Aggregated ratings
    avg_rating: Optional[float] = None
    total_review_count: int = 0
    ratings_by_source: dict[str, float] = Field(default_factory=dict)

    # Combined data
    price_level: PriceLevel = PriceLevel.UNKNOWN
    cuisine_types: list[str] = Field(default_factory=list)
    all_review_highlights: list[str] = Field(default_factory=list)
    popular_dishes: list[str] = Field(default_factory=list)

    # Source tracking
    sources: list[str] = Field(default_factory=list)
    primary_source_url: Optional[str] = None

    @classmethod
    def from_reviews(cls, reviews: list[RestaurantReview]) -> "AggregatedRestaurant":
        """Create aggregated restaurant from multiple review sources."""
        if not reviews:
            raise ValueError("Cannot aggregate empty review list")

        first = reviews[0]

        # Aggregate ratings
        ratings = [r.rating for r in reviews if r.rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        total_reviews = sum(r.review_count or 0 for r in reviews)

        ratings_by_source = {
            r.source.value: r.rating
            for r in reviews
            if r.rating is not None
        }

        # Combine unique cuisines
        all_cuisines = []
        for r in reviews:
            all_cuisines.extend(r.cuisine_types)
        cuisines = list(dict.fromkeys(all_cuisines))  # Preserve order, remove dups

        # Combine highlights (limit to top 10)
        all_highlights = []
        for r in reviews:
            all_highlights.extend(r.review_highlights)
        highlights = list(dict.fromkeys(all_highlights))[:10]

        # Combine popular dishes
        all_dishes = []
        for r in reviews:
            all_dishes.extend(r.popular_dishes)
        dishes = list(dict.fromkeys(all_dishes))[:10]

        # Get price level (prefer non-unknown)
        price_levels = [r.price_level for r in reviews if r.price_level != PriceLevel.UNKNOWN]
        price_level = price_levels[0] if price_levels else PriceLevel.UNKNOWN

        return cls(
            name=first.name,
            city=first.city,
            country=first.country,
            address=first.address,
            avg_rating=avg_rating,
            total_review_count=total_reviews,
            ratings_by_source=ratings_by_source,
            price_level=price_level,
            cuisine_types=cuisines,
            all_review_highlights=highlights,
            popular_dishes=dishes,
            sources=[r.source.value for r in reviews],
            primary_source_url=next((r.source_url for r in reviews if r.source_url), None),
        )
