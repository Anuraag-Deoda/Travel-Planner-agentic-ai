"""Models for scraped transport price data."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PriceSource(str, Enum):
    """Source of transport price data."""

    GOOGLE_FLIGHTS = "google_flights"
    ROME2RIO = "rome2rio"
    TWELVE_GO_ASIA = "12go_asia"
    REDBUS = "redbus"
    TRAINMAN = "trainman"
    MAKEMYTRIP = "makemytrip"
    LLM_ESTIMATE = "llm_estimate"


class TransportAvailability(str, Enum):
    """Availability status of a transport option."""

    AVAILABLE = "available"
    LIMITED = "limited"
    SOLD_OUT = "sold_out"
    UNKNOWN = "unknown"


class ScrapedTransportPrice(BaseModel):
    """Result from scraping a transport booking site."""

    source: PriceSource
    mode: str  # flight, train, bus, ferry
    from_location: str
    to_location: str
    from_station: Optional[str] = None  # Actual station/airport name
    to_station: Optional[str] = None
    travel_date: str  # ISO format
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    duration_hours: Optional[float] = None
    price_usd: float
    price_local: Optional[float] = None
    currency_local: Optional[str] = None
    operator: Optional[str] = None  # Airline, bus company, etc.
    class_type: Optional[str] = None  # economy, business, sleeper, AC
    availability: TransportAvailability = TransportAvailability.UNKNOWN
    booking_url: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)

    # Alternative dates with prices
    alternative_dates: list[dict] = Field(
        default_factory=list,
        description="Cheaper alternative dates: [{date: str, price: float}]",
    )


class NearestStation(BaseModel):
    """Nearest airport/station info for a city without direct routes."""

    city: str
    country: str
    airport_code: Optional[str] = None
    airport_name: Optional[str] = None
    airport_distance_km: Optional[float] = None
    train_station: Optional[str] = None
    train_station_distance_km: Optional[float] = None
    bus_station: Optional[str] = None
    notes: Optional[str] = None


class TransportSearchResult(BaseModel):
    """Result from a transport price scraping operation."""

    from_location: str
    to_location: str
    travel_date: Optional[str] = None
    source: PriceSource
    options: list[ScrapedTransportPrice] = Field(default_factory=list)
    alternative_dates: list[dict] = Field(default_factory=list)
    error: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
