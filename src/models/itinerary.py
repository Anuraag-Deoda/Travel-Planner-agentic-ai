"""Pydantic models for the final travel itinerary output."""

from datetime import date, time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BudgetLevel(str, Enum):
    """Budget tier for the trip."""

    BUDGET = "budget"
    MID_RANGE = "mid_range"
    LUXURY = "luxury"


class TransportMode(str, Enum):
    """Mode of transportation."""

    FLIGHT = "flight"
    TRAIN = "train"
    BUS = "bus"
    CAR = "car"
    FERRY = "ferry"
    WALKING = "walking"
    METRO = "metro"
    TAXI = "taxi"
    AUTO_RICKSHAW = "auto_rickshaw"


class TravelerProfile(str, Enum):
    """Type of traveler/group."""

    SOLO = "solo"
    COUPLE = "couple"
    FAMILY = "family"
    GROUP = "group"


class Attraction(BaseModel):
    """A tourist attraction or point of interest."""

    name: str
    city: str
    description: Optional[str] = None
    category: str  # museum, landmark, nature, market, temple, etc.
    estimated_duration_hours: float
    address: Optional[str] = None
    opening_hours: Optional[str] = None
    entrance_fee_usd: Optional[float] = None
    booking_required: bool = False
    tips: Optional[str] = None
    source_url: Optional[str] = None


class Meal(BaseModel):
    """A meal recommendation."""

    meal_type: str  # breakfast, lunch, dinner, snack
    restaurant_name: Optional[str] = None
    cuisine_type: str
    budget_level: BudgetLevel
    estimated_cost_usd: float
    address: Optional[str] = None
    must_try_dishes: list[str] = Field(default_factory=list)
    dietary_notes: Optional[str] = None


class TransportSegment(BaseModel):
    """A transportation segment between locations."""

    mode: TransportMode
    from_location: str
    to_location: str
    departure_time: Optional[str] = None
    duration_hours: float
    estimated_cost_usd: float
    booking_link: Optional[str] = None
    notes: Optional[str] = None


class DayActivity(BaseModel):
    """A single activity within a day."""

    time_slot: str  # "09:00-12:00" or "morning", "afternoon", "evening"
    activity_type: str  # attraction, meal, transport, free_time, check_in
    title: str  # Brief title for the activity
    attraction: Optional[Attraction] = None
    meal: Optional[Meal] = None
    transport: Optional[TransportSegment] = None
    notes: Optional[str] = None


class DayPlan(BaseModel):
    """Complete plan for a single day."""

    day_number: int
    date: Optional[date] = None
    city: str
    theme: Optional[str] = None  # "Historical exploration", "Beach day", etc.
    activities: list[DayActivity]
    accommodation: Optional[str] = None
    daily_budget_usd: float
    weather_note: Optional[str] = None


class TravelItinerary(BaseModel):
    """Complete travel itinerary - the final output model."""

    trip_title: str
    destination_summary: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    total_days: int
    travelers_count: int = 1
    traveler_profile: TravelerProfile = TravelerProfile.SOLO
    budget_level: BudgetLevel
    total_estimated_cost_usd: float
    cities_visited: list[str]
    daily_plans: list[DayPlan]
    inter_city_transport: list[TransportSegment] = Field(default_factory=list)
    cultural_tips: list[str] = Field(default_factory=list)
    packing_suggestions: list[str] = Field(default_factory=list)
    emergency_info: Optional[dict] = None
    warnings: list[str] = Field(default_factory=list)
    sources_consulted: list[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "trip_title": "5-Day Rajasthan Heritage Circuit",
                "destination_summary": "Exploring the royal heritage of Rajasthan through Udaipur, Jodhpur, and Jaipur",
                "total_days": 5,
                "budget_level": "mid_range",
                "cities_visited": ["Udaipur", "Jodhpur", "Jaipur"],
            }
        }
