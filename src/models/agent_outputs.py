"""Pydantic models for structured agent outputs."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .itinerary import Attraction, BudgetLevel, Meal, TransportMode, TransportSegment


class IssueSeverity(str, Enum):
    """Severity level of validation issues."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Planner Agent Output
# ============================================================================


class CityAllocation(BaseModel):
    """Allocation of days to a city."""

    city: str
    country: str
    days: int = Field(ge=1, description="Number of days to spend")
    visit_order: int = Field(ge=1, description="Order in the trip sequence")
    highlights: list[str] = Field(
        default_factory=list,
        description="Key attractions/experiences in this city",
    )
    reasoning: str = Field(description="Why this allocation makes sense")


class PlannerOutput(BaseModel):
    """Output from the Planner Agent."""

    trip_understanding: str = Field(
        description="Summary of understood trip requirements"
    )
    total_days: int = Field(ge=1)
    budget_level: BudgetLevel
    traveler_profile: str  # solo, couple, family, group
    travel_style: str  # adventure, relaxed, cultural, mixed
    city_allocations: list[CityAllocation]
    overall_strategy: str = Field(
        description="High-level approach to the trip"
    )


# ============================================================================
# Geography Agent Output
# ============================================================================


class RouteSegment(BaseModel):
    """A segment of the travel route between cities."""

    from_city: str
    to_city: str
    distance_km: float
    recommended_transport: TransportMode
    travel_time_hours: float
    is_feasible: bool = True
    issues: list[str] = Field(default_factory=list)


class GeographyOutput(BaseModel):
    """Output from the Geography/Routing Agent."""

    route_is_valid: bool
    original_order: list[str]
    optimized_order: list[str]
    route_changed: bool = Field(
        description="Whether the order was modified from planner's suggestion"
    )
    route_segments: list[RouteSegment]
    total_travel_time_hours: float
    total_distance_km: float
    suggestions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ============================================================================
# Research Agent Output
# ============================================================================


class ResearchOutput(BaseModel):
    """Output from the Research/Browser Agent for a single city."""

    city: str
    attractions_found: list[Attraction]
    current_events: list[str] = Field(
        default_factory=list,
        description="Events happening during the visit period",
    )
    seasonal_notes: Optional[str] = None
    weather_info: Optional[str] = None
    local_tips: list[str] = Field(default_factory=list)
    sources_browsed: list[str] = Field(default_factory=list)


# ============================================================================
# Food/Culture Agent Output
# ============================================================================


class FoodCultureOutput(BaseModel):
    """Output from the Food/Culture Agent."""

    city: str
    must_try_dishes: list[str]
    restaurant_recommendations: list[Meal]
    street_food_tips: Optional[str] = None
    food_safety_notes: Optional[str] = None
    cultural_dos: list[str] = Field(default_factory=list)
    cultural_donts: list[str] = Field(default_factory=list)
    dress_code_notes: Optional[str] = None
    local_customs: list[str] = Field(default_factory=list)
    language_tips: Optional[str] = None


# ============================================================================
# Transport/Budget Agent Output
# ============================================================================


class CityTransportTips(BaseModel):
    """Local transport tips for a city."""

    city: str
    tips: list[str] = Field(default_factory=list)


class TransportOption(BaseModel):
    """A transport option between two points."""

    from_location: str
    to_location: str
    options: list[TransportSegment]
    recommended: TransportSegment
    recommendation_reason: str


class BudgetBreakdown(BaseModel):
    """Detailed budget breakdown."""

    transport_inter_city: float
    transport_local: float
    accommodation: float
    food: float
    activities_entrance_fees: float
    miscellaneous: float
    total: float
    currency: str = "USD"
    notes: list[str] = Field(default_factory=list)


class TransportBudgetOutput(BaseModel):
    """Output from the Transport/Budget Agent."""

    inter_city_options: list[TransportOption]
    local_transport_recommendations: list[CityTransportTips]
    budget_breakdown: BudgetBreakdown
    money_saving_tips: list[str] = Field(default_factory=list)
    booking_tips: list[str] = Field(default_factory=list)


# ============================================================================
# Critic Agent Output
# ============================================================================


class ValidationIssue(BaseModel):
    """A specific issue found during validation."""

    category: str  # timing, budget, logistics, safety, feasibility
    description: str
    severity: IssueSeverity
    affected_days: list[int] = Field(default_factory=list)
    affected_cities: list[str] = Field(default_factory=list)
    suggested_fix: Optional[str] = None


class CriticOutput(BaseModel):
    """Output from the Critic/Validator Agent."""

    is_valid: bool
    overall_score: float = Field(ge=0, le=100, description="Quality score 0-100")
    issues: list[ValidationIssue]
    requires_replanning: bool
    replan_focus: Optional[str] = Field(
        default=None,
        description="What aspect needs replanning (city_allocation, timing, etc.)",
    )
    replan_instructions: Optional[str] = Field(
        default=None,
        description="Specific instructions for the planner if replanning is needed",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="What's good about the plan",
    )
    final_recommendations: list[str] = Field(
        default_factory=list,
        description="Final suggestions to improve the trip",
    )


# ============================================================================
# Clarification Agent Output
# ============================================================================


class ClarificationQuestion(BaseModel):
    """A single clarification question to ask the user."""

    question_id: str = Field(description="Unique identifier (e.g., 'travel_dates', 'origin_city', 'dietary')")
    question_text: str = Field(description="The question to display to the user")
    question_type: str = Field(
        description="Type: travel_dates, origin_city, specific_destinations, visited_places, dietary, travel_pace"
    )
    required: bool = Field(default=True, description="Whether answer is required")
    options: list[str] = Field(
        default_factory=list,
        description="Predefined options if applicable (empty for free-text)",
    )
    allow_multiple: bool = Field(
        default=False,
        description="Whether multiple options can be selected",
    )


class InferredTripInfo(BaseModel):
    """Information inferred from the user's request."""

    duration_days: Optional[int] = Field(default=None, description="Trip duration if mentioned")
    destination_country: Optional[str] = Field(default=None, description="Country if mentioned")
    destination_state: Optional[str] = Field(default=None, description="State/region if mentioned")
    destination_cities: list[str] = Field(default_factory=list, description="Specific cities if mentioned")
    budget_level: Optional[str] = Field(default=None, description="Budget level if mentioned")
    travel_style: Optional[str] = Field(default=None, description="Travel style if mentioned")
    # Travel dates inferred from request
    travel_start_date: Optional[str] = Field(default=None, description="Start date if mentioned (ISO format)")
    travel_end_date: Optional[str] = Field(default=None, description="End date if mentioned (ISO format)")
    has_specific_dates: bool = Field(default=False, description="Whether specific dates were mentioned")


class ClarificationOutput(BaseModel):
    """Output from the Clarification Agent."""

    needs_clarification: bool = Field(
        description="Whether clarification is needed before planning"
    )
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    inferred_info: InferredTripInfo = Field(
        default_factory=InferredTripInfo,
        description="Information already inferred from the request",
    )
    ready_to_plan: bool = Field(
        default=False,
        description="Whether enough information is available to proceed with planning",
    )
