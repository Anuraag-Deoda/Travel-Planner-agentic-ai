"""LangGraph AgentState definition."""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from operator import add

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State shared across all agents in the travel planner graph.

    Uses TypedDict with Annotated types for LangGraph reducers:
    - add_messages: Appends messages to conversation history
    - add (operator): Accumulates list items (for attractions, etc.)
    """

    # Core conversation history
    messages: Annotated[list, add_messages]

    # User input
    user_request: str

    # Clarification Agent outputs
    clarification_needed: Optional[bool]  # Whether clarification questions are needed
    clarification_questions: Optional[list[dict]]  # Questions to ask the user
    clarification_answers: Optional[dict]  # User's answers to questions

    # User preferences (from clarification or explicit in request)
    origin_city: Optional[str]  # Where the traveler is coming from
    dietary_preferences: Optional[list[str]]  # Dietary restrictions/preferences
    travel_pace: Optional[str]  # fast, moderate, relaxed
    places_visited: Optional[list[str]]  # Places already visited (to avoid repeats)
    specific_destinations: Optional[list[str]]  # Specific cities/places requested

    # Travel dates (for real-time pricing)
    travel_start_date: Optional[str]  # ISO format: "2025-03-15"
    travel_end_date: Optional[str]  # ISO format: "2025-03-22"
    travel_date_flexibility: Optional[str]  # "specific", "flexible_week", "flexible_month"
    travel_date_description: Optional[str]  # Natural language for flexible dates

    # Planner Agent outputs
    trip_summary: Optional[dict]  # {destination, duration, budget_level, traveler_profile}
    city_allocations: Optional[list[dict]]  # [{city, country, days, visit_order, highlights}]

    # Geography Agent outputs
    route_validation: Optional[dict]  # {is_valid, optimized_order, warnings}
    route_segments: Optional[list[dict]]  # [{from_city, to_city, distance_km, travel_time_hours}]

    # Research Agent outputs - accumulates across multiple calls
    attractions: Annotated[list, add]  # [{city, name, description, duration_hours, ...}]
    hotels: Annotated[list, add]  # [{city, name, rating, photo_urls, ...}] from Google Places API
    research_sources: Annotated[list, add]  # URLs browsed

    # Food/Culture Agent outputs
    food_recommendations: Optional[list[dict]]  # [{city, dish, restaurant, budget_level}]
    cultural_tips: Optional[list[str]]

    # Transport Scraper outputs (real-time pricing)
    scraped_transport_prices: Optional[list[dict]]  # Raw scraped price data
    nearest_stations: Optional[dict]  # {city: {airport, train_station, bus_station}}

    # Transport/Budget Agent outputs
    transport_options: Optional[list[dict]]  # [{from, to, mode, cost, duration}]
    budget_breakdown: Optional[dict]  # {transport, accommodation, food, activities, total}

    # Critic Agent outputs
    validation_result: Optional[dict]  # {is_valid, issues, severity, requires_replanning}
    critic_feedback: Optional[str]  # Instructions for re-planning if needed

    # Control flow
    iteration_count: int  # Track re-planning iterations (max 3)
    current_city_index: int  # For iterating through cities in research

    # Final output
    final_itinerary: Optional[dict]  # Complete TravelItinerary as dict


def get_initial_state(user_request: str) -> AgentState:
    """Create initial state for a new planning session."""
    return AgentState(
        messages=[],
        user_request=user_request,
        # Clarification fields
        clarification_needed=None,
        clarification_questions=None,
        clarification_answers=None,
        origin_city=None,
        dietary_preferences=None,
        travel_pace=None,
        places_visited=None,
        specific_destinations=None,
        # Travel dates
        travel_start_date=None,
        travel_end_date=None,
        travel_date_flexibility=None,
        travel_date_description=None,
        # Planning fields
        trip_summary=None,
        city_allocations=None,
        route_validation=None,
        route_segments=None,
        attractions=[],
        hotels=[],
        research_sources=[],
        food_recommendations=None,
        cultural_tips=None,
        scraped_transport_prices=None,
        nearest_stations=None,
        transport_options=None,
        budget_breakdown=None,
        validation_result=None,
        critic_feedback=None,
        iteration_count=0,
        current_city_index=0,
        final_itinerary=None,
    )
