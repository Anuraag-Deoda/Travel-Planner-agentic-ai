"""Node functions that wrap agent calls for the LangGraph workflow."""

import re
from datetime import datetime, date
from typing import Any

from langchain_core.messages import AIMessage

from src.models.state import AgentState
from src.agents.clarification import ClarificationAgent
from src.agents.planner import PlannerAgent
from src.agents.geography import GeographyAgent
from src.agents.research import ResearchAgent
from src.agents.food_culture import FoodCultureAgent
from src.agents.transport_budget import TransportBudgetAgent
from src.agents.critic import CriticAgent
from src.agents.transport_scraper import TransportScraperAgent


# Agent instances (created once, reused across invocations)
_clarification_agent: ClarificationAgent | None = None
_planner_agent: PlannerAgent | None = None
_geography_agent: GeographyAgent | None = None
_research_agent: ResearchAgent | None = None
_food_culture_agent: FoodCultureAgent | None = None
_transport_budget_agent: TransportBudgetAgent | None = None
_critic_agent: CriticAgent | None = None
_transport_scraper_agent: TransportScraperAgent | None = None


def _get_clarification() -> ClarificationAgent:
    """Get or create the Clarification agent instance."""
    global _clarification_agent
    if _clarification_agent is None:
        _clarification_agent = ClarificationAgent()
    return _clarification_agent


def _get_planner() -> PlannerAgent:
    """Get or create the Planner agent instance."""
    global _planner_agent
    if _planner_agent is None:
        _planner_agent = PlannerAgent()
    return _planner_agent


def _get_geography() -> GeographyAgent:
    """Get or create the Geography agent instance."""
    global _geography_agent
    if _geography_agent is None:
        _geography_agent = GeographyAgent()
    return _geography_agent


def _get_research() -> ResearchAgent:
    """Get or create the Research agent instance."""
    global _research_agent
    if _research_agent is None:
        _research_agent = ResearchAgent()
    return _research_agent


def _get_food_culture() -> FoodCultureAgent:
    """Get or create the Food/Culture agent instance."""
    global _food_culture_agent
    if _food_culture_agent is None:
        _food_culture_agent = FoodCultureAgent()
    return _food_culture_agent


def _get_transport_budget() -> TransportBudgetAgent:
    """Get or create the Transport/Budget agent instance."""
    global _transport_budget_agent
    if _transport_budget_agent is None:
        _transport_budget_agent = TransportBudgetAgent()
    return _transport_budget_agent


def _get_critic() -> CriticAgent:
    """Get or create the Critic agent instance."""
    global _critic_agent
    if _critic_agent is None:
        _critic_agent = CriticAgent()
    return _critic_agent


def _get_transport_scraper() -> TransportScraperAgent:
    """Get or create the Transport Scraper agent instance."""
    global _transport_scraper_agent
    if _transport_scraper_agent is None:
        _transport_scraper_agent = TransportScraperAgent()
    return _transport_scraper_agent


async def clarification_node(state: AgentState) -> dict[str, Any]:
    """Node function for the Clarification Agent.

    Analyzes user request and generates clarification questions if needed.
    """
    agent = _get_clarification()
    result = await agent.run(state)

    questions_count = len(result.get("clarification_questions", []))
    needs_clarification = result.get("clarification_needed", False)

    if needs_clarification:
        msg = f"[Clarification] Need {questions_count} answers before planning"
    else:
        msg = "[Clarification] Request is complete, proceeding to planning"

    message = AIMessage(content=msg, name="clarification")

    return {
        **result,
        "messages": [message],
    }


def parse_travel_dates(date_answer: str) -> dict:
    """Parse travel date answer into structured format.

    Handles:
    - Specific: "January 15-22, 2026", "2026-01-15 to 2026-01-22", "Jan 15-22 2026"
    - Flexible: "mid-January", "around February", "sometime in spring"

    Returns dict with start_date, end_date, flexibility, description.
    """
    result = {
        "start_date": None,
        "end_date": None,
        "flexibility": "specific",
        "description": date_answer,
    }

    if not date_answer:
        return result

    # Check for flexible date indicators
    flexible_indicators = [
        "around", "sometime", "mid-", "early", "late",
        "flexible", "approximately", "about", "roughly"
    ]

    is_flexible = any(ind in date_answer.lower() for ind in flexible_indicators)

    if is_flexible:
        result["flexibility"] = "flexible_week"
        return result

    # Month name mapping
    month_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
        'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
        'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
        'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
        'dec': 12, 'december': 12
    }

    # Try to parse specific dates
    # Pattern 1: "January 15-22, 2026" or "Jan 15-22 2026"
    pattern1 = r"(\w+)\s+(\d{1,2})\s*[-–to]+\s*(\d{1,2}),?\s*(\d{4})"
    match = re.search(pattern1, date_answer, re.IGNORECASE)
    if match:
        try:
            month_str, start_day, end_day, year = match.groups()
            month = month_map.get(month_str.lower(), 1)
            result["start_date"] = date(int(year), month, int(start_day)).isoformat()
            result["end_date"] = date(int(year), month, int(end_day)).isoformat()
            return result
        except (ValueError, KeyError):
            pass

    # Pattern 2: "2026-01-15 to 2026-01-22" (ISO format range)
    pattern2 = r"(\d{4})-(\d{2})-(\d{2})\s*(?:to|-|–)\s*(\d{4})-(\d{2})-(\d{2})"
    match = re.search(pattern2, date_answer)
    if match:
        try:
            result["start_date"] = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            result["end_date"] = f"{match.group(4)}-{match.group(5)}-{match.group(6)}"
            return result
        except (ValueError, IndexError):
            pass

    # Pattern 3: Single date with duration "January 15, 2026 for 7 days"
    pattern3 = r"(\w+)\s+(\d{1,2}),?\s*(\d{4})"
    match = re.search(pattern3, date_answer, re.IGNORECASE)
    if match:
        try:
            month_str, day, year = match.groups()
            month = month_map.get(month_str.lower(), 1)
            result["start_date"] = date(int(year), month, int(day)).isoformat()
            # End date will be calculated based on trip duration later
            return result
        except (ValueError, KeyError):
            pass

    return result


async def process_answers_node(state: AgentState) -> dict[str, Any]:
    """Process clarification answers and enrich the user request.

    Takes answers from clarification_answers and adds them to state fields.
    """
    answers = state.get("clarification_answers", {})
    original_request = state["user_request"]

    # Build enhanced request with answers - make it very explicit for the planner
    enriched_parts = [original_request]

    # Extract and set individual fields from answers
    origin_city = answers.get("origin_city")
    dietary = answers.get("dietary")
    travel_pace = answers.get("travel_pace")
    visited = answers.get("visited_places")
    destinations = answers.get("specific_destinations")

    # Parse travel dates
    travel_dates_answer = answers.get("travel_dates")
    travel_start_date = None
    travel_end_date = None
    travel_date_flexibility = None
    travel_date_description = None

    if travel_dates_answer:
        parsed_dates = parse_travel_dates(travel_dates_answer)
        travel_start_date = parsed_dates.get("start_date")
        travel_end_date = parsed_dates.get("end_date")
        travel_date_flexibility = parsed_dates.get("flexibility")
        travel_date_description = parsed_dates.get("description")

        # Add to enriched request
        if travel_start_date and travel_end_date:
            enriched_parts.append(f"\nIMPORTANT - Travel dates: {travel_start_date} to {travel_end_date}")
        elif travel_date_description:
            enriched_parts.append(f"\nIMPORTANT - Travel timing: {travel_date_description} (flexible)")

    if origin_city:
        enriched_parts.append(f"\nIMPORTANT - Traveling from: {origin_city}")
    if destinations:
        # Make destinations VERY explicit
        enriched_parts.append(f"\nIMPORTANT - MUST visit these specific cities: {destinations}")
        enriched_parts.append("Do NOT substitute different cities. Plan ONLY for the cities listed above.")
    if dietary:
        enriched_parts.append(f"\nDietary preferences: {dietary}")
    if travel_pace:
        enriched_parts.append(f"\nTravel pace preference: {travel_pace}")
    if visited:
        enriched_parts.append(f"\nAlready visited (avoid these): {visited}")

    message = AIMessage(
        content=f"[Process Answers] Enriched request with user preferences (dates: {travel_dates_answer}, destinations: {destinations})",
        name="process_answers",
    )

    # Parse destinations into a list if it's a string
    destinations_list = None
    if destinations:
        if isinstance(destinations, str):
            # Split by common separators
            destinations_list = [d.strip() for d in destinations.replace(" and ", ",").replace("(if possible)", "").split(",") if d.strip()]
        else:
            destinations_list = destinations

    return {
        "user_request": "\n".join(enriched_parts),
        "origin_city": origin_city,
        "dietary_preferences": [dietary] if dietary and isinstance(dietary, str) else dietary,
        "travel_pace": travel_pace,
        "places_visited": [visited] if visited and isinstance(visited, str) else visited,
        "specific_destinations": destinations_list,
        # Travel date fields
        "travel_start_date": travel_start_date,
        "travel_end_date": travel_end_date,
        "travel_date_flexibility": travel_date_flexibility,
        "travel_date_description": travel_date_description,
        "messages": [message],
    }


async def planner_node(state: AgentState) -> dict[str, Any]:
    """Node function for the Planner Agent.

    Parses user request and allocates days to cities.
    """
    agent = _get_planner()
    result = await agent.run(state)

    # Add a message to track what happened
    iteration = state.get("iteration_count", 0)
    cities = [c["city"] for c in result.get("city_allocations", [])]

    message = AIMessage(
        content=f"[Planner] {'Re-planned' if iteration > 0 else 'Planned'} trip with cities: {', '.join(cities)}",
        name="planner",
    )

    return {
        **result,
        "messages": [message],
    }


async def geography_node(state: AgentState) -> dict[str, Any]:
    """Node function for the Geography/Routing Agent.

    Validates and optimizes the route between cities.
    """
    agent = _get_geography()
    result = await agent.run(state)

    # Add a message to track what happened
    validation = result.get("route_validation", {})
    is_valid = validation.get("is_valid", False)
    route_changed = validation.get("route_changed", False)

    if route_changed:
        new_order = validation.get("optimized_order", [])
        msg = f"[Geography] Route optimized to: {' → '.join(new_order)}"
    elif is_valid:
        msg = "[Geography] Route validated successfully"
    else:
        warnings = validation.get("warnings", [])
        msg = f"[Geography] Route has issues: {', '.join(warnings[:2])}"

    message = AIMessage(content=msg, name="geography")

    return {
        **result,
        "messages": [message],
    }


async def critic_node(state: AgentState) -> dict[str, Any]:
    """Node function for the Critic/Validator Agent.

    Validates the complete plan and decides if re-planning is needed.
    """
    agent = _get_critic()
    result = await agent.run(state)

    # Add a message to track what happened
    validation = result.get("validation_result", {})
    is_valid = validation.get("is_valid", False)
    score = validation.get("overall_score", 0)
    requires_replanning = validation.get("requires_replanning", False)

    if requires_replanning:
        issues = validation.get("issues", [])
        critical_count = sum(1 for i in issues if i.get("severity") in ("critical", "high"))
        msg = f"[Critic] Plan needs revision (score: {score}/100, {critical_count} critical/high issues)"
    elif is_valid:
        msg = f"[Critic] Plan approved (score: {score}/100)"
    else:
        msg = f"[Critic] Plan has issues but proceeding (score: {score}/100)"

    message = AIMessage(content=msg, name="critic")

    return {
        **result,
        "messages": [message],
    }


async def finalize_node(state: AgentState) -> dict[str, Any]:
    """Node function to finalize the itinerary.

    Assembles the final itinerary from all agent outputs.
    """
    trip_summary = state.get("trip_summary", {})
    city_allocations = state.get("city_allocations", [])
    route_segments = state.get("route_segments", [])
    attractions = state.get("attractions", [])
    hotels = state.get("hotels", [])
    food_recommendations = state.get("food_recommendations", [])
    transport_options = state.get("transport_options", [])
    budget_breakdown = state.get("budget_breakdown", {})
    validation_result = state.get("validation_result", {})

    # Sort cities by visit order
    sorted_cities = sorted(city_allocations, key=lambda x: x.get("visit_order", 0))

    # Build daily plans
    daily_plans = []
    day_number = 1

    for city_info in sorted_cities:
        city = city_info["city"]
        days_in_city = city_info.get("days", 1)

        # Get attractions for this city - deduplicate by name
        city_attractions_raw = [a for a in attractions if a.get("city") == city]
        seen_names = set()
        city_attractions = []
        for a in city_attractions_raw:
            name = a.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                city_attractions.append(a)

        # Limit to reasonable number of attractions per city (4-5 per day max)
        max_attractions = days_in_city * 4
        city_attractions = city_attractions[:max_attractions]

        # Get food for this city, categorized by meal type
        city_food = [f for f in food_recommendations if f.get("city") == city]
        city_breakfasts = [f for f in city_food if f.get("meal_type") == "breakfast"]
        city_lunches = [f for f in city_food if f.get("meal_type") == "lunch"]
        city_dinners = [f for f in city_food if f.get("meal_type") == "dinner"]

        # Calculate how to distribute attractions across days
        total_attractions = len(city_attractions)
        base_per_day = total_attractions // days_in_city if days_in_city > 0 else 0
        extra = total_attractions % days_in_city if days_in_city > 0 else 0
        attraction_idx = 0

        for day_offset in range(days_in_city):
            day_plan = {
                "day_number": day_number,
                "city": city,
                "theme": f"Day {day_offset + 1} in {city}",
                "activities": [],
                "daily_budget_usd": budget_breakdown.get("total", 0) / trip_summary.get("total_days", 1) if budget_breakdown else 0,
            }

            # Add breakfast
            if day_offset < len(city_breakfasts):
                breakfast = city_breakfasts[day_offset]
                day_plan["activities"].append({
                    "time_slot": "08:00 - 09:00",
                    "activity_type": "meal",
                    "title": f"Breakfast: {breakfast.get('restaurant_name', 'Local breakfast spot')}",
                    "meal": breakfast,
                })

            # Get attractions for this specific day
            attractions_today = base_per_day + (1 if day_offset < extra else 0)
            day_attractions = city_attractions[attraction_idx:attraction_idx + attractions_today]
            attraction_idx += attractions_today

            # Limit to max 4 attractions per day for realistic scheduling
            day_attractions = day_attractions[:4]

            # Split into morning (2 max) and afternoon (2 max)
            morning_attractions = day_attractions[:2]
            afternoon_attractions = day_attractions[2:4]

            # Morning activities (9:00 onwards)
            current_hour = 9
            for attr in morning_attractions:
                duration = max(1, int(attr.get("estimated_duration_hours", 2)))
                end_hour = current_hour + duration
                if end_hour > 12:
                    end_hour = 12
                day_plan["activities"].append({
                    "time_slot": f"{current_hour:02d}:00 - {end_hour:02d}:00",
                    "activity_type": "attraction",
                    "title": attr.get("name", "Activity"),
                    "attraction": attr,
                })
                current_hour = end_hour + 1  # Add gap between activities
                if current_hour > 12:
                    break

            # Add lunch
            if day_offset < len(city_lunches):
                lunch = city_lunches[day_offset]
                day_plan["activities"].append({
                    "time_slot": "12:30 - 14:00",
                    "activity_type": "meal",
                    "title": f"Lunch: {lunch.get('restaurant_name', 'Local restaurant')}",
                    "meal": lunch,
                })

            # Afternoon activities (14:00 onwards)
            current_hour = 14
            for attr in afternoon_attractions:
                duration = max(1, int(attr.get("estimated_duration_hours", 2)))
                end_hour = current_hour + duration
                if end_hour > 18:
                    end_hour = 18
                day_plan["activities"].append({
                    "time_slot": f"{current_hour:02d}:00 - {end_hour:02d}:00",
                    "activity_type": "attraction",
                    "title": attr.get("name", "Activity"),
                    "attraction": attr,
                })
                current_hour = end_hour + 1  # Add gap between activities
                if current_hour > 18:
                    break

            # Add dinner
            if day_offset < len(city_dinners):
                dinner = city_dinners[day_offset]
                day_plan["activities"].append({
                    "time_slot": "19:00 - 21:00",
                    "activity_type": "meal",
                    "title": f"Dinner: {dinner.get('restaurant_name', 'Local restaurant')}",
                    "meal": dinner,
                })

            daily_plans.append(day_plan)
            day_number += 1

    # Build comprehensive transport section
    origin_transport = None
    inter_city_transport = []

    for t_opt in transport_options:
        transport_entry = {
            "from_location": t_opt.get("from_location"),
            "to_location": t_opt.get("to_location"),
            "recommended": t_opt.get("recommended", {}),
            "alternatives": t_opt.get("alternatives", []),
            "reason": t_opt.get("reason", ""),
        }

        if t_opt.get("is_origin_transport"):
            origin_transport = transport_entry
        else:
            # Enrich with distance from route_segments if available
            matching_segment = next(
                (s for s in route_segments
                 if s.get("from_city") == t_opt.get("from_location")
                 and s.get("to_city") == t_opt.get("to_location")),
                None
            )
            if matching_segment:
                transport_entry["distance_km"] = matching_segment.get("distance_km")
            inter_city_transport.append(transport_entry)

    # Get local transport tips from budget breakdown
    local_transport_tips = budget_breakdown.get("local_transport_tips", {}) if budget_breakdown else {}

    # Build final itinerary
    final_itinerary = {
        "trip_title": f"{trip_summary.get('total_days', len(sorted_cities))}-Day {sorted_cities[0].get('country', 'Trip') if sorted_cities else 'Trip'}",
        "destination_summary": trip_summary.get("understanding", ""),
        "total_days": trip_summary.get("total_days", len(daily_plans)),
        "travelers_count": 1,  # Default, could be extracted from trip_summary
        "traveler_profile": trip_summary.get("traveler_profile", "solo"),
        "budget_level": trip_summary.get("budget_level", "mid_range"),
        "total_estimated_cost_usd": budget_breakdown.get("total", 0) if budget_breakdown else 0,
        "cities_visited": [c["city"] for c in sorted_cities],
        "daily_plans": daily_plans,
        "origin_transport": origin_transport,
        "inter_city_transport": inter_city_transport,
        "local_transport_tips": local_transport_tips,
        "budget_breakdown": {
            "transport_inter_city": budget_breakdown.get("transport_inter_city", 0),
            "transport_local": budget_breakdown.get("transport_local", 0),
            "accommodation": budget_breakdown.get("accommodation", 0),
            "food": budget_breakdown.get("food", 0),
            "activities": budget_breakdown.get("activities_entrance_fees", 0),
            "miscellaneous": budget_breakdown.get("miscellaneous", 0),
            "total": budget_breakdown.get("total", 0),
            "money_saving_tips": budget_breakdown.get("money_saving_tips", []),
        } if budget_breakdown else {},
        "cultural_tips": state.get("cultural_tips", []),
        "packing_suggestions": [],
        "warnings": validation_result.get("final_recommendations", []),
        "sources_consulted": state.get("research_sources", []),
        "hotels": hotels,
    }

    message = AIMessage(
        content=f"[Finalize] Itinerary complete: {final_itinerary['trip_title']}",
        name="finalize",
    )

    return {
        "final_itinerary": final_itinerary,
        "messages": [message],
    }


async def research_node(state: AgentState) -> dict[str, Any]:
    """Node function for the Research/Browser Agent.

    Browses the web to find attractions and current information.
    """
    agent = _get_research()
    result = await agent.run(state)

    # Add a message to track what happened
    attractions_count = len(result.get("attractions", []))
    cities = list(set(a.get("city", "") for a in result.get("attractions", [])))

    message = AIMessage(
        content=f"[Research] Found {attractions_count} attractions in {', '.join(cities)}",
        name="research",
    )

    return {
        **result,
        "messages": [message],
    }


async def food_culture_node(state: AgentState) -> dict[str, Any]:
    """Node function for the Food/Culture Agent.

    Provides food recommendations and cultural guidance.
    """
    agent = _get_food_culture()
    result = await agent.run(state)

    # Add a message to track what happened
    food_count = len(result.get("food_recommendations", []))
    tips_count = len(result.get("cultural_tips", []))

    message = AIMessage(
        content=f"[Food/Culture] Generated {food_count} food recommendations and {tips_count} cultural tips",
        name="food_culture",
    )

    return {
        **result,
        "messages": [message],
    }


async def transport_scraper_node(state: AgentState) -> dict[str, Any]:
    """Node function for the Transport Scraper Agent.

    Scrapes real transport prices before budget calculation.
    """
    agent = _get_transport_scraper()
    result = await agent.run(state)

    prices_count = len(result.get("scraped_transport_prices", []))
    stations_count = len(result.get("nearest_stations", {}))

    message = AIMessage(
        content=f"[Transport Scraper] Found {prices_count} real prices, {stations_count} station lookups",
        name="transport_scraper",
    )

    return {
        **result,
        "messages": [message],
    }


async def transport_budget_node(state: AgentState) -> dict[str, Any]:
    """Node function for the Transport/Budget Agent.

    Calculates transport options and budget breakdown.
    """
    agent = _get_transport_budget()
    result = await agent.run(state)

    # Add a message to track what happened
    budget = result.get("budget_breakdown", {})
    total = budget.get("total", 0)

    message = AIMessage(
        content=f"[Transport/Budget] Estimated total trip cost: ${total} USD",
        name="transport_budget",
    )

    return {
        **result,
        "messages": [message],
    }
