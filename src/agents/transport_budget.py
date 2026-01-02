"""Transport/Budget Agent - Calculates transport options and budget breakdown."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.config.constants import TRANSPORT_TEMPERATURE
from src.models.agent_outputs import TransportBudgetOutput
from src.models.state import AgentState


TRANSPORT_BUDGET_SYSTEM_PROMPT = """You are a travel logistics and budget expert. Your job is to:

1. Recommend the best transport options between cities
2. Suggest local transport within cities
3. Calculate a realistic budget breakdown

TRANSPORT GUIDELINES:

**Inter-city Transport:**
- Under 300km: Train or bus preferred (cheaper, scenic)
- 300-800km: Train is usually best; consider overnight options
- Over 800km: Flight recommended (save time)
- Always provide 2-3 options with pros/cons
- Include approximate costs and durations

**Local Transport:**
- Research what's available (metro, bus, taxi, rickshaw, etc.)
- Recommend the most practical option for tourists
- Note any passes or cards that save money
- Warn about common scams or issues

BUDGET BREAKDOWN:

Provide realistic estimates in USD for:
1. Inter-city transport (total for all segments)
2. Local transport (daily estimate × days)
3. Accommodation (daily estimate × nights)
4. Food (daily estimate × days)
5. Activities/entrance fees (based on planned attractions)
6. Miscellaneous (10-15% buffer)

Budget level expectations:
- BUDGET: $30-60/day total (hostels, street food, public transport)
- MID_RANGE: $80-150/day total (hotels, restaurants, comfort)
- LUXURY: $200+/day total (upscale hotels, fine dining, private transport)

Always recommend:
- Money-saving tips specific to the destination
- Best booking platforms for the region
- Timing tips (book in advance vs. on-the-spot)
"""


class TransportBudgetAgent(BaseAgent):
    """Transport/Budget Agent for logistics and cost estimation.

    This agent:
    - Recommends transport between cities
    - Suggests local transport options
    - Calculates detailed budget breakdown
    - Provides money-saving tips

    Uses GPT-4o-mini for efficient calculations.
    """

    agent_name = "transport_budget"

    def __init__(self, **kwargs):
        kwargs.setdefault("temperature", TRANSPORT_TEMPERATURE)
        super().__init__(**kwargs)

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Calculate transport options and budget for the trip.

        Args:
            state: Current graph state with route segments and trip details.

        Returns:
            State updates with transport_options and budget_breakdown.
        """
        city_allocations = state.get("city_allocations", [])
        route_segments = state.get("route_segments", [])
        trip_summary = state.get("trip_summary", {})
        attractions = state.get("attractions", [])

        if not city_allocations:
            return {
                "transport_options": [],
                "budget_breakdown": {},
            }

        total_days = trip_summary.get("total_days", sum(c.get("days", 1) for c in city_allocations))
        budget_level = trip_summary.get("budget_level", "mid_range")
        origin_city = state.get("origin_city")

        # Sort cities by visit order
        sorted_cities = sorted(city_allocations, key=lambda x: x.get("visit_order", 0))
        first_city = sorted_cities[0] if sorted_cities else None

        # Build context for the LLM
        cities_info = "\n".join(
            f"- {c['city']}, {c['country']}: {c['days']} days"
            for c in sorted_cities
        )

        routes_info = "\n".join(
            f"- {r['from_city']} → {r['to_city']}: {r['distance_km']}km, "
            f"recommended {r['recommended_transport']}, ~{r['travel_time_hours']}h"
            for r in route_segments
        ) if route_segments else "No inter-city travel"

        attractions_summary = f"{len(attractions)} attractions planned"
        if attractions:
            # Count attractions per city
            by_city = {}
            for a in attractions:
                city = a.get("city", "Unknown")
                by_city[city] = by_city.get(city, 0) + 1
            attractions_summary += " (" + ", ".join(f"{c}: {n}" for c, n in by_city.items()) + ")"

        # Build origin-to-destination section if origin is specified
        origin_section = ""
        if origin_city and first_city:
            origin_section = f"""
ORIGIN TO DESTINATION (IMPORTANT - Include this as the first transport segment):
- Traveler starts from: {origin_city}
- First destination: {first_city['city']}, {first_city['country']}
- Please provide flight/train options with:
  * Recommended option (mode, duration, cost estimate)
  * 2-3 alternatives
  * Booking tips (best platforms, when to book)
  * Departure timing suggestions
"""

        human_content = f"""Calculate transport options and budget for this trip:

TRIP OVERVIEW:
- Total days: {total_days}
- Budget level: {budget_level}
{f"- Origin city: {origin_city}" if origin_city else ""}

CITIES:
{cities_info}
{origin_section}
INTER-CITY ROUTES:
{routes_info}

ATTRACTIONS:
{attractions_summary}

Please provide:
1. {"Origin-to-first-city transport options (flights/trains) with costs and timings" if origin_city else ""}
{"2." if origin_city else "1."} Detailed transport options for each inter-city segment (2-3 options each)
{"3." if origin_city else "2."} Local transport recommendations for each city
{"4." if origin_city else "3."} Complete budget breakdown (include origin transport if applicable)
{"5." if origin_city else "4."} Money-saving tips specific to these destinations
"""

        structured_llm = self.get_structured_llm(TransportBudgetOutput)

        messages = [
            SystemMessage(content=TRANSPORT_BUDGET_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        result = await structured_llm.ainvoke(messages)

        # Convert to state update format
        transport_options = []
        for option in result.inter_city_options:
            # Check if this is origin-to-destination transport
            is_origin_transport = (
                origin_city and
                first_city and
                option.from_location.lower() == origin_city.lower()
            )

            transport_options.append({
                "from_location": option.from_location,
                "to_location": option.to_location,
                "is_origin_transport": is_origin_transport,
                "recommended": {
                    "mode": option.recommended.mode.value,
                    "duration_hours": option.recommended.duration_hours,
                    "estimated_cost_usd": option.recommended.estimated_cost_usd,
                    "notes": option.recommended.notes,
                },
                "alternatives": [
                    {
                        "mode": alt.mode.value,
                        "duration_hours": alt.duration_hours,
                        "estimated_cost_usd": alt.estimated_cost_usd,
                        "notes": alt.notes,
                    }
                    for alt in option.options if alt != option.recommended
                ],
                "reason": option.recommendation_reason,
            })

        # Convert local transport recommendations list to dict format
        local_transport_tips = {
            tip.city: tip.tips for tip in result.local_transport_recommendations
        }

        budget_breakdown = {
            "transport_inter_city": result.budget_breakdown.transport_inter_city,
            "transport_local": result.budget_breakdown.transport_local,
            "accommodation": result.budget_breakdown.accommodation,
            "food": result.budget_breakdown.food,
            "activities_entrance_fees": result.budget_breakdown.activities_entrance_fees,
            "miscellaneous": result.budget_breakdown.miscellaneous,
            "total": result.budget_breakdown.total,
            "currency": result.budget_breakdown.currency,
            "notes": result.budget_breakdown.notes,
            "money_saving_tips": result.money_saving_tips,
            "booking_tips": result.booking_tips,
            "local_transport_tips": local_transport_tips,
        }

        return {
            "transport_options": transport_options,
            "budget_breakdown": budget_breakdown,
        }
