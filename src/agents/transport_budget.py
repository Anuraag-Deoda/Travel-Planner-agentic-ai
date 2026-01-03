"""Transport/Budget Agent - Calculates transport options and budget breakdown."""

from typing import Any, Optional

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

REAL-TIME PRICE DATA:
When real prices are provided from scraped booking sites:
- PRIORITIZE real prices over estimates - these are current market prices
- Use the source (google_flights, rome2rio, redbus, trainman, etc.) to indicate price reliability
- If cheaper alternative dates are available, include them in your recommendation
- If prices vary significantly by operator, recommend the best value option
- Always show both real price (if available) and your estimate for comparison
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
        scraped_transport_prices = state.get("scraped_transport_prices", [])
        nearest_stations = state.get("nearest_stations", {})
        travel_start_date = state.get("travel_start_date")
        travel_end_date = state.get("travel_end_date")

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

        # Build real-time prices section from scraped data
        real_prices_section = self._build_real_prices_section(
            scraped_transport_prices, nearest_stations
        )

        # Build travel dates section
        dates_section = ""
        if travel_start_date:
            dates_section = f"\n- Travel dates: {travel_start_date}"
            if travel_end_date:
                dates_section += f" to {travel_end_date}"

        human_content = f"""Calculate transport options and budget for this trip:

TRIP OVERVIEW:
- Total days: {total_days}
- Budget level: {budget_level}
{f"- Origin city: {origin_city}" if origin_city else ""}{dates_section}

CITIES:
{cities_info}
{origin_section}
INTER-CITY ROUTES:
{routes_info}
{real_prices_section}
ATTRACTIONS:
{attractions_summary}

Please provide:
1. {"Origin-to-first-city transport options (flights/trains) with costs and timings" if origin_city else ""}
{"2." if origin_city else "1."} Detailed transport options for each inter-city segment (2-3 options each)
{"3." if origin_city else "2."} Local transport recommendations for each city
{"4." if origin_city else "3."} Complete budget breakdown (include origin transport if applicable)
{"5." if origin_city else "4."} Money-saving tips specific to these destinations

NOTE: When real prices are provided above, USE THEM as primary cost reference. Include cheaper date alternatives if available.
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

            # Find matching scraped prices for this segment
            segment_scraped = self._find_scraped_prices_for_segment(
                option.from_location,
                option.to_location,
                scraped_transport_prices,
            )

            # Get real price and cheaper dates if available
            real_price_info = self._get_best_real_price(segment_scraped)
            cheaper_dates = self._get_cheaper_dates(segment_scraped)

            transport_option = {
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
            }

            # Add real price info if available
            if real_price_info:
                transport_option["real_price"] = real_price_info
            if cheaper_dates:
                transport_option["cheaper_dates"] = cheaper_dates

            transport_options.append(transport_option)

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

    def _build_real_prices_section(
        self,
        scraped_prices: list[dict],
        nearest_stations: dict,
    ) -> str:
        """Build the real-time prices section for the LLM prompt."""
        if not scraped_prices:
            return ""

        lines = ["\nREAL-TIME PRICES (from booking sites):"]

        # Group prices by route
        routes: dict[str, list[dict]] = {}
        for price in scraped_prices:
            route_key = f"{price.get('from_location', '')} → {price.get('to_location', '')}"
            if route_key not in routes:
                routes[route_key] = []
            routes[route_key].append(price)

        for route, prices in routes.items():
            lines.append(f"\n{route}:")
            for p in prices[:5]:  # Limit to top 5 per route
                mode = p.get("mode", "unknown")
                price_usd = p.get("price_usd")
                source = p.get("source", "")
                operator = p.get("operator", "")
                duration = p.get("duration")
                departure = p.get("departure_time", "")

                if price_usd:
                    price_str = f"  - {mode.upper()}: ${price_usd:.0f}"
                    if operator:
                        price_str += f" ({operator})"
                    if duration:
                        price_str += f", {duration}"
                    if departure:
                        price_str += f", dep: {departure}"
                    price_str += f" [via {source}]"
                    lines.append(price_str)

                # Add alternative dates if available
                alt_dates = p.get("alternative_dates", [])
                if alt_dates:
                    cheaper = [d for d in alt_dates if d.get("price_usd", 999999) < (price_usd or 999999)]
                    if cheaper:
                        cheaper_str = ", ".join(
                            f"{d.get('date')}: ${d.get('price_usd'):.0f}" for d in cheaper[:3]
                        )
                        lines.append(f"    ↳ Cheaper dates: {cheaper_str}")

        # Add nearest station info
        if nearest_stations:
            lines.append("\nNEAREST STATIONS/AIRPORTS:")
            for city, info in nearest_stations.items():
                if info:
                    airport = info.get("airport_name") or info.get("airport_code")
                    train = info.get("train_station")
                    if airport:
                        dist = info.get("airport_distance_km")
                        lines.append(f"  - {city}: Airport '{airport}'" + (f" ({dist}km away)" if dist else ""))
                    if train:
                        dist = info.get("train_station_distance_km")
                        lines.append(f"  - {city}: Train station '{train}'" + (f" ({dist}km away)" if dist else ""))

        return "\n".join(lines)

    def _find_scraped_prices_for_segment(
        self,
        from_loc: str,
        to_loc: str,
        scraped_prices: list[dict],
    ) -> list[dict]:
        """Find scraped prices matching a transport segment."""
        from_lower = from_loc.lower()
        to_lower = to_loc.lower()

        matching = []
        for p in scraped_prices:
            p_from = (p.get("from_location") or "").lower()
            p_to = (p.get("to_location") or "").lower()

            # Match either direction or partial city name match
            if (from_lower in p_from or p_from in from_lower) and \
               (to_lower in p_to or p_to in to_lower):
                matching.append(p)

        return matching

    def _get_best_real_price(self, scraped_prices: list[dict]) -> Optional[dict]:
        """Get the best real price from scraped data."""
        if not scraped_prices:
            return None

        # Find lowest price with valid data
        valid_prices = [p for p in scraped_prices if p.get("price_usd")]
        if not valid_prices:
            return None

        best = min(valid_prices, key=lambda x: x.get("price_usd", float("inf")))

        return {
            "price_usd": best.get("price_usd"),
            "source": best.get("source"),
            "mode": best.get("mode"),
            "operator": best.get("operator"),
            "departure_time": best.get("departure_time"),
            "duration": best.get("duration"),
            "travel_date": best.get("travel_date"),
        }

    def _get_cheaper_dates(self, scraped_prices: list[dict]) -> list[dict]:
        """Extract cheaper alternative dates from scraped data."""
        all_alternatives = []

        for p in scraped_prices:
            base_price = p.get("price_usd", float("inf"))
            alt_dates = p.get("alternative_dates", [])

            for alt in alt_dates:
                alt_price = alt.get("price_usd")
                if alt_price and alt_price < base_price:
                    all_alternatives.append({
                        "date": alt.get("date"),
                        "price_usd": alt_price,
                        "savings_usd": base_price - alt_price,
                    })

        # Sort by price and return top 3
        all_alternatives.sort(key=lambda x: x.get("price_usd", float("inf")))
        return all_alternatives[:3]
