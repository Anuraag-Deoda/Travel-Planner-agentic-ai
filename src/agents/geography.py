"""Geography/Routing Agent - Validates distances and optimizes city order."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.config.constants import GEOGRAPHY_TEMPERATURE
from src.models.agent_outputs import GeographyOutput
from src.models.state import AgentState


GEOGRAPHY_SYSTEM_PROMPT = """You are a geography and travel logistics expert. Your job is to:

1. Validate that the proposed city route is feasible
2. Check for inefficient routing (zig-zag patterns)
3. Estimate distances and travel times between cities
4. Suggest route optimizations if needed

KEY VALIDATION RULES:

**Distance Sanity:**
- Cities 50-200km apart: 1-3 hours by car/bus
- Cities 200-500km apart: 3-6 hours by car, 2-4 hours by train
- Cities 500-1000km apart: Consider flight or overnight train
- Cities 1000km+ apart: Flight strongly recommended

**Route Efficiency:**
- Detect zig-zag routes where backtracking occurs
- The optimal route should minimize total travel time
- If A→B→C causes backtracking, suggest A→C→B or similar

**Feasibility Issues to Flag:**
- More than 4-5 hours of travel on a single day
- Multiple city changes in one day
- Unrealistic connections (no direct routes, requires 2+ transfers)

**Transport Mode Recommendations:**
- Under 300km: Train or bus preferred
- 300-800km: Train preferred, bus acceptable, flight optional
- Over 800km: Flight recommended

For each segment, provide:
- Distance in km (estimate if exact unknown)
- Recommended transport mode
- Estimated travel time
- Any feasibility concerns

If the route is already optimal, confirm it. Only suggest changes if there's a clear improvement.
"""


class GeographyAgent(BaseAgent):
    """Geography/Routing Agent for route validation and optimization.

    This agent:
    - Validates the route proposed by the Planner
    - Detects inefficient patterns (zig-zag routes)
    - Estimates distances and travel times
    - Suggests route optimizations

    Uses GPT-4o-mini for faster, deterministic validation.
    """

    agent_name = "geography"

    def __init__(self, **kwargs):
        kwargs.setdefault("temperature", GEOGRAPHY_TEMPERATURE)
        super().__init__(**kwargs)

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Execute route validation and optimization.

        Args:
            state: Current graph state containing city_allocations.

        Returns:
            State updates with route_validation and route_segments.
        """
        city_allocations = state.get("city_allocations", [])
        trip_summary = state.get("trip_summary", {})

        if not city_allocations:
            return {
                "route_validation": {
                    "is_valid": False,
                    "error": "No cities to validate",
                },
                "route_segments": [],
            }

        # Sort by visit order to get the proposed route
        sorted_cities = sorted(
            city_allocations, key=lambda x: x.get("visit_order", 0)
        )
        proposed_order = [c["city"] for c in sorted_cities]

        # Build context for the LLM
        cities_info = "\n".join(
            f"- {c['city']}, {c['country']} ({c['days']} days)"
            for c in sorted_cities
        )

        human_content = f"""Validate and optimize this travel route:

Trip Duration: {trip_summary.get('total_days', len(sorted_cities))} days
Budget Level: {trip_summary.get('budget_level', 'mid_range')}

Proposed city order:
{cities_info}

Route: {' → '.join(proposed_order)}

Please:
1. Check if this route is geographically efficient
2. Estimate distances and travel times between each pair of consecutive cities
3. Identify any routing issues (zig-zag patterns, unrealistic connections)
4. Suggest an optimized order if the current one is inefficient
5. Flag any feasibility concerns
"""

        structured_llm = self.get_structured_llm(GeographyOutput)

        messages = [
            SystemMessage(content=GEOGRAPHY_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        result: GeographyOutput = await structured_llm.ainvoke(messages)

        # Convert to state update format
        route_validation = {
            "is_valid": result.route_is_valid,
            "original_order": result.original_order,
            "optimized_order": result.optimized_order,
            "route_changed": result.route_changed,
            "total_travel_time_hours": result.total_travel_time_hours,
            "total_distance_km": result.total_distance_km,
            "suggestions": result.suggestions,
            "warnings": result.warnings,
        }

        route_segments = [
            {
                "from_city": seg.from_city,
                "to_city": seg.to_city,
                "distance_km": seg.distance_km,
                "recommended_transport": seg.recommended_transport.value,
                "travel_time_hours": seg.travel_time_hours,
                "is_feasible": seg.is_feasible,
                "issues": seg.issues,
            }
            for seg in result.route_segments
        ]

        # If route was changed, update city_allocations order
        updated_allocations = None
        if result.route_changed:
            # Reorder city_allocations based on optimized_order
            city_map = {c["city"]: c for c in city_allocations}
            updated_allocations = []
            for i, city_name in enumerate(result.optimized_order, 1):
                if city_name in city_map:
                    allocation = city_map[city_name].copy()
                    allocation["visit_order"] = i
                    updated_allocations.append(allocation)

        state_update = {
            "route_validation": route_validation,
            "route_segments": route_segments,
        }

        if updated_allocations:
            state_update["city_allocations"] = updated_allocations

        return state_update
