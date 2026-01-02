"""Planner Agent - Understands user intent and allocates days to cities."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.config.constants import PLANNER_TEMPERATURE
from src.models.agent_outputs import PlannerOutput
from src.models.state import AgentState


PLANNER_SYSTEM_PROMPT = """You are an expert travel planner specializing in creating realistic, well-balanced trip itineraries.

Your job is to:
1. Understand the user's travel request (destination, duration, style, budget)
2. Identify which cities/locations to visit
3. Allocate the right number of days to each city
4. Consider travel time between cities (don't pack too many cities for short trips)

IMPORTANT GUIDELINES:

**Day Allocation Rules:**
- Minimum 1 day per city, but 2+ days is better for meaningful exploration
- Major cities (capitals, tourist hotspots) need 2-4 days
- Small towns/villages can be day trips or 1-2 days
- Account for travel days - a city that's 5+ hours away eats into activity time

**Realistic Planning:**
- For a 5-day trip, 2-3 cities maximum is realistic
- For a 7-day trip, 3-4 cities is comfortable
- For 10+ days, can go up to 5 cities if travel is efficient
- Never suggest zig-zag routes (A → C → B when A → B → C is more logical)

**Budget Level Inference:**
- "backpacking", "cheap", "budget" → budget
- "moderate", "comfortable", "mid-range" → mid_range
- "luxury", "premium", "upscale" → luxury
- If not specified, default to mid_range

**Traveler Profile:**
- Look for mentions of "solo", "with partner/spouse", "family with kids", "group of friends"
- Default to solo if not specified

**Travel Style:**
- "adventure", "active", "hiking" → adventure
- "relaxed", "slow", "leisurely" → relaxed
- "culture", "history", "museums" → cultural
- Mix of styles → mixed

When re-planning based on critic feedback, carefully address the specific issues mentioned.
The critic_feedback field will contain specific instructions if this is a re-planning iteration.
"""


class PlannerAgent(BaseAgent):
    """Planner Agent for trip understanding and city allocation.

    This agent:
    - Parses user's travel request
    - Determines budget level, traveler profile, travel style
    - Allocates appropriate number of days to each city
    - Provides reasoning for allocations

    Uses GPT-4o for complex reasoning about trip planning.
    """

    agent_name = "planner"

    def __init__(self, **kwargs):
        # Override temperature for planner
        kwargs.setdefault("temperature", PLANNER_TEMPERATURE)
        super().__init__(**kwargs)

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Execute the planning task.

        Args:
            state: Current graph state containing user_request and
                   optionally critic_feedback for re-planning.

        Returns:
            State updates with trip_summary and city_allocations.
        """
        user_request = state["user_request"]
        critic_feedback = state.get("critic_feedback")
        iteration = state.get("iteration_count", 0)

        # Build the human message
        human_content = f"Plan this trip:\n\n{user_request}"

        if critic_feedback:
            human_content += f"""

---
IMPORTANT: This is re-planning iteration {iteration + 1}.
The previous plan had issues. Please address the following feedback:

{critic_feedback}

Make specific changes to address these issues while maintaining a coherent trip plan.
"""

        # Get structured output from LLM
        structured_llm = self.get_structured_llm(PlannerOutput)

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        result: PlannerOutput = await structured_llm.ainvoke(messages)

        # Convert to state update format
        trip_summary = {
            "understanding": result.trip_understanding,
            "total_days": result.total_days,
            "budget_level": result.budget_level.value,
            "traveler_profile": result.traveler_profile,
            "travel_style": result.travel_style,
            "strategy": result.overall_strategy,
        }

        city_allocations = [
            {
                "city": ca.city,
                "country": ca.country,
                "days": ca.days,
                "visit_order": ca.visit_order,
                "highlights": ca.highlights,
                "reasoning": ca.reasoning,
            }
            for ca in result.city_allocations
        ]

        return {
            "trip_summary": trip_summary,
            "city_allocations": city_allocations,
            # Clear critic feedback after addressing it
            "critic_feedback": None,
        }
