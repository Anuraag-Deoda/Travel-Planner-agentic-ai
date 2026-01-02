"""Food/Culture Agent - Provides food recommendations and cultural tips."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.config.constants import FOOD_CULTURE_TEMPERATURE
from src.models.agent_outputs import FoodCultureOutput
from src.models.itinerary import BudgetLevel
from src.models.state import AgentState


FOOD_CULTURE_SYSTEM_PROMPT = """You are an expert in local cuisine and cultural practices. Your job is to provide authentic food recommendations and cultural guidance for travelers.

For each destination, you should provide:

**FOOD RECOMMENDATIONS:**
1. Must-try local dishes (3-5 signature dishes)
2. Restaurant recommendations for different budgets
3. Street food tips (what's safe, what's popular)
4. Food safety considerations

**CULTURAL GUIDANCE:**
1. Important cultural dos (customs to follow)
2. Cultural don'ts (things to avoid)
3. Dress code notes (especially for religious sites)
4. Local customs and etiquette
5. Basic language phrases if helpful

IMPORTANT GUIDELINES:
- Focus on authentic local food, not international chains
- For budget travelers, emphasize street food and local eateries
- For mid-range, balance local favorites with comfortable settings
- For luxury, include fine dining options if available
- Always include at least one vegetarian-friendly option
- Note any common allergens in local cuisine
- Be specific about cultural sensitivities (religious, social)

When recommending restaurants:
- Prefer places with local reputation over tourist traps
- Consider safety and hygiene
- Note if reservation is recommended
- Indicate price range clearly
"""


class FoodCultureAgent(BaseAgent):
    """Food/Culture Agent for dining and cultural recommendations.

    This agent:
    - Recommends local dishes and restaurants
    - Provides cultural dos and don'ts
    - Offers practical tips for travelers
    - Considers budget level and dietary needs

    Uses GPT-4o-mini for efficient recommendations.
    """

    agent_name = "food_culture"

    def __init__(self, **kwargs):
        kwargs.setdefault("temperature", FOOD_CULTURE_TEMPERATURE)
        super().__init__(**kwargs)

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Generate food and culture recommendations for all cities.

        Args:
            state: Current graph state containing city_allocations and trip_summary.

        Returns:
            State updates with food_recommendations and cultural_tips.
        """
        city_allocations = state.get("city_allocations", [])
        trip_summary = state.get("trip_summary", {})

        if not city_allocations:
            return {
                "food_recommendations": [],
                "cultural_tips": [],
            }

        budget_level = trip_summary.get("budget_level", "mid_range")
        traveler_profile = trip_summary.get("traveler_profile", "solo")
        dietary_preferences = state.get("dietary_preferences", [])

        all_food_recommendations = []
        all_cultural_tips = []

        # Get recommendations for each city
        for allocation in city_allocations:
            city = allocation.get("city", "")
            country = allocation.get("country", "")
            days = allocation.get("days", 1)

            if not city:
                continue

            result = await self._get_city_recommendations(
                city=city,
                country=country,
                days=days,
                budget_level=budget_level,
                traveler_profile=traveler_profile,
                dietary_preferences=dietary_preferences,
            )

            # Add food recommendations
            for meal in result.restaurant_recommendations:
                all_food_recommendations.append({
                    "city": city,
                    "meal_type": meal.meal_type,
                    "restaurant_name": meal.restaurant_name,
                    "cuisine_type": meal.cuisine_type,
                    "budget_level": meal.budget_level.value,
                    "estimated_cost_usd": meal.estimated_cost_usd,
                    "address": meal.address,
                    "must_try_dishes": meal.must_try_dishes,
                    "dietary_notes": meal.dietary_notes,
                })

            # Collect cultural tips (deduplicate later)
            all_cultural_tips.extend(result.cultural_dos)
            all_cultural_tips.extend([f"Don't: {dont}" for dont in result.cultural_donts])

            if result.dress_code_notes:
                all_cultural_tips.append(f"Dress code: {result.dress_code_notes}")

            if result.language_tips:
                all_cultural_tips.append(f"Language: {result.language_tips}")

        # Deduplicate cultural tips while preserving order
        seen = set()
        unique_tips = []
        for tip in all_cultural_tips:
            if tip not in seen:
                seen.add(tip)
                unique_tips.append(tip)

        return {
            "food_recommendations": all_food_recommendations,
            "cultural_tips": unique_tips,
        }

    async def _get_city_recommendations(
        self,
        city: str,
        country: str,
        days: int,
        budget_level: str,
        traveler_profile: str,
        dietary_preferences: list[str] | None = None,
    ) -> FoodCultureOutput:
        """Get food and culture recommendations for a single city.

        Args:
            city: City name.
            country: Country name.
            days: Number of days in city.
            budget_level: Trip budget level.
            traveler_profile: Type of traveler.
            dietary_preferences: List of dietary restrictions/preferences.

        Returns:
            FoodCultureOutput with recommendations.
        """
        dietary_info = ""
        if dietary_preferences:
            dietary_info = f"- Dietary preferences: {', '.join(dietary_preferences)}"

        human_content = f"""Provide food and cultural recommendations for {city}, {country}.

Trip details:
- Days in {city}: {days}
- Budget level: {budget_level}
- Traveler type: {traveler_profile}
{dietary_info}

Please provide:
1. 3-5 must-try local dishes
2. Restaurant recommendations with SPECIFIC meal types:
   - {days} BREAKFAST spots (meal_type: "breakfast") - local breakfast places, cafes
   - {days} LUNCH restaurants (meal_type: "lunch") - good for midday meals
   - {days} DINNER restaurants (meal_type: "dinner") - evening dining options
3. Street food tips
4. Cultural dos and don'ts
5. Dress code guidance
6. Any language tips

IMPORTANT: Each restaurant recommendation MUST have the correct meal_type set to exactly one of: "breakfast", "lunch", or "dinner".
Focus on authentic local experiences appropriate for the budget level.
"""

        structured_llm = self.get_structured_llm(FoodCultureOutput)

        messages = [
            SystemMessage(content=FOOD_CULTURE_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        result = await structured_llm.ainvoke(messages)

        # Ensure city is set
        result.city = city

        return result
