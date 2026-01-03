"""Food/Culture Agent - Provides food recommendations and cultural tips."""

import json
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.config.constants import FOOD_CULTURE_TEMPERATURE
from src.models.agent_outputs import FoodCultureOutput
from src.models.itinerary import BudgetLevel
from src.models.state import AgentState
from src.tools.google_api import (
    search_restaurants_places_api,
    get_restaurant_details_places_api,
)
from src.tools.browser.restaurant_review_tools import (
    scrape_zomato_restaurants,
    scrape_swiggy_restaurants,
)


# India cities for Zomato/Swiggy integration
INDIA_CITIES = {
    "delhi", "mumbai", "bangalore", "bengaluru", "chennai", "kolkata",
    "hyderabad", "pune", "jaipur", "udaipur", "jodhpur", "goa", "agra",
    "varanasi", "lucknow", "kochi", "trivandrum", "mysore", "shimla",
    "manali", "rishikesh", "haridwar", "amritsar", "chandigarh", "ahmedabad",
    "surat", "indore", "bhopal", "nagpur", "aurangabad", "nashik", "coimbatore",
    "madurai", "thiruvananthapuram", "cochin", "ooty", "munnar", "alleppey",
}


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

REAL REVIEW DATA:
When real restaurant reviews are provided:
- PRIORITIZE highly-rated restaurants (4.0+ stars) from the scraped data
- Use the review_highlights to mention what diners love about each place
- Include popular dishes mentioned in reviews as must_try_dishes
- Show the rating and review count in your recommendations
- If a restaurant has thousands of reviews, it's generally reliable
- Use the scraped source (google_maps, zomato, swiggy) in your output
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

            # Scrape restaurant reviews for this city
            scraped_reviews = await self._scrape_restaurant_reviews(city, country)

            result = await self._get_city_recommendations(
                city=city,
                country=country,
                days=days,
                budget_level=budget_level,
                traveler_profile=traveler_profile,
                dietary_preferences=dietary_preferences,
                scraped_reviews=scraped_reviews,
            )

            # Add food recommendations with review data
            for meal in result.restaurant_recommendations:
                # Try to match with scraped review data
                review_data = self._find_matching_review(
                    meal.restaurant_name, scraped_reviews
                )

                food_rec = {
                    "city": city,
                    "meal_type": meal.meal_type,
                    "restaurant_name": meal.restaurant_name,
                    "cuisine_type": meal.cuisine_type,
                    "budget_level": meal.budget_level.value,
                    "estimated_cost_usd": meal.estimated_cost_usd,
                    "address": meal.address,
                    "must_try_dishes": meal.must_try_dishes,
                    "dietary_notes": meal.dietary_notes,
                }

                # Add review data if available
                if review_data:
                    food_rec["rating"] = review_data.get("rating")
                    food_rec["review_count"] = review_data.get("review_count")
                    food_rec["review_source"] = review_data.get("source")
                    food_rec["review_highlights"] = review_data.get("review_highlights", [])
                    food_rec["popular_dishes_from_reviews"] = review_data.get("popular_dishes", [])
                    food_rec["source_url"] = review_data.get("source_url")
                    # Enhanced data from Google Places API
                    food_rec["photo_urls"] = review_data.get("photo_urls", [])
                    food_rec["google_maps_url"] = review_data.get("google_maps_url")
                    food_rec["website"] = review_data.get("website")
                    food_rec["phone"] = review_data.get("phone")
                    food_rec["opening_hours"] = review_data.get("opening_hours", [])
                else:
                    food_rec["review_source"] = "llm_generated"
                    food_rec["photo_urls"] = []

                all_food_recommendations.append(food_rec)

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
        scraped_reviews: list[dict] | None = None,
    ) -> FoodCultureOutput:
        """Get food and culture recommendations for a single city.

        Args:
            city: City name.
            country: Country name.
            days: Number of days in city.
            budget_level: Trip budget level.
            traveler_profile: Type of traveler.
            dietary_preferences: List of dietary restrictions/preferences.
            scraped_reviews: List of scraped restaurant reviews.

        Returns:
            FoodCultureOutput with recommendations.
        """
        dietary_info = ""
        if dietary_preferences:
            dietary_info = f"- Dietary preferences: {', '.join(dietary_preferences)}"

        # Build scraped reviews section
        reviews_section = self._build_reviews_section(scraped_reviews)

        human_content = f"""Provide food and cultural recommendations for {city}, {country}.

Trip details:
- Days in {city}: {days}
- Budget level: {budget_level}
- Traveler type: {traveler_profile}
{dietary_info}
{reviews_section}
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
When real review data is provided above, PRIORITIZE those highly-rated restaurants in your recommendations.
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

    async def _scrape_restaurant_reviews(
        self,
        city: str,
        country: str,
    ) -> list[dict]:
        """Get restaurant data from Google Places API and other sources.

        Args:
            city: City name.
            country: Country name.

        Returns:
            List of restaurant data from all sources.
        """
        all_reviews = []
        is_india = (
            city.lower() in INDIA_CITIES or
            country.lower() == "india"
        )

        # Use Google Places API (much more reliable and detailed than scraping)
        try:
            google_result = await search_restaurants_places_api.ainvoke({
                "city": city,
                "max_results": 20,
            })
            parsed = json.loads(google_result)
            if not parsed.get("error"):
                for r in parsed.get("restaurants", []):
                    r["source"] = "google_places_api"
                    all_reviews.append(r)
        except Exception:
            pass

        # For India, also try Zomato and Swiggy for additional options
        if is_india:
            try:
                zomato_result = await scrape_zomato_restaurants.ainvoke({
                    "city": city,
                    "max_results": 10,
                })
                parsed = json.loads(zomato_result)
                if not parsed.get("error"):
                    for r in parsed.get("restaurants", []):
                        r["source"] = "zomato"
                        all_reviews.append(r)
            except Exception:
                pass

            try:
                swiggy_result = await scrape_swiggy_restaurants.ainvoke({
                    "city": city,
                    "max_results": 10,
                })
                parsed = json.loads(swiggy_result)
                if not parsed.get("error"):
                    for r in parsed.get("restaurants", []):
                        r["source"] = "swiggy"
                        all_reviews.append(r)
            except Exception:
                pass

        # Sort by rating and review count (highest first)
        all_reviews.sort(
            key=lambda x: (x.get("rating") or 0, x.get("review_count") or 0),
            reverse=True,
        )

        return all_reviews

    def _build_reviews_section(self, scraped_reviews: list[dict] | None) -> str:
        """Build the reviews section for the LLM prompt.

        Args:
            scraped_reviews: List of scraped restaurant reviews.

        Returns:
            Formatted string with review data.
        """
        if not scraped_reviews:
            return ""

        lines = ["\nREAL RESTAURANT REVIEWS (from Google Maps, Zomato, Swiggy):"]
        lines.append("(Prioritize these highly-rated restaurants in your recommendations)\n")

        for r in scraped_reviews[:15]:  # Limit to top 15
            name = r.get("name", "Unknown")
            rating = r.get("rating")
            review_count = r.get("review_count")
            source = r.get("source", "unknown")
            cuisines = r.get("cuisine_types", [])
            price_level = r.get("price_level", "unknown")

            line = f"- {name}"
            if rating:
                line += f" ★{rating:.1f}"
            if review_count:
                line += f" ({review_count:,} reviews)"
            line += f" [{source}]"

            if cuisines:
                line += f" | {', '.join(cuisines[:3])}"
            if price_level and price_level != "unknown":
                line += f" | {price_level}"

            lines.append(line)

            # Add review highlights if available
            highlights = r.get("review_highlights", [])
            if highlights:
                lines.append(f"  → \"{highlights[0]}\"")

            # Add popular dishes if available
            dishes = r.get("popular_dishes", [])
            if dishes:
                lines.append(f"  → Popular: {', '.join(dishes[:3])}")

        return "\n".join(lines)

    def _find_matching_review(
        self,
        restaurant_name: Optional[str],
        scraped_reviews: list[dict] | None,
    ) -> Optional[dict]:
        """Find a matching review for a restaurant name.

        Args:
            restaurant_name: Name of the restaurant to match.
            scraped_reviews: List of scraped reviews.

        Returns:
            Matching review data or None.
        """
        if not restaurant_name or not scraped_reviews:
            return None

        name_lower = restaurant_name.lower().strip()

        # Try exact match first
        for r in scraped_reviews:
            scraped_name = (r.get("name") or "").lower().strip()
            if scraped_name == name_lower:
                return r

        # Try partial match
        for r in scraped_reviews:
            scraped_name = (r.get("name") or "").lower().strip()
            # Check if either name contains the other
            if name_lower in scraped_name or scraped_name in name_lower:
                return r

            # Check if main words match
            name_words = set(name_lower.split())
            scraped_words = set(scraped_name.split())
            common_words = name_words & scraped_words
            # If more than half the words match, consider it a match
            if len(common_words) >= min(len(name_words), len(scraped_words)) / 2:
                return r

        return None
