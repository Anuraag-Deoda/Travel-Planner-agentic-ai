"""Research/Browser Agent - Browses the web for attractions and current information."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.config.constants import RESEARCH_TEMPERATURE, MAX_ATTRACTIONS_PER_CITY
from src.models.agent_outputs import ResearchOutput
from src.models.state import AgentState
from src.tools.browser.playwright_tools import search_attractions, get_attraction_details
from src.tools.google_api import search_attractions_places_api, get_attraction_details_places_api


RESEARCH_SYSTEM_PROMPT = """You are a travel research specialist. Your job is to find accurate, current information about tourist attractions and things to do in cities.

For each city, you need to:
1. Search for top attractions and things to do
2. Get details about opening hours, ticket prices when available
3. Categorize attractions (landmark, museum, nature, market, temple, etc.)
4. Estimate reasonable visit durations
5. Note any seasonal considerations or tips

IMPORTANT:
- Only include attractions that actually exist - do not make them up
- If search results are unclear, acknowledge uncertainty
- Prioritize well-known, highly-rated attractions
- Include a mix of categories (culture, nature, food, etc.)
- Note if attractions require advance booking

When processing search results:
- Extract the most relevant attractions from the search snippets
- Remove duplicates and irrelevant results
- Assign appropriate categories based on the description
- Estimate duration based on the type of attraction

Output attractions in the structured format requested.
"""


class ResearchAgent(BaseAgent):
    """Research/Browser Agent for finding attractions and current information.

    This agent:
    - Uses Playwright to browse for attraction information
    - Searches multiple sources for comprehensive data
    - Structures findings into attraction records
    - Caches results to avoid redundant browsing

    Uses GPT-4o for interpreting and structuring browsed content.
    """

    agent_name = "research"

    def __init__(self, **kwargs):
        kwargs.setdefault("temperature", RESEARCH_TEMPERATURE)
        super().__init__(**kwargs)

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Execute research for attractions in planned cities.

        Args:
            state: Current graph state containing city_allocations.

        Returns:
            State updates with attractions and research_sources.
        """
        city_allocations = state.get("city_allocations", [])

        if not city_allocations:
            return {
                "attractions": [],
                "research_sources": [],
            }

        all_attractions = []
        all_sources = []

        # Research each city
        for allocation in city_allocations:
            city = allocation.get("city", "")
            country = allocation.get("country", "")
            days = allocation.get("days", 1)

            if not city:
                continue

            # Calculate how many attractions to find based on days
            # Roughly 3-4 attractions per day is realistic
            target_attractions = min(days * 4, MAX_ATTRACTIONS_PER_CITY)

            # First try Google Places API (more detailed data with photos)
            places_api_data = []
            try:
                places_result = await search_attractions_places_api.ainvoke({
                    "city": city,
                    "max_results": target_attractions,
                })
                places_data = json.loads(places_result)
                if not places_data.get("error"):
                    places_api_data = places_data.get("attractions", [])
                    all_sources.append(f"Google Places API: {city}")
            except Exception:
                pass

            # Fallback to browser tool if Places API didn't return enough
            browser_data = []
            if len(places_api_data) < target_attractions // 2:
                try:
                    search_result = await search_attractions.ainvoke({
                        "city": city,
                        "country": country,
                        "max_results": target_attractions,
                    })
                    search_data = json.loads(search_result)
                    if "error" not in search_data:
                        browser_data = search_data.get("attractions", [])
                except Exception:
                    pass

            # Combine data, preferring Places API data
            combined_data = places_api_data + browser_data

            if combined_data:
                # Use LLM to clean up and structure the attractions
                structured_attractions = await self._structure_attractions(
                    city=city,
                    country=country,
                    raw_data=combined_data,
                    days=days,
                    places_api_data=places_api_data,
                )

                all_attractions.extend(structured_attractions.attractions_found)
                all_sources.extend(structured_attractions.sources_browsed)

        # Build final attractions list with enhanced data
        final_attractions = []
        for a in all_attractions:
            attraction_dict = {
                "name": a.name,
                "city": a.city,
                "description": a.description,
                "category": a.category,
                "estimated_duration_hours": a.estimated_duration_hours,
                "address": a.address,
                "opening_hours": a.opening_hours,
                "entrance_fee_usd": a.entrance_fee_usd,
                "booking_required": a.booking_required,
                "tips": a.tips,
                "source_url": a.source_url,
            }

            # Add enhanced data if available from Places API
            if hasattr(a, "rating") and a.rating:
                attraction_dict["rating"] = a.rating
            if hasattr(a, "review_count") and a.review_count:
                attraction_dict["review_count"] = a.review_count
            if hasattr(a, "photo_urls") and a.photo_urls:
                attraction_dict["photo_urls"] = a.photo_urls
            if hasattr(a, "google_maps_url") and a.google_maps_url:
                attraction_dict["google_maps_url"] = a.google_maps_url
            if hasattr(a, "website") and a.website:
                attraction_dict["website"] = a.website
            if hasattr(a, "phone") and a.phone:
                attraction_dict["phone"] = a.phone
            if hasattr(a, "review_highlights") and a.review_highlights:
                attraction_dict["review_highlights"] = a.review_highlights

            final_attractions.append(attraction_dict)

        return {
            "attractions": final_attractions,
            "research_sources": all_sources,
        }

    async def _structure_attractions(
        self,
        city: str,
        country: str,
        raw_data: list[dict],
        days: int,
        places_api_data: list[dict] | None = None,
    ) -> ResearchOutput:
        """Use LLM to structure raw search results into proper attractions.

        Args:
            city: City name.
            country: Country name.
            raw_data: Raw search result data.
            days: Number of days in this city (for context).
            places_api_data: Data from Google Places API with photos and ratings.

        Returns:
            Structured ResearchOutput with attractions.
        """
        # Build enhanced data section if we have Places API data
        places_section = ""
        if places_api_data:
            places_section = "\n\nDETAILED DATA FROM GOOGLE PLACES API (use this as primary source):\n"
            for p in places_api_data[:15]:
                places_section += f"\n- {p.get('name')}"
                if p.get("rating"):
                    places_section += f" ★{p.get('rating')}"
                if p.get("review_count"):
                    places_section += f" ({p.get('review_count'):,} reviews)"
                places_section += f"\n  Category: {p.get('category', 'unknown')}"
                if p.get("address"):
                    places_section += f"\n  Address: {p.get('address')}"
                if p.get("opening_hours"):
                    hours = p.get("opening_hours")
                    if isinstance(hours, list):
                        places_section += f"\n  Hours: {hours[0] if hours else 'N/A'}..."
                if p.get("website"):
                    places_section += f"\n  Website: {p.get('website')}"
                if p.get("photo_urls"):
                    places_section += f"\n  Photos available: {len(p.get('photo_urls', []))} images"
                if p.get("review_highlights"):
                    highlights = p.get("review_highlights", [])
                    if highlights:
                        preview = highlights[0].get("text", "")[:100] if isinstance(highlights[0], dict) else str(highlights[0])[:100]
                        places_section += f'\n  Review: "{preview}..."'

        human_content = f"""Process these search results for {city}, {country} into structured attraction data.

The traveler will spend {days} days in {city}.
{places_section}
Raw search results:
{json.dumps(raw_data[:20], indent=2)}

Please:
1. Extract valid attractions from these results
2. PRIORITIZE attractions from the Google Places API data (they have ratings, reviews, and photos)
3. Remove duplicates and irrelevant entries
4. Assign appropriate categories (landmark, museum, temple, nature, market, etc.)
5. Estimate realistic visit durations
6. Note any information about booking requirements
7. Include the rating and review_count in your output when available
8. Preserve photo_urls, google_maps_url, website, and phone from the Places API data

Return structured attraction data for these results.
"""

        structured_llm = self.get_structured_llm(ResearchOutput)

        messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        result = await structured_llm.ainvoke(messages)

        # Enrich the structured results with Places API data
        if places_api_data:
            result = self._enrich_with_places_data(result, places_api_data)

        return result

    def _enrich_with_places_data(
        self,
        result: ResearchOutput,
        places_data: list[dict],
    ) -> ResearchOutput:
        """Enrich LLM-structured attractions with Places API data.

        Args:
            result: LLM-structured ResearchOutput.
            places_data: Raw data from Google Places API.

        Returns:
            Enriched ResearchOutput with photos and ratings.
        """
        # Create lookup by name
        places_lookup = {}
        for p in places_data:
            name_lower = p.get("name", "").lower().strip()
            places_lookup[name_lower] = p

        # Enrich each attraction
        for attraction in result.attractions_found:
            name_lower = attraction.name.lower().strip()

            # Try exact match first
            places_match = places_lookup.get(name_lower)

            # Try partial match
            if not places_match:
                for places_name, places_info in places_lookup.items():
                    if name_lower in places_name or places_name in name_lower:
                        places_match = places_info
                        break

            if places_match:
                # Set rating and review count
                if places_match.get("rating"):
                    attraction.rating = places_match["rating"]
                if places_match.get("review_count"):
                    attraction.review_count = places_match["review_count"]
                if places_match.get("photo_urls"):
                    attraction.photo_urls = places_match["photo_urls"]
                if places_match.get("google_maps_url"):
                    attraction.google_maps_url = places_match["google_maps_url"]
                if places_match.get("website"):
                    attraction.website = places_match["website"]
                if places_match.get("phone"):
                    attraction.phone = places_match["phone"]
                if places_match.get("review_highlights"):
                    highlights = places_match["review_highlights"]
                    attraction.review_highlights = [
                        h.get("text", "") if isinstance(h, dict) else str(h)
                        for h in highlights[:5]
                    ]

        return result
