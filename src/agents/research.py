"""Research/Browser Agent - Browses the web for attractions and current information."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.config.constants import RESEARCH_TEMPERATURE, MAX_ATTRACTIONS_PER_CITY
from src.models.agent_outputs import ResearchOutput
from src.models.state import AgentState
from src.tools.browser.playwright_tools import search_attractions, get_attraction_details


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

            # Search for attractions using the browser tool
            try:
                search_result = await search_attractions.ainvoke({
                    "city": city,
                    "country": country,
                    "max_results": target_attractions,
                })

                search_data = json.loads(search_result)

                if "error" not in search_data:
                    raw_attractions = search_data.get("attractions", [])

                    # Use LLM to clean up and structure the attractions
                    structured_attractions = await self._structure_attractions(
                        city=city,
                        country=country,
                        raw_data=raw_attractions,
                        days=days,
                    )

                    all_attractions.extend(structured_attractions.attractions_found)
                    all_sources.extend(structured_attractions.sources_browsed)

            except Exception as e:
                # Log error but continue with other cities
                all_sources.append(f"Error researching {city}: {str(e)}")

        return {
            "attractions": [
                {
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
                for a in all_attractions
            ],
            "research_sources": all_sources,
        }

    async def _structure_attractions(
        self,
        city: str,
        country: str,
        raw_data: list[dict],
        days: int,
    ) -> ResearchOutput:
        """Use LLM to structure raw search results into proper attractions.

        Args:
            city: City name.
            country: Country name.
            raw_data: Raw search result data.
            days: Number of days in this city (for context).

        Returns:
            Structured ResearchOutput with attractions.
        """
        human_content = f"""Process these search results for {city}, {country} into structured attraction data.

The traveler will spend {days} days in {city}.

Raw search results:
{json.dumps(raw_data, indent=2)}

Please:
1. Extract valid attractions from these results
2. Remove duplicates and irrelevant entries
3. Assign appropriate categories (landmark, museum, temple, nature, market, etc.)
4. Estimate realistic visit durations
5. Note any information about booking requirements

Return structured attraction data for these results.
"""

        structured_llm = self.get_structured_llm(ResearchOutput)

        messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        result = await structured_llm.ainvoke(messages)
        return result
