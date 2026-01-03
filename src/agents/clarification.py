"""Clarification Agent - Gathers user preferences before planning."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.models.agent_outputs import ClarificationOutput
from src.models.state import AgentState


CLARIFICATION_SYSTEM_PROMPT = """You are a travel planning assistant. Before creating an itinerary, you need to gather essential information from the user.

Analyze the user's travel request and determine what additional information is needed for a personalized plan.

## WHAT TO CHECK FOR:

1. **Travel Dates** (question_id: "travel_dates") - ASK THIS FIRST!
   - ALWAYS ask for travel dates unless explicitly provided
   - This is REQUIRED for getting accurate transport prices
   - Accept specific dates: "January 15-22, 2026"
   - Accept flexible dates: "mid-January", "around February", "sometime in spring"
   - question_type: "travel_dates"

2. **Origin City** (question_id: "origin_city")
   - If NOT mentioned: Ask where they're traveling from
   - This helps plan flights/trains to the destination

3. **Specific Destinations** (question_id: "specific_destinations")
   - If user mentions only a state/country (e.g., "Rajasthan", "Japan"): Ask which specific cities
   - If specific cities ARE mentioned (e.g., "Tokyo and Kyoto"): DON'T ask

4. **Places Already Visited** (question_id: "visited_places")
   - Ask if they've visited the destination before
   - Helps avoid recommending places they've already seen

5. **Dietary Preferences** (question_id: "dietary")
   - Ask about food restrictions/preferences
   - Options: Vegetarian, Vegan, Non-vegetarian, Halal, Kosher, No restrictions

6. **Travel Pace** (question_id: "travel_pace")
   - Ask about preferred pace
   - Options: Relaxed (fewer activities, more free time), Moderate (balanced), Fast-paced (packed itinerary)

## RULES:
- ALWAYS ask travel_dates FIRST unless specific dates are in the request
- Only ask questions for information NOT already in the request
- Maximum 6 questions total
- Mark required=True for travel_dates, origin_city, and specific_destinations (if needed)
- For dietary and pace, include common options
- If the request has dates AND is very detailed and complete, set needs_clarification=False and ready_to_plan=True

## EXAMPLES:

Request: "Plan a 5-day trip to Rajasthan"
- Missing: travel dates (CRITICAL), origin city, specific cities in Rajasthan, dietary, pace
- Questions: travel_dates (first!), origin_city, specific_destinations, dietary, travel_pace

Request: "Plan a 7-day trip from Delhi to Mumbai and Goa, vegetarian food only"
- Has: origin (Delhi), destinations (Mumbai, Goa), dietary (vegetarian)
- Missing: travel dates, pace
- Questions: travel_dates (first!), travel_pace

Request: "Plan a trip to Tokyo from January 15-22, 2026"
- Has: travel dates (Jan 15-22, 2026)
- Missing: origin city, pace, dietary
- Questions: origin_city, dietary, travel_pace

Request: "5 days in Tokyo and Kyoto from New York, January 10-15 2026, relaxed pace"
- Has: dates, origin, destinations, pace
- Missing: dietary
- Questions: dietary (optional)
- Could set needs_clarification=False if dietary is not critical

Request: "5 days in Tokyo and Kyoto from New York, Jan 10-15 2026, relaxed pace, vegetarian"
- Has everything essential including dates
- needs_clarification=False, ready_to_plan=True
"""


class ClarificationAgent(BaseAgent):
    """Clarification Agent for gathering user preferences.

    This agent:
    - Analyzes the user request for missing information
    - Generates targeted clarification questions
    - Returns questions for CLI prompts or API response

    Uses GPT-4o-mini for efficient question generation.
    """

    agent_name = "clarification"

    def __init__(self, **kwargs):
        kwargs.setdefault("temperature", 0.3)
        super().__init__(**kwargs)

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Analyze request and generate clarification questions if needed.

        Args:
            state: Current graph state with user_request.

        Returns:
            State updates with clarification_needed, clarification_questions.
        """
        user_request = state["user_request"]

        structured_llm = self.get_structured_llm(ClarificationOutput)

        messages = [
            SystemMessage(content=CLARIFICATION_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Analyze this travel request and determine what clarification is needed:\n\n{user_request}"
            ),
        ]

        result = await structured_llm.ainvoke(messages)

        return {
            "clarification_needed": result.needs_clarification,
            "clarification_questions": [q.model_dump() for q in result.questions],
        }
