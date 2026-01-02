"""Clarification Agent - Gathers user preferences before planning."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.models.agent_outputs import ClarificationOutput
from src.models.state import AgentState


CLARIFICATION_SYSTEM_PROMPT = """You are a travel planning assistant. Before creating an itinerary, you need to gather essential information from the user.

Analyze the user's travel request and determine what additional information is needed for a personalized plan.

## WHAT TO CHECK FOR:

1. **Origin City** (question_id: "origin_city")
   - If NOT mentioned: Ask where they're traveling from
   - This helps plan flights/trains to the destination

2. **Specific Destinations** (question_id: "specific_destinations")
   - If user mentions only a state/country (e.g., "Rajasthan", "Japan"): Ask which specific cities
   - If specific cities ARE mentioned (e.g., "Tokyo and Kyoto"): DON'T ask

3. **Places Already Visited** (question_id: "visited_places")
   - Ask if they've visited the destination before
   - Helps avoid recommending places they've already seen

4. **Dietary Preferences** (question_id: "dietary")
   - Ask about food restrictions/preferences
   - Options: Vegetarian, Vegan, Non-vegetarian, Halal, Kosher, No restrictions

5. **Travel Pace** (question_id: "travel_pace")
   - Ask about preferred pace
   - Options: Relaxed (fewer activities, more free time), Moderate (balanced), Fast-paced (packed itinerary)

## RULES:
- Only ask questions for information NOT already in the request
- Maximum 5 questions total
- Mark required=True only for truly essential questions (origin_city, specific_destinations if needed)
- For dietary and pace, include common options
- If the request is very detailed and complete, set needs_clarification=False and ready_to_plan=True

## EXAMPLES:

Request: "Plan a 5-day trip to Rajasthan"
- Missing: origin city, specific cities in Rajasthan, dietary, pace
- Questions: origin_city, specific_destinations, dietary, travel_pace

Request: "Plan a 7-day trip from Delhi to Mumbai and Goa, vegetarian food only"
- Has: origin (Delhi), destinations (Mumbai, Goa), dietary (vegetarian)
- Missing: pace, places visited
- Questions: travel_pace, visited_places (optional)

Request: "5 days in Tokyo and Kyoto from New York, relaxed pace, no dietary restrictions"
- Has everything essential
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
