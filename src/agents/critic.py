"""Critic/Validator Agent - Validates the entire plan and triggers re-planning if needed."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.config.constants import CRITIC_TEMPERATURE
from src.config.settings import get_settings
from src.models.agent_outputs import CriticOutput, IssueSeverity, ValidationIssue
from src.models.state import AgentState


CRITIC_SYSTEM_PROMPT = """You are a meticulous travel plan validator. Your job is to critically review travel itineraries and identify issues that would make the trip problematic.

VALIDATION CATEGORIES:

**1. Timing Issues:**
- Too many activities packed into a single day (more than 4-5 major attractions)
- Not enough time for meals and rest
- Activities scheduled during closed hours
- Unrealistic travel times between attractions

**2. Logistics Issues:**
- Impossible connections (city too far for day trip)
- Zig-zag routing not caught by geography agent
- No time allocated for check-in/check-out at hotels
- Transport not available at suggested times

**3. Budget Issues:**
- Total costs significantly exceed budget level expectations
- Missing major cost categories
- Unrealistic price estimates

**4. Feasibility Issues:**
- Attractions that require advance booking not flagged
- Seasonal closures not considered
- Weather-inappropriate activities
- Physical demands too high for traveler profile

**5. Balance Issues:**
- All activities of one type (only museums, only nature)
- No free time or flexibility
- Important local experiences missing

SEVERITY LEVELS:
- LOW: Minor suggestions, plan is still good
- MEDIUM: Should be addressed but trip is viable
- HIGH: Significant problem that affects trip quality
- CRITICAL: Plan is broken, must be fixed

DECISION CRITERIA FOR RE-PLANNING:
- Any CRITICAL issue → requires re-planning
- 3+ HIGH issues → requires re-planning
- Many MEDIUM issues that compound → consider re-planning
- Only LOW issues → approve the plan

When requiring re-planning, provide SPECIFIC instructions:
- What exactly needs to change
- Which cities/days are affected
- Concrete suggestions for fixes

Be fair but thorough. A good trip plan should pass validation.
"""


class CriticAgent(BaseAgent):
    """Critic/Validator Agent for plan validation.

    This agent:
    - Reviews the complete travel plan
    - Identifies issues across multiple categories
    - Assigns severity levels to issues
    - Decides if re-planning is needed
    - Provides specific feedback for the Planner if re-planning

    Uses GPT-4o for critical validation decisions.
    """

    agent_name = "critic"

    def __init__(self, **kwargs):
        kwargs.setdefault("temperature", CRITIC_TEMPERATURE)
        super().__init__(**kwargs)
        self.settings = get_settings()

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Execute plan validation.

        Args:
            state: Current graph state with all agent outputs.

        Returns:
            State updates with validation_result, critic_feedback,
            and potentially iteration_count increment.
        """
        iteration = state.get("iteration_count", 0)
        max_iterations = self.settings.max_replan_iterations

        # Gather all relevant state for validation
        trip_summary = state.get("trip_summary", {})
        city_allocations = state.get("city_allocations", [])
        route_validation = state.get("route_validation", {})
        route_segments = state.get("route_segments", [])
        attractions = state.get("attractions", [])
        food_recommendations = state.get("food_recommendations", [])
        transport_options = state.get("transport_options", [])
        budget_breakdown = state.get("budget_breakdown", {})

        # Build comprehensive summary for validation
        human_content = self._build_validation_prompt(
            trip_summary=trip_summary,
            city_allocations=city_allocations,
            route_validation=route_validation,
            route_segments=route_segments,
            attractions=attractions,
            food_recommendations=food_recommendations,
            transport_options=transport_options,
            budget_breakdown=budget_breakdown,
            iteration=iteration,
            max_iterations=max_iterations,
        )

        structured_llm = self.get_structured_llm(CriticOutput)

        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        result: CriticOutput = await structured_llm.ainvoke(messages)

        # Check if we've hit max iterations
        if iteration >= max_iterations and result.requires_replanning:
            # Force approval after max iterations with warning
            result.is_valid = True
            result.requires_replanning = False
            result.issues.append(
                ValidationIssue(
                    category="process",
                    description=f"Max re-planning iterations ({max_iterations}) reached. Approving with known issues.",
                    severity=IssueSeverity.MEDIUM,
                )
            )

        # Build state update
        validation_result = {
            "is_valid": result.is_valid,
            "overall_score": result.overall_score,
            "issues": [
                {
                    "category": issue.category,
                    "description": issue.description,
                    "severity": issue.severity.value,
                    "affected_days": issue.affected_days,
                    "affected_cities": issue.affected_cities,
                    "suggested_fix": issue.suggested_fix,
                }
                for issue in result.issues
            ],
            "requires_replanning": result.requires_replanning,
            "strengths": result.strengths,
            "final_recommendations": result.final_recommendations,
        }

        state_update = {
            "validation_result": validation_result,
        }

        # If re-planning needed, set feedback and increment iteration
        if result.requires_replanning:
            feedback_parts = []
            if result.replan_focus:
                feedback_parts.append(f"Focus area: {result.replan_focus}")
            if result.replan_instructions:
                feedback_parts.append(f"Instructions: {result.replan_instructions}")

            # Add critical/high issues as specific feedback
            critical_issues = [
                i for i in result.issues
                if i.severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH)
            ]
            if critical_issues:
                feedback_parts.append("\nCritical issues to address:")
                for issue in critical_issues:
                    feedback_parts.append(
                        f"- [{issue.severity.value.upper()}] {issue.description}"
                    )
                    if issue.suggested_fix:
                        feedback_parts.append(f"  Suggestion: {issue.suggested_fix}")

            state_update["critic_feedback"] = "\n".join(feedback_parts)
            state_update["iteration_count"] = iteration + 1
        else:
            state_update["critic_feedback"] = None

        return state_update

    def _build_validation_prompt(
        self,
        trip_summary: dict,
        city_allocations: list,
        route_validation: dict,
        route_segments: list,
        attractions: list,
        food_recommendations: list,
        transport_options: list,
        budget_breakdown: dict,
        iteration: int,
        max_iterations: int,
    ) -> str:
        """Build the validation prompt with all plan details."""

        # Cities summary
        cities_info = ""
        if city_allocations:
            sorted_cities = sorted(city_allocations, key=lambda x: x.get("visit_order", 0))
            cities_info = "\n".join(
                f"  {c['visit_order']}. {c['city']}, {c['country']} - {c['days']} days"
                for c in sorted_cities
            )

        # Route info
        route_info = ""
        if route_segments:
            route_info = "\n".join(
                f"  {s['from_city']} → {s['to_city']}: {s['distance_km']}km, "
                f"{s['travel_time_hours']}h by {s['recommended_transport']}"
                for s in route_segments
            )

        # Attractions by city
        attractions_by_city = {}
        for attr in attractions:
            city = attr.get("city", "Unknown")
            if city not in attractions_by_city:
                attractions_by_city[city] = []
            attractions_by_city[city].append(attr)

        attractions_info = ""
        for city, attrs in attractions_by_city.items():
            attractions_info += f"\n  {city}:\n"
            for a in attrs[:5]:  # Limit to 5 per city for prompt length
                duration = a.get("estimated_duration_hours", "?")
                attractions_info += f"    - {a.get('name', 'Unknown')} ({duration}h)\n"

        # Budget info
        budget_info = ""
        if budget_breakdown:
            budget_info = f"""
  Total: ${budget_breakdown.get('total', 'Unknown')}
  Breakdown: {budget_breakdown}"""

        prompt = f"""Please validate this travel plan:

=== TRIP OVERVIEW ===
Understanding: {trip_summary.get('understanding', 'N/A')}
Duration: {trip_summary.get('total_days', 'N/A')} days
Budget Level: {trip_summary.get('budget_level', 'N/A')}
Travel Style: {trip_summary.get('travel_style', 'N/A')}
Traveler Profile: {trip_summary.get('traveler_profile', 'N/A')}

=== CITIES & ALLOCATION ===
{cities_info if cities_info else '  No cities planned'}

=== ROUTE ===
Valid: {route_validation.get('is_valid', 'Unknown')}
Total Travel Time: {route_validation.get('total_travel_time_hours', 'Unknown')} hours
Warnings: {route_validation.get('warnings', [])}

Route Segments:
{route_info if route_info else '  No route segments'}

=== ATTRACTIONS ===
{attractions_info if attractions_info else '  No attractions researched yet'}

=== FOOD RECOMMENDATIONS ===
{len(food_recommendations)} recommendations gathered

=== TRANSPORT OPTIONS ===
{len(transport_options)} options identified

=== BUDGET ===
{budget_info if budget_info else '  Budget not calculated yet'}

=== VALIDATION CONTEXT ===
This is iteration {iteration + 1} of {max_iterations} maximum.
{"This is a re-planning attempt - be strict about whether issues were addressed." if iteration > 0 else "This is the first validation pass."}

Please provide your validation assessment."""

        return prompt
