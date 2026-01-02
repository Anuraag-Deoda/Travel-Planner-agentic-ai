"""Conditional edge logic for the LangGraph workflow."""

from typing import Literal

from src.models.state import AgentState


def needs_clarification(state: AgentState) -> Literal["wait_for_answers", "proceed_to_planner"]:
    """Check if clarification is needed before planning.

    Args:
        state: Current graph state after Clarification agent has run.

    Returns:
        "wait_for_answers" to pause for user input, "proceed_to_planner" to continue.
    """
    clarification_needed = state.get("clarification_needed", False)
    clarification_answers = state.get("clarification_answers")

    # If clarification is needed and we don't have answers yet, pause
    if clarification_needed and not clarification_answers:
        return "wait_for_answers"

    return "proceed_to_planner"


def should_replan(state: AgentState) -> Literal["replan", "finalize"]:
    """Determine if the plan needs revision based on Critic's assessment.

    This is the key decision point for the feedback loop.

    Args:
        state: Current graph state after Critic has run.

    Returns:
        "replan" to route back to Planner, "finalize" to complete.
    """
    validation_result = state.get("validation_result", {})
    requires_replanning = validation_result.get("requires_replanning", False)

    if requires_replanning:
        return "replan"

    return "finalize"


def check_route_validity(state: AgentState) -> Literal["continue", "warn"]:
    """Check if route validation passed.

    Can be used to add a warning step if route has issues.

    Args:
        state: Current graph state after Geography has run.

    Returns:
        "continue" if route is valid, "warn" if there are issues.
    """
    route_validation = state.get("route_validation", {})
    is_valid = route_validation.get("is_valid", True)
    warnings = route_validation.get("warnings", [])

    if not is_valid or len(warnings) > 2:
        return "warn"

    return "continue"


def has_cities_to_research(state: AgentState) -> Literal["research", "skip"]:
    """Check if there are cities that need research.

    Used to conditionally run the research agent.

    Args:
        state: Current graph state.

    Returns:
        "research" if cities need research, "skip" otherwise.
    """
    city_allocations = state.get("city_allocations", [])
    attractions = state.get("attractions", [])

    if not city_allocations:
        return "skip"

    # Check if we already have attractions for all cities
    researched_cities = set(a.get("city") for a in attractions)
    planned_cities = set(c.get("city") for c in city_allocations)

    if planned_cities.issubset(researched_cities):
        return "skip"

    return "research"
