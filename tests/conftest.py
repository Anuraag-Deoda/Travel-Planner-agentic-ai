"""Pytest fixtures for testing the Travel Planner."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.models.agent_outputs import (
    CityAllocation,
    GeographyOutput,
    PlannerOutput,
    RouteSegment,
)
from src.models.itinerary import BudgetLevel, TransportMode


# Set test environment
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns predefined responses."""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=AIMessage(content="Mocked response"))
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Mocked response"))
    return llm


@pytest.fixture
def mock_structured_llm(mock_llm):
    """Create a mock LLM with structured output support."""
    mock_llm.with_structured_output = MagicMock(return_value=mock_llm)
    return mock_llm


@pytest.fixture
def sample_user_request():
    """Sample user request for testing."""
    return "Plan a 5-day trip to Rajasthan visiting Udaipur, Jodhpur, and Jaipur on a mid-range budget"


@pytest.fixture
def sample_planner_output():
    """Sample Planner agent output."""
    return PlannerOutput(
        trip_understanding="5-day Rajasthan trip covering three major cities",
        total_days=5,
        budget_level=BudgetLevel.MID_RANGE,
        traveler_profile="solo",
        travel_style="cultural",
        city_allocations=[
            CityAllocation(
                city="Udaipur",
                country="India",
                days=2,
                visit_order=1,
                highlights=["City Palace", "Lake Pichola"],
                reasoning="Venice of the East, needs 2 days for lake exploration",
            ),
            CityAllocation(
                city="Jodhpur",
                country="India",
                days=2,
                visit_order=2,
                highlights=["Mehrangarh Fort", "Blue City"],
                reasoning="Major fort city with rich history",
            ),
            CityAllocation(
                city="Jaipur",
                country="India",
                days=1,
                visit_order=3,
                highlights=["Amber Fort", "Hawa Mahal"],
                reasoning="Final stop, highlights only due to time",
            ),
        ],
        overall_strategy="West to East route through Rajasthan's major heritage cities",
    )


@pytest.fixture
def sample_geography_output():
    """Sample Geography agent output."""
    return GeographyOutput(
        route_is_valid=True,
        original_order=["Udaipur", "Jodhpur", "Jaipur"],
        optimized_order=["Udaipur", "Jodhpur", "Jaipur"],
        route_changed=False,
        route_segments=[
            RouteSegment(
                from_city="Udaipur",
                to_city="Jodhpur",
                distance_km=250,
                recommended_transport=TransportMode.TRAIN,
                travel_time_hours=4.5,
                is_feasible=True,
                issues=[],
            ),
            RouteSegment(
                from_city="Jodhpur",
                to_city="Jaipur",
                distance_km=340,
                recommended_transport=TransportMode.TRAIN,
                travel_time_hours=5,
                is_feasible=True,
                issues=[],
            ),
        ],
        total_travel_time_hours=9.5,
        total_distance_km=590,
        suggestions=["Consider overnight train for Jodhpur-Jaipur segment"],
        warnings=[],
    )


@pytest.fixture
def sample_initial_state(sample_user_request):
    """Sample initial state for graph testing."""
    return {
        "messages": [],
        "user_request": sample_user_request,
        "trip_summary": None,
        "city_allocations": None,
        "route_validation": None,
        "route_segments": None,
        "attractions": [],
        "research_sources": [],
        "food_recommendations": None,
        "cultural_tips": None,
        "transport_options": None,
        "budget_breakdown": None,
        "validation_result": None,
        "critic_feedback": None,
        "iteration_count": 0,
        "current_city_index": 0,
        "final_itinerary": None,
    }


@pytest.fixture
def mock_browser_page():
    """Create a mock Playwright page."""
    page = AsyncMock()
    page.goto = AsyncMock(return_value=MagicMock(ok=True))
    page.evaluate = AsyncMock(return_value=[])
    page.query_selector = AsyncMock(return_value=None)
    page.close = AsyncMock()
    return page


@pytest.fixture
def mock_browser_context(mock_browser_page):
    """Create a mock Playwright browser context."""
    context = AsyncMock()
    context.new_page = AsyncMock(return_value=mock_browser_page)
    context.close = AsyncMock()
    return context


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary directory for cache testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)
