"""Unit tests for data models."""

import pytest
from pydantic import ValidationError

from src.models.itinerary import (
    Attraction,
    BudgetLevel,
    DayPlan,
    Meal,
    TransportMode,
    TravelItinerary,
    TravelerProfile,
)
from src.models.agent_outputs import (
    CityAllocation,
    CriticOutput,
    IssueSeverity,
    PlannerOutput,
    ValidationIssue,
)
from src.models.state import AgentState, get_initial_state


class TestBudgetLevel:
    """Tests for BudgetLevel enum."""

    def test_budget_values(self):
        assert BudgetLevel.BUDGET.value == "budget"
        assert BudgetLevel.MID_RANGE.value == "mid_range"
        assert BudgetLevel.LUXURY.value == "luxury"


class TestTransportMode:
    """Tests for TransportMode enum."""

    def test_transport_values(self):
        assert TransportMode.TRAIN.value == "train"
        assert TransportMode.FLIGHT.value == "flight"
        assert TransportMode.BUS.value == "bus"


class TestAttraction:
    """Tests for Attraction model."""

    def test_valid_attraction(self):
        attraction = Attraction(
            name="City Palace",
            city="Udaipur",
            category="landmark",
            estimated_duration_hours=2.5,
        )
        assert attraction.name == "City Palace"
        assert attraction.estimated_duration_hours == 2.5

    def test_attraction_with_all_fields(self):
        attraction = Attraction(
            name="City Palace",
            city="Udaipur",
            description="Historic royal residence",
            category="landmark",
            estimated_duration_hours=2.5,
            address="City Palace Complex, Udaipur",
            opening_hours="9:00 AM - 5:30 PM",
            entrance_fee_usd=15.0,
            booking_required=True,
            tips="Visit early morning to avoid crowds",
            source_url="https://example.com",
        )
        assert attraction.booking_required is True
        assert attraction.entrance_fee_usd == 15.0


class TestMeal:
    """Tests for Meal model."""

    def test_valid_meal(self):
        meal = Meal(
            meal_type="lunch",
            cuisine_type="local",
            budget_level=BudgetLevel.MID_RANGE,
            estimated_cost_usd=20.0,
        )
        assert meal.meal_type == "lunch"
        assert meal.budget_level == BudgetLevel.MID_RANGE


class TestCityAllocation:
    """Tests for CityAllocation model."""

    def test_valid_allocation(self):
        allocation = CityAllocation(
            city="Udaipur",
            country="India",
            days=2,
            visit_order=1,
            highlights=["City Palace", "Lake Pichola"],
            reasoning="Cultural hub requiring 2 days",
        )
        assert allocation.days == 2
        assert len(allocation.highlights) == 2

    def test_invalid_days(self):
        with pytest.raises(ValidationError):
            CityAllocation(
                city="Udaipur",
                country="India",
                days=0,  # Invalid: must be >= 1
                visit_order=1,
                highlights=[],
                reasoning="Test",
            )


class TestPlannerOutput:
    """Tests for PlannerOutput model."""

    def test_valid_planner_output(self, sample_planner_output):
        assert sample_planner_output.total_days == 5
        assert len(sample_planner_output.city_allocations) == 3

    def test_city_allocation_total_matches(self, sample_planner_output):
        total_allocated = sum(c.days for c in sample_planner_output.city_allocations)
        assert total_allocated == sample_planner_output.total_days


class TestCriticOutput:
    """Tests for CriticOutput model."""

    def test_valid_critic_output(self):
        output = CriticOutput(
            is_valid=True,
            overall_score=85.0,
            issues=[],
            requires_replanning=False,
            strengths=["Well-balanced itinerary", "Realistic timing"],
            final_recommendations=["Book trains in advance"],
        )
        assert output.is_valid is True
        assert output.overall_score == 85.0

    def test_critic_with_issues(self):
        output = CriticOutput(
            is_valid=False,
            overall_score=40.0,
            issues=[
                ValidationIssue(
                    category="timing",
                    description="Day 2 is overpacked",
                    severity=IssueSeverity.HIGH,
                    affected_days=[2],
                    suggested_fix="Remove one attraction",
                ),
            ],
            requires_replanning=True,
            replan_focus="timing",
            replan_instructions="Reduce Day 2 activities",
            strengths=[],
        )
        assert output.requires_replanning is True
        assert len(output.issues) == 1
        assert output.issues[0].severity == IssueSeverity.HIGH


class TestAgentState:
    """Tests for AgentState."""

    def test_get_initial_state(self):
        state = get_initial_state("Plan a trip to Japan")
        assert state["user_request"] == "Plan a trip to Japan"
        assert state["iteration_count"] == 0
        assert state["attractions"] == []
        assert state["final_itinerary"] is None


class TestTravelItinerary:
    """Tests for TravelItinerary model."""

    def test_minimal_itinerary(self):
        itinerary = TravelItinerary(
            trip_title="5-Day Japan Trip",
            destination_summary="Tokyo and Kyoto exploration",
            total_days=5,
            budget_level=BudgetLevel.MID_RANGE,
            total_estimated_cost_usd=2500.0,
            cities_visited=["Tokyo", "Kyoto"],
            daily_plans=[],
        )
        assert itinerary.trip_title == "5-Day Japan Trip"
        assert len(itinerary.cities_visited) == 2
