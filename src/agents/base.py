"""Base agent class with LLM initialization and model selection."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.config.settings import get_settings
from src.config.constants import AGENT_MODELS
from src.models.state import AgentState


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Handles LLM initialization with appropriate model selection based on
    agent type and provides common utilities.
    """

    # Subclasses should set this to their agent name (e.g., "planner", "critic")
    agent_name: str = "base"

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        temperature: Optional[float] = None,
    ):
        """Initialize the agent with LLM.

        Args:
            llm: Optional pre-configured LLM instance. If not provided,
                 one will be created based on agent_name configuration.
            temperature: Optional temperature override.
        """
        self.settings = get_settings()

        if llm is not None:
            self.llm = llm
        else:
            self.llm = self._create_llm(temperature)

    def _create_llm(self, temperature: Optional[float] = None) -> ChatOpenAI:
        """Create LLM instance based on agent configuration.

        Uses AGENT_MODELS mapping to determine which model (GPT-4o vs GPT-4o-mini)
        to use for each agent type.
        """
        # Determine model based on agent name
        model_key = AGENT_MODELS.get(self.agent_name, "gpt4o_mini")

        if model_key == "gpt4o":
            model_name = self.settings.gpt4o_model
        else:
            model_name = self.settings.gpt4o_mini_model

        return ChatOpenAI(
            model=model_name,
            temperature=temperature if temperature is not None else 0.3,
            api_key=self.settings.openai_api_key,
        )

    def get_structured_llm(self, output_schema: Type[BaseModel]) -> ChatOpenAI:
        """Get LLM configured for structured output.

        Args:
            output_schema: Pydantic model class for structured output.

        Returns:
            LLM with structured output configuration.
        """
        return self.llm.with_structured_output(output_schema)

    @abstractmethod
    async def run(self, state: AgentState) -> dict[str, Any]:
        """Execute the agent's task.

        Args:
            state: Current graph state.

        Returns:
            Dictionary of state updates to merge into the graph state.
        """
        pass

    def _extract_cities(self, state: AgentState) -> list[str]:
        """Extract list of cities from state.

        Helper method to get cities from city_allocations.
        """
        allocations = state.get("city_allocations", [])
        if not allocations:
            return []

        # Sort by visit_order and extract city names
        sorted_allocations = sorted(allocations, key=lambda x: x.get("visit_order", 0))
        return [a["city"] for a in sorted_allocations]

    def _get_city_days(self, state: AgentState, city: str) -> int:
        """Get number of days allocated to a city."""
        allocations = state.get("city_allocations", [])
        for allocation in allocations:
            if allocation.get("city") == city:
                return allocation.get("days", 1)
        return 1

    def _format_attractions_for_city(
        self, state: AgentState, city: str
    ) -> list[dict]:
        """Get attractions for a specific city from state."""
        all_attractions = state.get("attractions", [])
        return [a for a in all_attractions if a.get("city") == city]


def create_agent(
    agent_class: Type["BaseAgent"],
    llm: Optional[ChatOpenAI] = None,
    **kwargs,
) -> "BaseAgent":
    """Factory function to create agent instances.

    Args:
        agent_class: The agent class to instantiate.
        llm: Optional pre-configured LLM.
        **kwargs: Additional arguments to pass to the agent.

    Returns:
        Configured agent instance.
    """
    return agent_class(llm=llm, **kwargs)
