"""Main LangGraph StateGraph definition for the travel planner."""

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from src.config.constants import MAX_GRAPH_ITERATIONS
from src.models.state import AgentState, get_initial_state
from src.graph.nodes import (
    clarification_node,
    process_answers_node,
    planner_node,
    geography_node,
    research_node,
    food_culture_node,
    transport_budget_node,
    critic_node,
    finalize_node,
)
from src.graph.edges import needs_clarification, should_replan


def create_travel_graph(
    checkpointer: SqliteSaver | MemorySaver | None = None,
) -> StateGraph:
    """Create the travel planner LangGraph workflow.

    The graph follows this flow:
    0. Clarification: Ask user for missing info (origin, dietary, pace, etc.)
       - If answers needed: Pause and wait for user input
       - If ready: Proceed to planning
    1. Planner: Understand request, allocate cities/days
    2. Geography: Validate and optimize route
    3. Research: Browse for attractions
    4. Food/Culture: Get food recommendations
    5. Transport/Budget: Calculate transport and budget
    6. Critic: Validate the complete plan
       - If issues found: Loop back to Planner with feedback
       - If approved: Finalize the itinerary

    Args:
        checkpointer: Optional checkpointer for state persistence.
            If None, uses in-memory checkpointing.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    # Create the graph with our state type
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("process_answers", process_answers_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("geography", geography_node)
    workflow.add_node("research", research_node)
    workflow.add_node("food_culture", food_culture_node)
    workflow.add_node("transport_budget", transport_budget_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("finalize", finalize_node)

    # Set the entry point - start with clarification
    workflow.set_entry_point("clarification")

    # Conditional edge from clarification
    workflow.add_conditional_edges(
        "clarification",
        needs_clarification,
        {
            "wait_for_answers": END,  # Pause for user input (resume later)
            "proceed_to_planner": "planner",  # Continue to planning
        },
    )

    # Process answers leads to planner (when resuming with answers)
    workflow.add_edge("process_answers", "planner")

    # Add edges for the main flow
    workflow.add_edge("planner", "geography")
    workflow.add_edge("geography", "research")
    workflow.add_edge("research", "food_culture")
    workflow.add_edge("food_culture", "transport_budget")
    workflow.add_edge("transport_budget", "critic")

    # Add conditional edge from critic
    # This is the feedback loop - critic decides if we need to replan
    workflow.add_conditional_edges(
        "critic",
        should_replan,
        {
            "replan": "planner",  # Loop back to planner with feedback
            "finalize": "finalize",  # Proceed to finalization
        },
    )

    # Finalize ends the graph
    workflow.add_edge("finalize", END)

    # Use provided checkpointer or create a memory one
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Compile the graph
    return workflow.compile(checkpointer=checkpointer)


def create_sqlite_checkpointer(db_path: str = "data/checkpoints/travel_planner.db") -> SqliteSaver:
    """Create a SQLite checkpointer for persistent state.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Configured SqliteSaver instance.
    """
    return SqliteSaver.from_conn_string(db_path)


def get_graph_with_persistence(db_path: str = "data/checkpoints/travel_planner.db") -> StateGraph:
    """Get the travel graph with SQLite persistence.

    This is the recommended way to create the graph for production use,
    as it enables resuming interrupted sessions.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        Compiled graph with persistent checkpointing.
    """
    checkpointer = create_sqlite_checkpointer(db_path)
    return create_travel_graph(checkpointer=checkpointer)


# Convenience function to run the graph
async def plan_trip(
    user_request: str,
    thread_id: str = "default",
    graph: StateGraph | None = None,
) -> dict:
    """Plan a trip using the travel agent graph.

    Args:
        user_request: Natural language trip request.
        thread_id: Session ID for state persistence.
        graph: Optional pre-created graph. If None, creates one with memory checkpointer.

    Returns:
        Final state dictionary containing the itinerary.
    """
    if graph is None:
        graph = create_travel_graph()

    initial_state = get_initial_state(user_request)

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_GRAPH_ITERATIONS + 20,  # ~40 steps to allow 3+ replans
    }

    # Run the graph
    result = await graph.ainvoke(initial_state, config)

    return result
