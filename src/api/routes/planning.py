"""Planning routes with SSE streaming support."""

import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.config.constants import MAX_GRAPH_ITERATIONS
from src.graph.workflow import create_travel_graph
from src.models.state import get_initial_state

router = APIRouter(prefix="/plan", tags=["planning"])


class StreamPlanRequest(BaseModel):
    """Request model for streaming planning."""

    description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Natural language trip description",
    )


class ClarificationResponse(BaseModel):
    """Response containing clarification questions."""

    thread_id: str
    needs_clarification: bool
    questions: list[dict] = Field(default_factory=list)
    ready_to_plan: bool = False


class AnswersRequest(BaseModel):
    """Request containing user's answers to clarification questions."""

    thread_id: str
    answers: dict = Field(
        ...,
        description="Map of question_id to answer value",
    )


# Graph instance (would be injected via dependency in production)
_graph = None


def get_graph():
    """Get or create the graph instance."""
    global _graph
    if _graph is None:
        _graph = create_travel_graph()
    return _graph


async def stream_planning_events(
    request: StreamPlanRequest,
    thread_id: str,
) -> AsyncGenerator[str, None]:
    """Stream planning events as SSE.

    Yields events as the agents process the request:
    - agent_start: When an agent begins processing
    - agent_complete: When an agent finishes
    - error: If an error occurs
    - complete: When planning is finished
    """
    graph = get_graph()

    initial_state = get_initial_state(request.description)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_GRAPH_ITERATIONS + 20,
    }

    try:
        # Stream events from the graph
        async for event in graph.astream_events(initial_state, config, version="v2"):
            event_type = event.get("event")
            event_name = event.get("name", "unknown")

            if event_type == "on_chain_start":
                # Agent starting
                yield f"event: agent_start\ndata: {json.dumps({'agent': event_name})}\n\n"

            elif event_type == "on_chain_end":
                # Agent completed
                output = event.get("data", {}).get("output", {})

                # Extract relevant info based on agent
                summary = {}
                if event_name == "planner":
                    cities = output.get("city_allocations", [])
                    summary["cities"] = [c.get("city") for c in cities]
                elif event_name == "geography":
                    validation = output.get("route_validation", {})
                    summary["route_valid"] = validation.get("is_valid", False)
                elif event_name == "research":
                    attractions = output.get("attractions", [])
                    summary["attractions_count"] = len(attractions)
                elif event_name == "critic":
                    validation = output.get("validation_result", {})
                    summary["score"] = validation.get("overall_score", 0)
                    summary["approved"] = validation.get("is_valid", False)
                elif event_name == "finalize":
                    itinerary = output.get("final_itinerary", {})
                    summary["title"] = itinerary.get("trip_title", "")

                yield f"event: agent_complete\ndata: {json.dumps({'agent': event_name, 'summary': summary})}\n\n"

            elif event_type == "on_chain_error":
                error_msg = str(event.get("data", {}).get("error", "Unknown error"))
                yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"

        # Get final result
        final_state = await graph.aget_state(config)
        itinerary = final_state.values.get("final_itinerary") if final_state.values else None

        yield f"event: complete\ndata: {json.dumps({'success': itinerary is not None, 'thread_id': thread_id})}\n\n"

    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


@router.post("/stream")
async def stream_plan(request: StreamPlanRequest):
    """Stream planning progress via Server-Sent Events.

    Returns a stream of events as agents process the request.
    Use this for real-time progress updates in the UI.

    Event types:
    - agent_start: {"agent": "planner"}
    - agent_complete: {"agent": "planner", "summary": {...}}
    - error: {"error": "message"}
    - complete: {"success": true, "thread_id": "..."}

    Example client usage (JavaScript):
    ```javascript
    const eventSource = new EventSource('/plan/stream');
    eventSource.addEventListener('agent_start', (e) => {
        console.log('Agent started:', JSON.parse(e.data));
    });
    eventSource.addEventListener('complete', (e) => {
        console.log('Planning complete:', JSON.parse(e.data));
        eventSource.close();
    });
    ```
    """
    thread_id = str(uuid.uuid4())
    return EventSourceResponse(
        stream_planning_events(request, thread_id),
        media_type="text/event-stream",
    )


@router.post("/clarify", response_model=ClarificationResponse)
async def start_planning_with_clarification(request: StreamPlanRequest):
    """Start planning and get clarification questions if needed.

    This endpoint runs the clarification agent and returns any questions
    that need to be answered before planning can proceed.

    Flow:
    1. Call this endpoint with the trip description
    2. If `needs_clarification` is True, show questions to user
    3. Call `/plan/answers` with user's answers
    4. Use `/plan/stream` or `/plan/{thread_id}` to get the result

    Returns:
        ClarificationResponse with thread_id and any questions
    """
    graph = get_graph()
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_GRAPH_ITERATIONS + 20,
    }

    initial_state = get_initial_state(request.description)

    # Run the graph - it will stop at END after clarification if questions needed
    result = await graph.ainvoke(initial_state, config)

    return ClarificationResponse(
        thread_id=thread_id,
        needs_clarification=result.get("clarification_needed", False),
        questions=result.get("clarification_questions", []),
        ready_to_plan=not result.get("clarification_needed", True),
    )


@router.post("/answers")
async def submit_answers_and_continue(request: AnswersRequest):
    """Submit answers to clarification questions and continue planning.

    After calling `/plan/clarify` and getting questions, submit the user's
    answers here to continue the planning process.

    Args:
        request: Contains thread_id and answers dict

    Returns:
        Status and thread_id for retrieving results
    """
    graph = get_graph()
    config = {
        "configurable": {"thread_id": request.thread_id},
        "recursion_limit": MAX_GRAPH_ITERATIONS + 20,
    }

    # Get current state
    current_state = await graph.aget_state(config)
    if not current_state.values:
        return {"success": False, "error": "Thread not found"}

    # Update state with answers
    updated_state = dict(current_state.values)
    updated_state["clarification_answers"] = request.answers

    # Continue the graph from process_answers
    try:
        result = await graph.ainvoke(updated_state, config)
        has_itinerary = result.get("final_itinerary") is not None

        return {
            "success": True,
            "thread_id": request.thread_id,
            "completed": has_itinerary,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/{thread_id}")
async def get_planning_result(thread_id: str):
    """Get the planning result for a thread.

    Use this after planning completes to retrieve the final itinerary.

    Args:
        thread_id: The thread ID from `/plan/clarify`

    Returns:
        The final itinerary if available, or current status
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    state = await graph.aget_state(config)
    if not state.values:
        return {"success": False, "error": "Thread not found"}

    values = state.values
    return {
        "success": True,
        "thread_id": thread_id,
        "completed": values.get("final_itinerary") is not None,
        "itinerary": values.get("final_itinerary"),
        "validation_result": values.get("validation_result"),
    }
