"""FastAPI application for the Travel Planner with WebSocket support."""

import uuid
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.config.constants import MAX_GRAPH_ITERATIONS
from src.graph.workflow import create_travel_graph, plan_trip
from src.models.state import get_initial_state
from src.tools.browser.browser_manager import BrowserManager
from src.cache.browser_cache import BrowserCache

logger = logging.getLogger(__name__)

# Agent info for UI
AGENT_INFO = {
    "clarification": {"name": "Clarification Agent", "description": "Understanding your travel preferences..."},
    "process_answers": {"name": "Processing Answers", "description": "Analyzing your responses..."},
    "planner": {"name": "Trip Planner", "description": "Designing your perfect itinerary..."},
    "geography": {"name": "Geography Expert", "description": "Optimizing your travel route..."},
    "research": {"name": "Destination Researcher", "description": "Discovering the best attractions..."},
    "food_culture": {"name": "Food & Culture Guide", "description": "Finding local cuisine and cultural insights..."},
    "transport_scraper": {"name": "Transport Price Finder", "description": "Searching for real-time transport prices..."},
    "transport_budget": {"name": "Budget Calculator", "description": "Calculating your trip budget..."},
    "critic": {"name": "Plan Reviewer", "description": "Reviewing and validating your plan..."},
    "finalize": {"name": "Finalizing", "description": "Assembling your complete itinerary..."},
}


# Request/Response models
class PlanRequest(BaseModel):
    description: str = Field(..., min_length=10, max_length=2000)
    thread_id: Optional[str] = None


class PlanResponse(BaseModel):
    success: bool
    thread_id: str
    itinerary: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"


class CacheStatsResponse(BaseModel):
    item_count: int
    size_bytes: int
    cache_dir: str


# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_sessions: dict[str, dict] = {}

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            "websocket": websocket,
            "thread_id": str(uuid.uuid4()),
            "is_cancelled": False,
            "answers_event": asyncio.Event(),
            "pending_answers": None,
        }
        await self.send(session_id, {"type": "connected", "session_id": session_id})
        return session_id

    def disconnect(self, session_id: str):
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

    def get_session(self, session_id: str):
        return self.active_sessions.get(session_id)

    async def send(self, session_id: str, message: dict):
        session = self.get_session(session_id)
        if session and not session["is_cancelled"]:
            try:
                await session["websocket"].send_json(message)
            except Exception:
                self.disconnect(session_id)

    async def send_agent_start(self, session_id: str, agent: str):
        info = AGENT_INFO.get(agent, {"name": agent, "description": "Processing..."})
        await self.send(session_id, {
            "type": "agent_start",
            "agent": agent,
            "description": info["description"],
        })

    async def send_agent_complete(self, session_id: str, agent: str, summary: str, data: dict = None):
        await self.send(session_id, {
            "type": "agent_complete",
            "agent": agent,
            "summary": summary,
            "data": data or {},
        })

    async def send_questions(self, session_id: str, questions: list):
        await self.send(session_id, {"type": "questions", "questions": questions})

    async def send_complete(self, session_id: str, itinerary: dict):
        await self.send(session_id, {"type": "planning_complete", "itinerary": itinerary})

    async def send_error(self, session_id: str, error: str):
        await self.send(session_id, {"type": "error", "error": error})


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await BrowserManager.shutdown()
    BrowserCache.reset_instance()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Travel Planner API",
        description="Multi-agent travel planning powered by LangGraph + OpenAI",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    graph = create_travel_graph()
    planning_tasks: dict[str, asyncio.Task] = {}

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(status="healthy")

    @app.post("/plan", response_model=PlanResponse)
    async def create_plan(request: PlanRequest):
        thread_id = request.thread_id or str(uuid.uuid4())
        try:
            result = await plan_trip(
                user_request=request.description,
                thread_id=thread_id,
                graph=graph,
            )
            itinerary = result.get("final_itinerary")
            if itinerary:
                return PlanResponse(success=True, thread_id=thread_id, itinerary=itinerary)
            else:
                return PlanResponse(success=False, thread_id=thread_id, error="Failed to generate itinerary")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Planning failed: {str(e)}")

    @app.websocket("/ws/plan")
    async def websocket_plan(websocket: WebSocket):
        session_id = None
        try:
            session_id = await manager.connect(websocket)
            logger.info(f"WebSocket connected: {session_id}")

            while True:
                try:
                    # Use receive with timeout for keep-alive
                    message = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=300  # 5 minute timeout
                    )

                    # Handle different message types
                    if message["type"] == "websocket.disconnect":
                        logger.info(f"WebSocket disconnect received: {session_id}")
                        break

                    if message["type"] == "websocket.receive":
                        if "text" in message:
                            data = json.loads(message["text"])
                            msg_type = data.get("type")
                            logger.info(f"Received message type: {msg_type}")

                            if msg_type == "ping":
                                await manager.send(session_id, {"type": "pong"})

                            elif msg_type == "start_planning":
                                user_request = data.get("request", "")
                                if len(user_request) < 10:
                                    await manager.send_error(session_id, "Please provide more details")
                                    continue

                                task = asyncio.create_task(
                                    run_planning_with_updates(session_id, user_request, graph)
                                )
                                planning_tasks[session_id] = task

                            elif msg_type == "answer_questions":
                                session = manager.get_session(session_id)
                                if session:
                                    session["pending_answers"] = data.get("answers", {})
                                    session["answers_event"].set()

                            elif msg_type == "cancel":
                                session = manager.get_session(session_id)
                                if session:
                                    session["is_cancelled"] = True
                                if session_id in planning_tasks:
                                    planning_tasks[session_id].cancel()

                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    try:
                        await manager.send(session_id, {"type": "ping"})
                    except Exception:
                        break

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {session_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
            if session_id:
                await manager.send_error(session_id, str(e))
        finally:
            if session_id:
                manager.disconnect(session_id)
                if session_id in planning_tasks:
                    planning_tasks[session_id].cancel()
                    del planning_tasks[session_id]

    @app.get("/cache/stats", response_model=CacheStatsResponse)
    async def cache_stats():
        cache = BrowserCache.get_instance()
        stats = cache.stats()
        return CacheStatsResponse(**stats)

    @app.post("/cache/clear")
    async def clear_cache():
        cache = BrowserCache.get_instance()
        cache.clear()
        return {"status": "cleared"}

    # Serve frontend
    frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
    frontend_src = Path(__file__).parent.parent.parent / "frontend"

    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

        @app.get("/")
        async def serve_frontend():
            return FileResponse(str(frontend_dist / "index.html"))
    elif (frontend_src / "index.html").exists():
        @app.get("/")
        async def serve_dev_frontend():
            return FileResponse(str(frontend_src / "index.html"))

    return app


async def run_planning_with_updates(session_id: str, user_request: str, graph):
    """Run the planning workflow with real-time WebSocket updates."""
    session = manager.get_session(session_id)
    if not session:
        return

    thread_id = session["thread_id"]
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_GRAPH_ITERATIONS + 20,
    }

    completed_nodes = set()

    async def stream_graph(input_state):
        """Stream graph execution and track node progress."""
        nonlocal completed_nodes
        result = None

        async for event in graph.astream_events(input_state, config, version="v2"):
            if session["is_cancelled"]:
                return None

            kind = event.get("event")
            name = event.get("name", "")

            # Track node execution
            if kind == "on_chain_start" and name in AGENT_INFO and name not in completed_nodes:
                await manager.send_agent_start(session_id, name)
                logger.info(f"Started: {name}")

            elif kind == "on_chain_end" and name in AGENT_INFO:
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict) and output:
                    result = output
                    if name not in completed_nodes:
                        completed_nodes.add(name)
                        summary = get_agent_summary(name, output)
                        agent_data = get_agent_data(name, output)
                        await manager.send_agent_complete(session_id, name, summary, agent_data)
                        logger.info(f"Completed: {name}")

        return result

    try:
        # Phase 1: Run clarification
        await manager.send_agent_start(session_id, "clarification")
        initial_state = get_initial_state(user_request)

        result = await stream_graph(initial_state)

        if session["is_cancelled"] or result is None:
            return

        # Check if we need clarification
        if result.get("clarification_needed") and result.get("clarification_questions"):
            completed_nodes.add("clarification")
            await manager.send_agent_complete(session_id, "clarification", "Questions ready", {})
            await manager.send_questions(session_id, result["clarification_questions"])

            # Wait for answers
            try:
                await asyncio.wait_for(session["answers_event"].wait(), timeout=600)
            except asyncio.TimeoutError:
                await manager.send_error(session_id, "Timeout waiting for answers")
                return

            if session["is_cancelled"]:
                return

            # Process answers
            answers = session["pending_answers"] or {}
            session["answers_event"].clear()

            await manager.send_agent_start(session_id, "process_answers")
            logger.info(f"Processing answers: {answers}")

            # Update state and resume
            graph.update_state(
                config,
                {"clarification_answers": answers, "clarification_needed": False},
                as_node="clarification"
            )

            completed_nodes.add("process_answers")
            await manager.send_agent_complete(session_id, "process_answers", "Answers processed", {})

            # Resume graph
            result = await stream_graph(None)

            if session["is_cancelled"] or result is None:
                return

        # Ensure finalize is marked complete
        if "finalize" not in completed_nodes and result.get("final_itinerary"):
            await manager.send_agent_start(session_id, "finalize")
            await asyncio.sleep(0.2)
            completed_nodes.add("finalize")
            await manager.send_agent_complete(session_id, "finalize", "Itinerary complete", {})

        # Send final itinerary
        itinerary = result.get("final_itinerary")
        if itinerary:
            logger.info("Sending complete itinerary")
            await manager.send_complete(session_id, itinerary)
        else:
            logger.error(f"No itinerary in result. Keys: {result.keys() if result else 'None'}")
            await manager.send_error(session_id, "Failed to generate itinerary. Please try again.")

    except asyncio.CancelledError:
        logger.info("Planning cancelled")
    except Exception as e:
        logger.error(f"Planning error: {e}", exc_info=True)
        await manager.send_error(session_id, f"Planning error: {str(e)}")


def get_agent_summary(agent: str, result: dict) -> str:
    """Get summary for agent completion."""
    try:
        if agent == "clarification":
            return "Ready to plan"
        elif agent == "planner":
            cities = result.get("city_allocations", [])
            return f"Planned {len(cities)} cities"
        elif agent == "geography":
            return "Route optimized"
        elif agent == "research":
            return f"Found {len(result.get('attractions', []))} attractions"
        elif agent == "food_culture":
            return f"Found {len(result.get('food_recommendations', []))} restaurants"
        elif agent == "transport_scraper":
            prices = result.get("scraped_transport_prices", [])
            return f"Found {len(prices)} transport options"
        elif agent == "transport_budget":
            budget = result.get("budget_breakdown", {})
            return f"Budget: ${budget.get('total', 0):.0f}"
        elif agent == "critic":
            return "Plan validated"
        elif agent == "finalize":
            return "Itinerary complete"
    except Exception:
        pass
    return "Completed"


def get_agent_data(agent: str, result: dict) -> dict:
    """Get data for frontend display."""
    try:
        if agent == "planner":
            allocations = result.get("city_allocations", [])
            return {
                "cities": [{"name": c.get("city"), "days": c.get("days")} for c in allocations],
                "total_days": sum(c.get("days", 0) for c in allocations),
            }
        elif agent == "research":
            return {"attractions_count": len(result.get("attractions", []))}
        elif agent == "transport_budget":
            budget = result.get("budget_breakdown", {})
            return {
                "total": budget.get("total", 0),
                "breakdown": {
                    "transport": budget.get("transport_inter_city", 0) + budget.get("transport_local", 0),
                    "accommodation": budget.get("accommodation", 0),
                    "food": budget.get("food", 0),
                    "activities": budget.get("activities_entrance_fees", 0),
                },
            }
    except Exception:
        pass
    return {}


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("src.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
