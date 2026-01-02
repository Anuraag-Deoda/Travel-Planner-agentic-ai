"""FastAPI application for the Travel Planner."""

import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.graph.workflow import create_travel_graph, plan_trip
from src.tools.browser.browser_manager import BrowserManager
from src.cache.browser_cache import BrowserCache


# Request/Response models
class PlanRequest(BaseModel):
    """Request model for trip planning."""

    description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Natural language trip description",
        json_schema_extra={"example": "Plan a 5-day trip to Rajasthan visiting Udaipur, Jodhpur, and Jaipur"},
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for resuming or continuing a planning session",
    )


class PlanResponse(BaseModel):
    """Response model for trip planning."""

    success: bool
    thread_id: str
    itinerary: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str = "0.1.0"


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""

    item_count: int
    size_bytes: int
    cache_dir: str


# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    yield
    # Shutdown
    await BrowserManager.shutdown()
    BrowserCache.reset_instance()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Travel Planner API",
        description="Multi-agent travel planning powered by LangGraph + OpenAI",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create graph instance
    graph = create_travel_graph()

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(status="healthy")

    @app.post("/plan", response_model=PlanResponse)
    async def create_plan(request: PlanRequest):
        """Create a travel plan from natural language input.

        This endpoint runs the full multi-agent planning workflow
        and returns the complete itinerary.
        """
        thread_id = request.thread_id or str(uuid.uuid4())

        try:
            result = await plan_trip(
                user_request=request.description,
                thread_id=thread_id,
                graph=graph,
            )

            itinerary = result.get("final_itinerary")

            if itinerary:
                return PlanResponse(
                    success=True,
                    thread_id=thread_id,
                    itinerary=itinerary,
                )
            else:
                # Planning completed but no valid itinerary
                validation = result.get("validation_result", {})
                error_msg = "Failed to generate valid itinerary"
                if validation.get("issues"):
                    issues = [i.get("description", "") for i in validation["issues"][:3]]
                    error_msg += f": {'; '.join(issues)}"

                return PlanResponse(
                    success=False,
                    thread_id=thread_id,
                    error=error_msg,
                )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Planning failed: {str(e)}",
            )

    @app.get("/cache/stats", response_model=CacheStatsResponse)
    async def cache_stats():
        """Get cache statistics."""
        cache = BrowserCache.get_instance()
        stats = cache.stats()
        return CacheStatsResponse(**stats)

    @app.post("/cache/clear")
    async def clear_cache():
        """Clear the browser cache."""
        cache = BrowserCache.get_instance()
        cache.clear()
        return {"status": "cleared"}

    return app


# Create app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
