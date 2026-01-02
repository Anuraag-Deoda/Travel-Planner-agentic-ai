"""Application constants."""

# Model temperature settings
PLANNER_TEMPERATURE = 0.7  # More creative for trip planning
GEOGRAPHY_TEMPERATURE = 0.2  # More deterministic for validation
RESEARCH_TEMPERATURE = 0.3  # Balanced for content extraction
FOOD_CULTURE_TEMPERATURE = 0.5  # Moderate creativity
TRANSPORT_TEMPERATURE = 0.2  # Deterministic for pricing/logistics
CRITIC_TEMPERATURE = 0.1  # Very deterministic for validation

# Agent model assignments
AGENT_MODELS = {
    "clarification": "gpt4o_mini",  # Quick question generation
    "planner": "gpt4o",  # Complex reasoning
    "geography": "gpt4o_mini",  # Simpler validation
    "research": "gpt4o",  # Content interpretation
    "food_culture": "gpt4o_mini",  # Structured recommendations
    "transport_budget": "gpt4o_mini",  # Option comparison
    "critic": "gpt4o",  # Critical validation
}

# Browser settings
BROWSER_TIMEOUT_MS = 30000
PAGE_LOAD_WAIT = "networkidle"
DEFAULT_VIEWPORT = {"width": 1920, "height": 1080}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Content limits
MAX_CONTENT_LENGTH = 10000  # Characters
MAX_ATTRACTIONS_PER_CITY = 10
MAX_RESTAURANTS_PER_CITY = 5

# Graph settings
MAX_GRAPH_ITERATIONS = 20  # Safety limit for the entire graph
