"""Transport Scraper Agent - Fetches real prices before budget estimation."""

import json
from datetime import datetime, timedelta
from typing import Any, Optional

from src.models.state import AgentState
from src.models.transport_price import PriceSource
from src.tools.browser.transport_scrapers import (
    scrape_google_flights,
    scrape_rome2rio,
    scrape_12go_asia,
    scrape_redbus,
    scrape_trainman,
    find_nearest_stations,
)


# Region detection for choosing appropriate scrapers
INDIA_CITIES = {
    "delhi", "mumbai", "bangalore", "bengaluru", "chennai", "kolkata",
    "hyderabad", "pune", "jaipur", "udaipur", "jodhpur", "goa", "agra",
    "varanasi", "lucknow", "kochi", "trivandrum", "mysore", "shimla",
    "manali", "rishikesh", "haridwar", "amritsar", "chandigarh", "ahmedabad",
    "surat", "indore", "bhopal", "nagpur", "aurangabad", "nashik", "coimbatore",
    "madurai", "thiruvananthapuram", "cochin", "ooty", "munnar", "alleppey",
    "darjeeling", "gangtok", "leh", "srinagar", "guwahati", "shillong",
    "pondicherry", "mahabalipuram", "rameswaram", "tirupati", "shirdi",
    "mount abu", "pushkar", "khajuraho", "hampi", "gokarna", "varkala",
}

ASIA_COUNTRIES = {"thailand", "vietnam", "cambodia", "laos", "malaysia", "indonesia", "myanmar", "philippines"}


class TransportScraperAgent:
    """Agent that scrapes real transport prices before budget estimation.

    This agent:
    - Determines appropriate scraping sources based on route
    - Fetches real prices for flights, trains, buses
    - Finds nearest stations for cities without direct routes
    - Provides alternative date pricing when available
    - Falls back gracefully if scraping fails
    """

    agent_name = "transport_scraper"

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Scrape transport prices for all route segments.

        Args:
            state: Current graph state with route_segments and travel_dates.

        Returns:
            State updates with scraped_transport_prices and nearest_stations.
        """
        route_segments = state.get("route_segments", [])
        city_allocations = state.get("city_allocations", [])
        origin_city = state.get("origin_city")
        travel_start_date = state.get("travel_start_date")
        travel_end_date = state.get("travel_end_date")

        if not route_segments and not origin_city:
            return {
                "scraped_transport_prices": [],
                "nearest_stations": {},
            }

        scraped_prices = []
        nearest_stations = {}

        # Calculate travel date for each segment
        segment_dates = self._calculate_segment_dates(
            city_allocations, travel_start_date
        )

        # If we have an origin city, scrape origin -> first destination
        if origin_city and city_allocations:
            sorted_cities = sorted(city_allocations, key=lambda x: x.get("visit_order", 0))
            if sorted_cities:
                first_city = sorted_cities[0]
                first_destination = first_city.get("city")
                first_country = first_city.get("country", "")

                if first_destination:
                    origin_prices = await self._scrape_segment(
                        from_city=origin_city,
                        to_city=first_destination,
                        country=first_country,
                        travel_date=travel_start_date,
                        is_international=self._is_international(origin_city, first_country),
                    )
                    scraped_prices.extend(origin_prices)

        # Scrape each route segment
        for segment in route_segments:
            from_city = segment.get("from_city")
            to_city = segment.get("to_city")

            if not from_city or not to_city:
                continue

            # Get country context
            country = self._get_country_for_city(to_city, city_allocations)

            # Determine travel date for this segment
            segment_date = segment_dates.get(from_city, travel_start_date)

            segment_prices = await self._scrape_segment(
                from_city=from_city,
                to_city=to_city,
                country=country,
                travel_date=segment_date,
                recommended_mode=segment.get("recommended_transport"),
            )
            scraped_prices.extend(segment_prices)

            # Check for nearest stations if no results found
            if not segment_prices:
                for city in [from_city, to_city]:
                    if city.lower() not in nearest_stations:
                        station_info = await self._find_stations(city, country)
                        if station_info:
                            nearest_stations[city] = station_info

        return {
            "scraped_transport_prices": scraped_prices,
            "nearest_stations": nearest_stations,
        }

    async def _scrape_segment(
        self,
        from_city: str,
        to_city: str,
        country: str,
        travel_date: Optional[str],
        recommended_mode: Optional[str] = None,
        is_international: bool = False,
    ) -> list[dict]:
        """Scrape prices for a single segment using appropriate sources."""
        results = []

        # Use default date if not provided
        if not travel_date:
            # Default to 30 days from now
            travel_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

        # Determine which scrapers to use based on region and mode
        scrapers_to_use = self._select_scrapers(
            from_city, to_city, country, recommended_mode, is_international
        )

        for scraper_name, scraper_func, kwargs in scrapers_to_use:
            try:
                raw_result = await scraper_func.ainvoke(kwargs)
                parsed = json.loads(raw_result)

                if parsed.get("error"):
                    continue

                # Normalize results to common format
                normalized = self._normalize_scrape_result(
                    scraper_name, parsed, from_city, to_city, travel_date
                )
                results.extend(normalized)

            except Exception:
                # Log but continue with other scrapers
                continue

        return results

    def _select_scrapers(
        self,
        from_city: str,
        to_city: str,
        country: str,
        recommended_mode: Optional[str],
        is_international: bool,
    ) -> list[tuple]:
        """Select appropriate scrapers based on route characteristics."""
        scrapers = []

        from_lower = from_city.lower()
        to_lower = to_city.lower()
        country_lower = country.lower()

        # Check if India route
        is_india = (
            from_lower in INDIA_CITIES or
            to_lower in INDIA_CITIES or
            country_lower == "india"
        )

        # Check if Southeast Asia
        is_se_asia = country_lower in ASIA_COUNTRIES

        # Default travel date for scraper calls
        travel_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

        # Always try Rome2Rio for comprehensive options
        scrapers.append((
            "rome2rio",
            scrape_rome2rio,
            {"from_city": from_city, "to_city": to_city}
        ))

        # Flights for longer distances or international
        if is_international or recommended_mode == "flight":
            scrapers.append((
                "google_flights",
                scrape_google_flights,
                {
                    "from_city": from_city,
                    "to_city": to_city,
                    "travel_date": travel_date,
                    "include_alternatives": True,
                }
            ))

        # India-specific scrapers
        if is_india:
            if recommended_mode in (None, "bus"):
                scrapers.append((
                    "redbus",
                    scrape_redbus,
                    {"from_city": from_city, "to_city": to_city, "travel_date": travel_date}
                ))

            if recommended_mode in (None, "train"):
                scrapers.append((
                    "trainman",
                    scrape_trainman,
                    {"from_station": from_city, "to_station": to_city, "travel_date": travel_date}
                ))

        # Southeast Asia scrapers
        if is_se_asia:
            scrapers.append((
                "12go_asia",
                scrape_12go_asia,
                {
                    "from_city": from_city,
                    "to_city": to_city,
                    "travel_date": travel_date,
                }
            ))

        return scrapers

    def _normalize_scrape_result(
        self,
        source: str,
        parsed: dict,
        from_city: str,
        to_city: str,
        travel_date: Optional[str],
    ) -> list[dict]:
        """Normalize scraped results to common format."""
        normalized = []

        if source == "google_flights":
            for flight in parsed.get("flights", []):
                normalized.append({
                    "source": PriceSource.GOOGLE_FLIGHTS.value,
                    "mode": "flight",
                    "from_location": from_city,
                    "to_location": to_city,
                    "travel_date": travel_date,
                    "price_usd": flight.get("price_usd"),
                    "departure_time": flight.get("departure_time"),
                    "arrival_time": flight.get("arrival_time"),
                    "operator": flight.get("operator"),
                    "duration": flight.get("duration"),
                    "alternative_dates": parsed.get("alternative_dates", []),
                })

        elif source == "rome2rio":
            for option in parsed.get("options", []):
                normalized.append({
                    "source": PriceSource.ROME2RIO.value,
                    "mode": option.get("mode", "unknown"),
                    "from_location": from_city,
                    "to_location": to_city,
                    "travel_date": travel_date,
                    "price_usd": option.get("price_usd"),
                    "duration": option.get("duration"),
                    "title": option.get("title"),
                })

        elif source == "redbus":
            for bus in parsed.get("buses", []):
                normalized.append({
                    "source": PriceSource.REDBUS.value,
                    "mode": "bus",
                    "from_location": from_city,
                    "to_location": to_city,
                    "travel_date": travel_date,
                    "price_usd": bus.get("price_usd"),
                    "price_local": bus.get("price_inr"),
                    "currency_local": "INR",
                    "departure_time": bus.get("departure_time"),
                    "arrival_time": bus.get("arrival_time"),
                    "operator": bus.get("operator"),
                    "class_type": bus.get("class_type"),
                    "duration": bus.get("duration"),
                    "availability": bus.get("availability", "available"),
                })

        elif source == "trainman":
            for train in parsed.get("trains", []):
                normalized.append({
                    "source": PriceSource.TRAINMAN.value,
                    "mode": "train",
                    "from_location": from_city,
                    "to_location": to_city,
                    "travel_date": travel_date,
                    "price_usd": train.get("price_usd"),
                    "price_local": train.get("price_inr"),
                    "currency_local": "INR",
                    "departure_time": train.get("departure_time"),
                    "arrival_time": train.get("arrival_time"),
                    "operator": train.get("name"),
                    "train_number": train.get("number"),
                    "class_type": train.get("class_type"),
                    "duration": train.get("duration"),
                })

        elif source == "12go_asia":
            for option in parsed.get("options", []):
                if option.get("price_usd"):
                    normalized.append({
                        "source": PriceSource.TWELVE_GO_ASIA.value,
                        "mode": option.get("mode", "unknown"),
                        "from_location": from_city,
                        "to_location": to_city,
                        "travel_date": travel_date,
                        "price_usd": option.get("price_usd"),
                        "departure_time": option.get("departure_time"),
                        "arrival_time": option.get("arrival_time"),
                        "operator": option.get("operator"),
                        "class_type": option.get("class_type"),
                        "duration": option.get("duration"),
                    })

        return normalized

    async def _find_stations(self, city: str, country: str) -> Optional[dict]:
        """Find nearest stations for a city."""
        try:
            result = await find_nearest_stations.ainvoke({
                "city": city,
                "country": country,
            })
            return json.loads(result)
        except Exception:
            return None

    def _calculate_segment_dates(
        self,
        city_allocations: list[dict],
        start_date: Optional[str],
    ) -> dict[str, str]:
        """Calculate travel date for each segment based on city days."""
        if not start_date or not city_allocations:
            return {}

        dates = {}
        sorted_cities = sorted(city_allocations, key=lambda x: x.get("visit_order", 0))

        try:
            current_date = datetime.fromisoformat(start_date)

            for city_info in sorted_cities:
                city = city_info.get("city")
                days = city_info.get("days", 1)

                if city:
                    dates[city] = current_date.strftime("%Y-%m-%d")
                    current_date += timedelta(days=days)
        except ValueError:
            pass

        return dates

    def _get_country_for_city(
        self, city: str, allocations: list[dict]
    ) -> str:
        """Get country for a city from allocations."""
        for alloc in allocations:
            if alloc.get("city") == city:
                return alloc.get("country", "")
        return ""

    def _is_international(self, origin: str, destination_country: str) -> bool:
        """Check if route is international."""
        # Simple heuristic - could be enhanced with country lookup
        if origin.lower() in INDIA_CITIES and destination_country.lower() == "india":
            return False
        if origin.lower() in INDIA_CITIES and destination_country.lower() != "india":
            return True

        # Default assumption
        return True
