"""Transport price scraping tools using Playwright."""

import json
from datetime import datetime
from typing import Optional
from urllib.parse import quote_plus

from langchain_core.tools import tool

from src.cache.browser_cache import BrowserCache
from src.cache.transport_cache import (
    transport_price_key,
    station_info_key,
    get_transport_cache_ttl,
    STATION_CACHE_TTL,
)
from src.tools.browser.browser_manager import BrowserManager, navigate_and_wait


# Currency conversion rates (approximate, for display purposes)
INR_TO_USD = 0.012
THB_TO_USD = 0.028


@tool
async def scrape_google_flights(
    from_city: str,
    to_city: str,
    travel_date: str,
    return_date: Optional[str] = None,
    include_alternatives: bool = True,
) -> str:
    """Scrape flight prices from Google Flights.

    Args:
        from_city: Origin city or airport code.
        to_city: Destination city or airport code.
        travel_date: Travel date in YYYY-MM-DD format.
        return_date: Optional return date for round trip.
        include_alternatives: Whether to check nearby dates for cheaper options.

    Returns:
        JSON string with flight price data.
    """
    cache = BrowserCache.get_instance()
    cache_key = transport_price_key("flight", from_city, to_city, travel_date)

    cached = cache.get(cache_key)
    if cached:
        return cached

    # Build Google Flights URL
    date_formatted = datetime.fromisoformat(travel_date).strftime("%B %d")
    search_query = f"flights from {from_city} to {to_city} on {date_formatted}"
    url = f"https://www.google.com/travel/flights?q={quote_plus(search_query)}"

    result = {
        "source": "google_flights",
        "from_city": from_city,
        "to_city": to_city,
        "travel_date": travel_date,
        "flights": [],
        "alternative_dates": [],
        "error": None,
    }

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, url, timeout=45000)

            if not success:
                result["error"] = "Failed to load Google Flights"
                return json.dumps(result)

            # Wait for results to load
            await page.wait_for_timeout(3000)

            # Extract flight data using JavaScript
            flights_data = await page.evaluate("""() => {
                const flights = [];

                // Google Flights uses various selectors
                const flightCards = document.querySelectorAll('[data-ved] .pIav2d, .yR1fYc, [jsname="IWWDBc"]');

                flightCards.forEach((card, idx) => {
                    if (idx >= 10) return;

                    try {
                        // Price extraction
                        const priceEl = card.querySelector('[data-gs], .YMlIz, .FpEdX span');
                        const price = priceEl ? priceEl.textContent.replace(/[^0-9]/g, '') : null;

                        // Time extraction
                        const timeEls = card.querySelectorAll('.mv1WYe span, .Ir0Voe, .zxVSec');
                        const times = Array.from(timeEls).map(el => el.textContent);

                        // Duration
                        const durationEl = card.querySelector('.gvkrdb, .AdWm1c, .Ak5kof');
                        const duration = durationEl ? durationEl.textContent : null;

                        // Airline
                        const airlineEl = card.querySelector('.h1fkLb, .Xsgmwe, .sSHqwe');
                        const airline = airlineEl ? airlineEl.textContent : null;

                        if (price) {
                            flights.push({
                                price: parseInt(price),
                                departure_time: times[0] || null,
                                arrival_time: times[1] || null,
                                duration: duration,
                                airline: airline,
                            });
                        }
                    } catch (e) {}
                });

                return flights;
            }""")

            for flight in flights_data:
                if flight.get("price"):
                    result["flights"].append({
                        "price_usd": flight["price"],
                        "departure_time": flight.get("departure_time"),
                        "arrival_time": flight.get("arrival_time"),
                        "duration": flight.get("duration"),
                        "operator": flight.get("airline"),
                        "availability": "available",
                    })

    except Exception as e:
        result["error"] = str(e)

    json_result = json.dumps(result)
    ttl = get_transport_cache_ttl(from_city, to_city)
    cache.set(cache_key, json_result, ttl=ttl)

    return json_result


@tool
async def scrape_rome2rio(
    from_city: str,
    to_city: str,
    travel_date: Optional[str] = None,
) -> str:
    """Scrape multi-modal transport options from Rome2Rio.

    Rome2Rio provides flights, trains, buses, and driving options between cities.

    Args:
        from_city: Origin city.
        to_city: Destination city.
        travel_date: Optional travel date (Rome2Rio often shows general prices).

    Returns:
        JSON string with transport options and prices.
    """
    cache = BrowserCache.get_instance()
    cache_key = transport_price_key("multimodal", from_city, to_city, travel_date or "general")

    cached = cache.get(cache_key)
    if cached:
        return cached

    # Build Rome2Rio URL
    url = f"https://www.rome2rio.com/s/{quote_plus(from_city)}/{quote_plus(to_city)}"

    result = {
        "source": "rome2rio",
        "from_city": from_city,
        "to_city": to_city,
        "options": [],
        "error": None,
    }

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, url, timeout=45000)

            if not success:
                result["error"] = "Failed to load Rome2Rio"
                return json.dumps(result)

            await page.wait_for_timeout(2000)

            # Extract transport options
            options_data = await page.evaluate("""() => {
                const options = [];

                // Rome2Rio route cards
                const cards = document.querySelectorAll('.pure-u-1.pure-u-md-1-2, .route-overview, [class*="SearchResult"]');

                cards.forEach((card, idx) => {
                    if (idx >= 6) return;

                    try {
                        const modeEl = card.querySelector('h3, .transport-title, [class*="title"]');
                        const priceEl = card.querySelector('.price, .fare, [class*="price"]');
                        const durationEl = card.querySelector('.duration, .time, [class*="duration"]');

                        if (modeEl && priceEl) {
                            const mode = modeEl.textContent.toLowerCase();
                            let transportType = 'other';
                            if (mode.includes('fly')) transportType = 'flight';
                            else if (mode.includes('train')) transportType = 'train';
                            else if (mode.includes('bus')) transportType = 'bus';
                            else if (mode.includes('drive')) transportType = 'car';
                            else if (mode.includes('ferry')) transportType = 'ferry';

                            options.push({
                                mode: transportType,
                                title: modeEl.textContent.trim(),
                                price: priceEl.textContent.replace(/[^0-9]/g, ''),
                                duration: durationEl ? durationEl.textContent : null,
                            });
                        }
                    } catch (e) {}
                });

                return options;
            }""")

            for option in options_data:
                if option.get("price"):
                    result["options"].append({
                        "mode": option.get("mode", "unknown"),
                        "title": option.get("title"),
                        "price_usd": int(option.get("price", 0)),
                        "duration": option.get("duration"),
                    })

    except Exception as e:
        result["error"] = str(e)

    json_result = json.dumps(result)
    ttl = get_transport_cache_ttl(from_city, to_city)
    cache.set(cache_key, json_result, ttl=ttl)

    return json_result


@tool
async def scrape_12go_asia(
    from_city: str,
    to_city: str,
    travel_date: str,
    transport_type: str = "any",
) -> str:
    """Scrape train/bus prices from 12go.asia for Asian routes.

    Args:
        from_city: Origin city.
        to_city: Destination city.
        travel_date: Travel date in YYYY-MM-DD format.
        transport_type: Filter by mode (train, bus, ferry, or any).

    Returns:
        JSON string with transport options.
    """
    cache = BrowserCache.get_instance()
    cache_key = transport_price_key(f"12go_{transport_type}", from_city, to_city, travel_date)

    cached = cache.get(cache_key)
    if cached:
        return cached

    # Build 12go.asia URL
    url = f"https://12go.asia/en/travel/{quote_plus(from_city.lower())}/{quote_plus(to_city.lower())}?date={travel_date}"

    result = {
        "source": "12go_asia",
        "from_city": from_city,
        "to_city": to_city,
        "travel_date": travel_date,
        "options": [],
        "error": None,
    }

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, url, timeout=45000)

            if not success:
                result["error"] = "Failed to load 12go.asia"
                return json.dumps(result)

            await page.wait_for_timeout(3000)

            # Extract transport options
            options_data = await page.evaluate("""() => {
                const options = [];

                // 12go.asia result cards
                const cards = document.querySelectorAll('.search-result-item, .route-item, [class*="TicketCard"]');

                cards.forEach((card, idx) => {
                    if (idx >= 10) return;

                    try {
                        const operatorEl = card.querySelector('.operator-name, .company, [class*="operator"]');
                        const priceEl = card.querySelector('.price-value, .fare-amount, [class*="price"]');
                        const durationEl = card.querySelector('.duration, .travel-time, [class*="duration"]');
                        const departureEl = card.querySelector('.departure-time, .depart, [class*="departure"]');
                        const arrivalEl = card.querySelector('.arrival-time, .arrive, [class*="arrival"]');
                        const classEl = card.querySelector('.class-type, .seat-class, [class*="class"]');
                        const modeEl = card.querySelector('.transport-type, .vehicle-type, [class*="vehicle"]');

                        if (priceEl) {
                            options.push({
                                mode: modeEl ? modeEl.textContent.trim().toLowerCase() : 'unknown',
                                operator: operatorEl ? operatorEl.textContent.trim() : null,
                                price: priceEl.textContent.replace(/[^0-9.]/g, ''),
                                duration: durationEl ? durationEl.textContent.trim() : null,
                                departure_time: departureEl ? departureEl.textContent.trim() : null,
                                arrival_time: arrivalEl ? arrivalEl.textContent.trim() : null,
                                class_type: classEl ? classEl.textContent.trim() : null,
                            });
                        }
                    } catch (e) {}
                });

                return options;
            }""")

            for option in options_data:
                if option.get("price"):
                    result["options"].append({
                        "mode": option.get("mode", "unknown"),
                        "operator": option.get("operator"),
                        "price_usd": float(option.get("price", 0)),
                        "duration": option.get("duration"),
                        "departure_time": option.get("departure_time"),
                        "arrival_time": option.get("arrival_time"),
                        "class_type": option.get("class_type"),
                    })

    except Exception as e:
        result["error"] = str(e)

    json_result = json.dumps(result)
    ttl = get_transport_cache_ttl(from_city, to_city)
    cache.set(cache_key, json_result, ttl=ttl)

    return json_result


@tool
async def scrape_redbus(
    from_city: str,
    to_city: str,
    travel_date: str,
) -> str:
    """Scrape bus prices from RedBus for India routes.

    Args:
        from_city: Origin city in India.
        to_city: Destination city in India.
        travel_date: Travel date in YYYY-MM-DD format.

    Returns:
        JSON string with bus options.
    """
    cache = BrowserCache.get_instance()
    cache_key = transport_price_key("redbus", from_city, to_city, travel_date)

    cached = cache.get(cache_key)
    if cached:
        return cached

    # Format date for RedBus (DD-MMM-YYYY)
    date_obj = datetime.fromisoformat(travel_date)
    date_formatted = date_obj.strftime("%d-%b-%Y")

    url = f"https://www.redbus.in/bus-tickets/{quote_plus(from_city.lower())}-to-{quote_plus(to_city.lower())}?date={date_formatted}"

    result = {
        "source": "redbus",
        "from_city": from_city,
        "to_city": to_city,
        "travel_date": travel_date,
        "buses": [],
        "error": None,
    }

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, url, timeout=45000)

            if not success:
                result["error"] = "Failed to load RedBus"
                return json.dumps(result)

            await page.wait_for_timeout(3000)

            # Extract bus data
            buses_data = await page.evaluate("""() => {
                const buses = [];

                const cards = document.querySelectorAll('.bus-item, .bus-items, [class*="bus-card"]');

                cards.forEach((card, idx) => {
                    if (idx >= 10) return;

                    try {
                        const operatorEl = card.querySelector('.travels, .operator-name, [class*="travels"]');
                        const priceEl = card.querySelector('.fare .f-19, .seat-fare, [class*="fare"]');
                        const departureEl = card.querySelector('.dp-time, .departure, [class*="departure"]');
                        const arrivalEl = card.querySelector('.bp-time, .arrival, [class*="arrival"]');
                        const durationEl = card.querySelector('.dur, .duration, [class*="duration"]');
                        const typeEl = card.querySelector('.bus-type, .vehicle-type, [class*="busType"]');
                        const seatsEl = card.querySelector('.seat-left, .seats, [class*="seats"]');
                        const ratingEl = card.querySelector('.rating, [class*="rating"]');

                        if (priceEl) {
                            buses.push({
                                operator: operatorEl ? operatorEl.textContent.trim() : null,
                                price_inr: priceEl.textContent.replace(/[^0-9]/g, ''),
                                departure_time: departureEl ? departureEl.textContent.trim() : null,
                                arrival_time: arrivalEl ? arrivalEl.textContent.trim() : null,
                                duration: durationEl ? durationEl.textContent.trim() : null,
                                bus_type: typeEl ? typeEl.textContent.trim() : null,
                                seats_available: seatsEl ? seatsEl.textContent : null,
                                rating: ratingEl ? ratingEl.textContent : null,
                            });
                        }
                    } catch (e) {}
                });

                return buses;
            }""")

            for bus in buses_data:
                if bus.get("price_inr"):
                    price_inr = int(bus["price_inr"])
                    result["buses"].append({
                        "mode": "bus",
                        "operator": bus.get("operator"),
                        "price_usd": round(price_inr * INR_TO_USD, 2),
                        "price_inr": price_inr,
                        "departure_time": bus.get("departure_time"),
                        "arrival_time": bus.get("arrival_time"),
                        "duration": bus.get("duration"),
                        "class_type": bus.get("bus_type"),
                        "availability": "limited" if bus.get("seats_available") else "available",
                        "rating": bus.get("rating"),
                    })

    except Exception as e:
        result["error"] = str(e)

    json_result = json.dumps(result)
    ttl = get_transport_cache_ttl(from_city, to_city)
    cache.set(cache_key, json_result, ttl=ttl)

    return json_result


@tool
async def scrape_trainman(
    from_station: str,
    to_station: str,
    travel_date: str,
) -> str:
    """Scrape train info from Trainman for India routes.

    Args:
        from_station: Origin station name or code.
        to_station: Destination station name or code.
        travel_date: Travel date in YYYY-MM-DD format.

    Returns:
        JSON string with train options.
    """
    cache = BrowserCache.get_instance()
    cache_key = transport_price_key("trainman", from_station, to_station, travel_date)

    cached = cache.get(cache_key)
    if cached:
        return cached

    # Trainman URL format
    url = f"https://www.trainman.in/trains/{quote_plus(from_station.lower())}-to-{quote_plus(to_station.lower())}"

    result = {
        "source": "trainman",
        "from_station": from_station,
        "to_station": to_station,
        "travel_date": travel_date,
        "trains": [],
        "error": None,
    }

    try:
        async with BrowserManager.get_page() as page:
            success = await navigate_and_wait(page, url, timeout=45000)

            if not success:
                result["error"] = "Failed to load Trainman"
                return json.dumps(result)

            await page.wait_for_timeout(3000)

            # Extract train data
            trains_data = await page.evaluate("""() => {
                const trains = [];

                const rows = document.querySelectorAll('.train-list-row, tr[data-train], [class*="trainCard"]');

                rows.forEach((row, idx) => {
                    if (idx >= 10) return;

                    try {
                        const nameEl = row.querySelector('.train-name, .name, [class*="trainName"]');
                        const numberEl = row.querySelector('.train-number, .number, [class*="trainNum"]');
                        const departureEl = row.querySelector('.departure, .dept, [class*="depart"]');
                        const arrivalEl = row.querySelector('.arrival, .arr, [class*="arrive"]');
                        const durationEl = row.querySelector('.duration, .dur, [class*="duration"]');

                        // Get prices for different classes
                        const sleeperEl = row.querySelector('.sl-fare, [data-class="SL"], [class*="SL"]');
                        const ac3El = row.querySelector('.3a-fare, [data-class="3A"], [class*="3A"]');
                        const ac2El = row.querySelector('.2a-fare, [data-class="2A"], [class*="2A"]');
                        const ac1El = row.querySelector('.1a-fare, [data-class="1A"], [class*="1A"]');

                        if (nameEl) {
                            trains.push({
                                name: nameEl.textContent.trim(),
                                number: numberEl ? numberEl.textContent.trim() : null,
                                departure: departureEl ? departureEl.textContent.trim() : null,
                                arrival: arrivalEl ? arrivalEl.textContent.trim() : null,
                                duration: durationEl ? durationEl.textContent.trim() : null,
                                prices: {
                                    sleeper: sleeperEl ? sleeperEl.textContent.replace(/[^0-9]/g, '') : null,
                                    ac3: ac3El ? ac3El.textContent.replace(/[^0-9]/g, '') : null,
                                    ac2: ac2El ? ac2El.textContent.replace(/[^0-9]/g, '') : null,
                                    ac1: ac1El ? ac1El.textContent.replace(/[^0-9]/g, '') : null,
                                }
                            });
                        }
                    } catch (e) {}
                });

                return trains;
            }""")

            for train in trains_data:
                prices = train.get("prices", {})
                # Create entry for each class with price
                for class_name, price_str in prices.items():
                    if price_str:
                        try:
                            price_inr = int(price_str)
                            result["trains"].append({
                                "mode": "train",
                                "name": train.get("name"),
                                "number": train.get("number"),
                                "departure_time": train.get("departure"),
                                "arrival_time": train.get("arrival"),
                                "duration": train.get("duration"),
                                "class_type": class_name.upper(),
                                "price_usd": round(price_inr * INR_TO_USD, 2),
                                "price_inr": price_inr,
                            })
                        except ValueError:
                            pass

    except Exception as e:
        result["error"] = str(e)

    json_result = json.dumps(result)
    ttl = get_transport_cache_ttl(from_station, to_station)
    cache.set(cache_key, json_result, ttl=ttl)

    return json_result


@tool
async def find_nearest_stations(
    city: str,
    country: str,
) -> str:
    """Find the nearest airport and train/bus stations for a city.

    Useful for cities without direct transport options.

    Args:
        city: City name.
        country: Country name.

    Returns:
        JSON string with nearest station information.
    """
    cache = BrowserCache.get_instance()
    cache_key = station_info_key(city, country)

    cached = cache.get(cache_key)
    if cached:
        return cached

    result = {
        "city": city,
        "country": country,
        "airport": None,
        "train_station": None,
        "bus_station": None,
        "error": None,
    }

    try:
        # Search for airport info
        search_url = f"https://www.google.com/search?q=nearest+airport+to+{quote_plus(city)}+{quote_plus(country)}"

        async with BrowserManager.get_page() as page:
            await navigate_and_wait(page, search_url)
            await page.wait_for_timeout(2000)

            # Extract airport info from search results
            airport_info = await page.evaluate("""() => {
                const featured = document.querySelector('.hgKElc, .IZ6rdc, .kp-header');
                const snippets = document.querySelectorAll('.VwiC3b');

                let text = featured ? featured.textContent : '';
                snippets.forEach((s, i) => {
                    if (i < 3) text += ' ' + s.textContent;
                });

                return text;
            }""")

            result["airport_info"] = airport_info[:500] if airport_info else None

            # Search for train station
            train_url = f"https://www.google.com/search?q=main+railway+station+{quote_plus(city)}+{quote_plus(country)}"
            await navigate_and_wait(page, train_url)
            await page.wait_for_timeout(2000)

            train_info = await page.evaluate("""() => {
                const featured = document.querySelector('.hgKElc, .IZ6rdc');
                return featured ? featured.textContent : null;
            }""")

            result["train_station_info"] = train_info[:500] if train_info else None

    except Exception as e:
        result["error"] = str(e)

    json_result = json.dumps(result)
    cache.set(cache_key, json_result, ttl=STATION_CACHE_TTL)

    return json_result
