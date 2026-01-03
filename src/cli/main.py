"""Typer CLI for the Travel Planner."""

import asyncio
import json
import sys
import uuid
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

from src.config.constants import MAX_GRAPH_ITERATIONS
from src.graph.workflow import create_travel_graph
from src.models.state import get_initial_state
from src.cache.browser_cache import BrowserCache
from src.tools.browser.browser_manager import BrowserManager

app = typer.Typer(
    name="travel",
    help="Agentic Travel Planner - Plan trips with AI agents",
    no_args_is_help=True,
)
console = Console()


@app.command()
def plan(
    request: str = typer.Argument(
        ...,
        help="Natural language trip request (e.g., 'Plan a 5-day trip to Rajasthan')",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path for JSON itinerary",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed agent progress",
    ),
):
    """Plan a trip based on natural language input.

    Example:
        travel plan "Plan a 7-day trip to Japan visiting Tokyo and Kyoto"
    """
    console.print(Panel(
        f"[bold blue]Planning trip:[/bold blue] {request}",
        title="Travel Planner",
        border_style="blue",
    ))

    try:
        result = asyncio.run(_run_planning(request, verbose))

        if result.get("final_itinerary"):
            _display_itinerary(result["final_itinerary"])

            if output:
                with open(output, "w") as f:
                    json.dump(result["final_itinerary"], f, indent=2, default=str)
                console.print(f"\n[green]Itinerary saved to:[/green] {output}")
        else:
            console.print("[red]Failed to generate itinerary[/red]")
            if verbose and result.get("validation_result"):
                console.print(f"Validation issues: {result['validation_result']}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Planning cancelled[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)
    finally:
        # Cleanup browser
        asyncio.run(BrowserManager.shutdown())


async def _run_planning(request: str, verbose: bool) -> dict:
    """Run the planning workflow with interactive clarification."""
    graph = create_travel_graph()
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_GRAPH_ITERATIONS + 20,
    }

    # Phase 1: Run until clarification completes (may pause for questions)
    console.print("[dim]Analyzing your request...[/dim]")
    initial_state = get_initial_state(request)
    result = await graph.ainvoke(initial_state, config)

    # Check if we need to ask clarification questions
    if result.get("clarification_needed") and result.get("clarification_questions"):
        questions = result["clarification_questions"]

        console.print()
        console.print(Panel(
            "[bold]Before I plan your trip, I have a few questions:[/bold]",
            border_style="yellow",
        ))
        console.print()

        answers = {}
        for q in questions:
            q_id = q.get("question_id", "unknown")
            q_text = q.get("question_text", "")
            q_type = q.get("question_type", "")
            options = q.get("options", [])

            # Special handling for travel dates
            if q_type == "travel_dates" or q_id == "travel_dates":
                console.print(f"[cyan]{q_text}[/cyan]")
                console.print("[dim]Format: 'Jan 15-22, 2026' or describe like 'mid-January 2026'[/dim]")
                console.print("[dim]You can also say 'flexible' or 'around mid-February'[/dim]")
                answer = Prompt.ask("Your travel dates")
            elif options:
                # Show options as numbered list
                console.print(f"[cyan]{q_text}[/cyan]")
                for i, opt in enumerate(options, 1):
                    console.print(f"  {i}. {opt}")
                answer = Prompt.ask("Your choice (number or type your own)")

                # If user entered a number, map to option
                try:
                    idx = int(answer) - 1
                    if 0 <= idx < len(options):
                        answer = options[idx]
                except ValueError:
                    pass  # User typed their own answer
            else:
                answer = Prompt.ask(f"[cyan]{q_text}[/cyan]")

            answers[q_id] = answer
            console.print()

        # Update state with answers and resume from process_answers node
        result["clarification_answers"] = answers

        console.print("[dim]Got it! Now planning your personalized trip...[/dim]")
        console.print()

        # Resume the graph - invoke process_answers then continue
        result = await graph.ainvoke(result, config, interrupt_before=["process_answers"])
        # Now run from process_answers onwards
        result = await graph.ainvoke(result, config)

    # Phase 2: Show progress during main planning
    if not result.get("final_itinerary"):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Planning your trip...", total=None)

            # Run the full planning if it hasn't completed yet
            if not result.get("final_itinerary"):
                result = await graph.ainvoke(result, config)

            progress.update(task, description="[green]Planning complete![/green]")

    # Show verbose output if requested
    if verbose:
        messages = result.get("messages", [])
        for msg in messages:
            if hasattr(msg, "name") and msg.name:
                console.print(f"  [dim]{msg.content}[/dim]")

    return result


def _display_itinerary(itinerary: dict):
    """Display the itinerary in a formatted way."""
    console.print()

    # Title and summary
    console.print(Panel(
        f"[bold]{itinerary.get('trip_title', 'Your Trip')}[/bold]\n\n"
        f"{itinerary.get('destination_summary', '')}",
        title="Itinerary",
        border_style="green",
    ))

    # Trip overview table
    overview = Table(title="Trip Overview", show_header=False)
    overview.add_column("Field", style="cyan")
    overview.add_column("Value")

    overview.add_row("Duration", f"{itinerary.get('total_days', '?')} days")
    overview.add_row("Cities", ", ".join(itinerary.get("cities_visited", [])))
    overview.add_row("Budget Level", itinerary.get("budget_level", "mid_range"))
    overview.add_row("Estimated Cost", f"${itinerary.get('total_estimated_cost_usd', 0):.0f} USD")

    console.print(overview)
    console.print()

    # Daily plans
    daily_plans = itinerary.get("daily_plans", [])
    if daily_plans:
        console.print("[bold]Daily Plans:[/bold]")
        console.print()

        for day in daily_plans:
            day_table = Table(
                title=f"Day {day.get('day_number', '?')}: {day.get('city', 'Unknown')}",
                show_header=True,
                header_style="bold magenta",
            )
            day_table.add_column("Time", style="cyan", width=12)
            day_table.add_column("Activity", style="green")
            day_table.add_column("Details")

            for activity in day.get("activities", []):
                time_slot = activity.get("time_slot", "")
                title = activity.get("title", "Activity")
                details = ""

                if activity.get("attraction"):
                    attr = activity["attraction"]
                    details = f"{attr.get('category', '')}, ~{attr.get('estimated_duration_hours', '?')}h"
                elif activity.get("meal"):
                    meal = activity["meal"]
                    details = f"${meal.get('estimated_cost_usd', '?')}"

                day_table.add_row(time_slot, title, details)

            console.print(day_table)
            console.print()

    # Origin transport (if available)
    origin_transport = itinerary.get("origin_transport")
    if origin_transport:
        console.print("[bold]Getting There:[/bold]")
        rec = origin_transport.get("recommended", {})
        console.print(
            f"  {origin_transport.get('from_location')} → {origin_transport.get('to_location')}: "
            f"{rec.get('mode', 'N/A')} ({rec.get('duration_hours', '?')}h, ~${rec.get('estimated_cost_usd', '?')})"
        )
        if rec.get("notes"):
            console.print(f"    [dim]{rec.get('notes')}[/dim]")
        console.print()

    # Inter-city transport
    inter_city = itinerary.get("inter_city_transport", [])
    if inter_city:
        console.print("[bold]Inter-City Transport:[/bold]")
        transport_table = Table(show_header=True, header_style="bold")
        transport_table.add_column("Route", style="cyan")
        transport_table.add_column("Mode")
        transport_table.add_column("Duration")
        transport_table.add_column("Cost")

        for t in inter_city:
            rec = t.get("recommended", {})
            transport_table.add_row(
                f"{t.get('from_location')} → {t.get('to_location')}",
                rec.get("mode", "N/A"),
                f"{rec.get('duration_hours', '?')}h",
                f"${rec.get('estimated_cost_usd', '?')}",
            )

        console.print(transport_table)
        console.print()

    # Budget breakdown
    budget = itinerary.get("budget_breakdown", {})
    if budget and budget.get("total"):
        console.print("[bold]Budget Breakdown:[/bold]")
        budget_table = Table(show_header=False)
        budget_table.add_column("Category", style="cyan")
        budget_table.add_column("Amount", justify="right")

        if budget.get("transport_inter_city"):
            budget_table.add_row("Inter-city Transport", f"${budget.get('transport_inter_city', 0):.0f}")
        if budget.get("transport_local"):
            budget_table.add_row("Local Transport", f"${budget.get('transport_local', 0):.0f}")
        if budget.get("accommodation"):
            budget_table.add_row("Accommodation", f"${budget.get('accommodation', 0):.0f}")
        if budget.get("food"):
            budget_table.add_row("Food", f"${budget.get('food', 0):.0f}")
        if budget.get("activities"):
            budget_table.add_row("Activities", f"${budget.get('activities', 0):.0f}")
        if budget.get("miscellaneous"):
            budget_table.add_row("Miscellaneous", f"${budget.get('miscellaneous', 0):.0f}")
        budget_table.add_row("[bold]Total[/bold]", f"[bold]${budget.get('total', 0):.0f}[/bold]")

        console.print(budget_table)
        console.print()

        # Money saving tips
        tips_list = budget.get("money_saving_tips", [])
        if tips_list:
            console.print("[bold green]Money Saving Tips:[/bold green]")
            for tip in tips_list[:3]:
                console.print(f"  • {tip}")
            console.print()

    # Cultural tips
    tips = itinerary.get("cultural_tips", [])
    if tips:
        console.print("[bold]Cultural Tips:[/bold]")
        for tip in tips[:5]:  # Limit to 5
            console.print(f"  • {tip}")
        console.print()

    # Warnings/recommendations
    warnings = itinerary.get("warnings", [])
    if warnings:
        console.print("[bold yellow]Recommendations:[/bold yellow]")
        for warning in warnings[:3]:
            console.print(f"  • {warning}")
        console.print()


@app.command()
def cache(
    action: str = typer.Argument(
        ...,
        help="Cache action: 'clear' to clear cache, 'stats' to show stats",
    ),
):
    """Manage the browser cache.

    Examples:
        travel cache clear
        travel cache stats
    """
    cache_instance = BrowserCache.get_instance()

    if action == "clear":
        cache_instance.clear()
        console.print("[green]Cache cleared successfully[/green]")
    elif action == "stats":
        stats = cache_instance.stats()
        console.print(Panel(
            f"Items: {stats['item_count']}\n"
            f"Size: {stats['size_bytes'] / 1024:.1f} KB\n"
            f"Location: {stats['cache_dir']}",
            title="Cache Statistics",
        ))
    else:
        console.print(f"[red]Unknown action:[/red] {action}")
        console.print("Use 'clear' or 'stats'")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    console.print("Travel Planner v0.1.0")
    console.print("Multi-agent travel planning powered by LangGraph + OpenAI")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
