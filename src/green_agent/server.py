"""
A2A Server setup for Werewolf Green Agent.
Main entry point following green-agent-template structure.
"""
import argparse
import logging
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from .executor import WerewolfExecutor


def main():
    """Main entry point for Werewolf Green Agent server."""
    parser = argparse.ArgumentParser(description="Run the Werewolf Green Agent (A2A)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress verbose HTTP and A2A protocol logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("a2a.client.card_resolver").setLevel(logging.WARNING)

    # Define agent skill
    skill = AgentSkill(
        id="werewolf-game-orchestration",
        name="Werewolf Game Orchestration",
        description="Orchestrates and evaluates multi-agent Werewolf games",
        tags=["gaming", "evaluation", "social-deduction"],
    )

    # Build agent card
    if args.card_url:
        card_url = args.card_url
    else:
        card_url = f"http://{args.host}:{args.port}"

    card = AgentCard(
        name="Werewolf Game Orchestrator",
        version="1.0.0",
        description="Green agent for evaluating LLM agents in Werewolf social deduction game",
        url=card_url,
        protocol_version="0.3.0",
        skills=[skill],
        capabilities=AgentCapabilities(streaming=False),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )

    # Create A2A application
    task_store = InMemoryTaskStore()
    executor = WerewolfExecutor()
    request_handler = DefaultRequestHandler(agent_executor=executor, task_store=task_store)
    a2a_app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )
    
    # Build Starlette app
    starlette_app = a2a_app.build()

    # Start server
    print(f"🐺 Starting Werewolf Green Agent on {args.host}:{args.port}")
    print(f"📋 Agent Card: {card_url}/.well-known/agent-card.json")
    print(f"🔧 Protocol Version: {card.protocol_version}")
    print("Ready to orchestrate Werewolf games!")
    
    uvicorn.run(
        starlette_app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
