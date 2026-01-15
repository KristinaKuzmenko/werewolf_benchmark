"""
A2A Server for Baseline Werewolf Agent.
Main entry point following A2A template structure.
"""
import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from .executor import BaselineExecutor


def create_app(agent_id: str, host: str, port: int, public_url: str = None):
    """
    Create A2A Starlette application.
    
    Args:
        agent_id: Unique identifier for this agent
        host: Host to bind the server
        port: Port to bind the server
        public_url: Public URL for the agent card (optional, defaults to http://host:port)
        
    Returns:
        Starlette application instance
    """
    skill = AgentSkill(
        id="werewolf-player",
        name="Werewolf Player",
        description="Rule-based baseline agent for Werewolf game",
        tags=["gaming", "social-deduction", "baseline"],
    )
    
    card_url = public_url or f"http://{host}:{port}"
    card = AgentCard(
        name=agent_id,
        version="1.0.0",
        description="Rule-based baseline agent for Werewolf game",
        url=card_url,
        protocol_version="0.3.0",
        skills=[skill],
        capabilities=AgentCapabilities(streaming=False),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )
    
    task_store = InMemoryTaskStore()
    executor = BaselineExecutor(agent_id=agent_id)
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store
    )
    
    a2a_app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )
    
    return a2a_app.build()


def main():
    """Main entry point for Baseline Werewolf Agent server."""
    parser = argparse.ArgumentParser(description="Baseline Werewolf Purple Agent (A2A)")
    parser.add_argument("--agent-id", type=str, default="baseline-agent", help="Agent ID")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8100, help="Port to bind")
    parser.add_argument("--public-url", type=str, default=None, help="Public URL for agent card")
    args = parser.parse_args()
    
    app = create_app(args.agent_id, args.host, args.port, args.public_url)
    
    card_url = args.public_url or f"http://{args.host}:{args.port}"
    print(f"🎮 Starting Baseline Agent '{args.agent_id}' on {args.host}:{args.port}")
    print(f"📋 Agent Card: {card_url}/.well-known/agent-card.json")
    print(f"🔧 Protocol Version: 0.3.0")
    print("Ready to play Werewolf!")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
