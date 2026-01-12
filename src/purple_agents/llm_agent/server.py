"""
A2A Server for LLM Werewolf Agent.
Main entry point following A2A template structure.
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

from .executor import LLMExecutor


def create_app(agent_id: str, host: str, port: int, provider: str, model: str, public_url: str = None):
    """
    Create A2A Starlette application.
    
    Args:
        agent_id: Unique identifier for this agent
        host: Host to bind the server
        port: Port to bind the server
        provider: LLM provider (openai, anthropic, groq)
        model: Model name
        public_url: Public URL for agent card (overrides host:port, useful for Docker)
        
    Returns:
        Starlette application instance
    """
    # Use public_url if provided, otherwise construct from host:port
    agent_url = public_url if public_url else f"http://{host}:{port}"
    
    skill = AgentSkill(
        id="werewolf-player",
        name="Werewolf Player",
        description=f"LLM-powered agent with reasoning & bidding ({provider}/{model})",
        tags=["gaming", "social-deduction", "llm"],
    )
    
    card = AgentCard(
        name=agent_id,
        version="2.0.0",
        description=f"LLM-powered Werewolf agent ({provider}/{model})",
        url=agent_url,
        protocol_version="0.3.0",
        skills=[skill],
        capabilities=AgentCapabilities(streaming=False),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )
    
    task_store = InMemoryTaskStore()
    executor = LLMExecutor(
        agent_id=agent_id,
        provider=provider,
        model=model
    )
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store
    )
    
    a2a_app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )
    
    app = a2a_app.build()
    
    # Add health check endpoint
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    
    async def health_check(request):
        return JSONResponse({"status": "healthy", "agent_id": agent_id})
    
    # Add health route
    app.routes.append(Route("/health", health_check, methods=["GET"]))
    
    return app


def main():
    """Main entry point for LLM Werewolf Agent server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress verbose HTTP logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    parser = argparse.ArgumentParser(description="LLM Werewolf Purple Agent (A2A)")
    parser.add_argument("--agent-id", type=str, default="llm-agent", help="Agent ID")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8100, help="Port to bind")
    parser.add_argument("--public-url", type=str, default=None, 
                        help="Public URL for agent card (e.g., http://purple_agent:8100 in Docker)")
    parser.add_argument("--provider", type=str, default="openai", 
                        choices=["openai", "anthropic", "groq"], help="LLM provider")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    args = parser.parse_args()
    
    app = create_app(args.agent_id, args.host, args.port, args.provider, args.model, args.public_url)
    
    print(f"🤖 Starting LLM Agent '{args.agent_id}' on {args.host}:{args.port}")
    print(f"📋 Agent Card: http://{args.host}:{args.port}/.well-known/agent-card.json")
    print(f"🧠 Provider: {args.provider}, Model: {args.model}")
    print(f"🔧 Protocol Version: 0.3.0")
    print("Ready to play Werewolf!")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
