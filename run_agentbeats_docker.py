"""
Run AgentBeats evaluation in Docker environment.
This script runs the green agent (evaluator) that orchestrates a game
with one external purple agent and NCP bots.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict

from src.green_agent.agent import WerewolfAgent, EvalRequest
from a2a.types import Message, Part, TextPart
from a2a.server.tasks import TaskUpdater
from a2a.server.events import EventQueue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class MockTaskUpdater(TaskUpdater):
    """Mock task updater for Docker testing."""
    
    def __init__(self):
        # Create mock EventQueue
        event_queue = EventQueue()
        super().__init__(
            event_queue=event_queue,
            task_id="agentbeats-docker-test",
            context_id="context-001"
        )
        self.results = None
        self.artifacts = []
    
    async def add_artifact(self, parts):
        """Capture artifacts."""
        self.artifacts.extend(parts)
        # Extract results from DataPart
        for part in parts:
            if hasattr(part, 'root') and hasattr(part.root, 'data'):
                self.results = part.root.data
    
    async def send_update(self, content: Dict[str, Any]) -> None:
        """Print updates to console."""
        print(f"[UPDATE] {json.dumps(content, indent=2)}")


async def main():
    """Run AgentBeats evaluation with Docker-hosted purple agent."""
    
    # Purple agent URL (accessible via Docker network)
    purple_agent_url = os.getenv("PURPLE_AGENT_URL", "http://purple_agent:8100")
    
    # Number of games from environment (set by AgentBeats platform via scenario.toml)
    num_games = int(os.getenv("NUM_TASKS", "5"))  # Default to 5 if not specified
    
    # Create evaluation request
    eval_request = EvalRequest(
        participant=purple_agent_url,  # Single agent (AgentBeats mode)
        config={
            "num_players": 8,
            "scenario_name": "werewolf_8p",  # Label only, not loaded from scenarios/
            "mode": "agentbeats",
            "num_games": num_games,  # Read from NUM_TASKS environment variable
            "max_concurrent_games": 1
        }
    )
    
    # Create A2A Message
    message = Message(
        messageId="agentbeats-docker-001",
        role="user",
        parts=[Part(root=TextPart(kind="text", text=json.dumps(eval_request.model_dump(mode="json"))))]
    )
    
    # Create mock task updater
    task_updater = MockTaskUpdater()
    
    print("=" * 80)
    print("Starting AgentBeats Evaluation in Docker")
    print("=" * 80)
    print(f"Purple Agent URL: {purple_agent_url}")
    print(f"Scenario: {eval_request.config.get('scenario_name')}")
    print(f"Mode: {eval_request.config.get('mode')}")
    print(f"Number of games: {eval_request.config.get('num_games')}")
    print("=" * 80)
    
    # Wait for purple agent to be fully ready
    print("\nWaiting for purple agent to be fully ready...")
    await asyncio.sleep(3)
    
    # Verify purple agent is accessible
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{purple_agent_url}/.well-known/agent-card.json")
            if response.status_code == 200:
                print(f"✓ Purple agent is accessible (status: {response.status_code})")
            else:
                print(f"⚠ Purple agent returned status: {response.status_code}")
    except Exception as e:
        print(f"✗ Cannot reach purple agent: {e}")
        return
    
    # Run the agent
    agent = WerewolfAgent()
    await agent.run(message=message, updater=task_updater)
    
    # Get results from task updater
    if task_updater.results:
        results_data = task_updater.results
        
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(json.dumps(results_data, indent=2))
        print("=" * 80)
        
        # Save results to file
        output_file = "/app/results/agentbeats_docker_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo results returned from evaluation")


if __name__ == "__main__":
    asyncio.run(main())
