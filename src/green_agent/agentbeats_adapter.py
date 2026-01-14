"""
AgentBeats Adapter for Werewolf Benchmark.

Enables testing a single external agent against NCP (Non-Conversational Player) bots.
Only the tested agent's metrics are returned to AgentBeats platform.

NCP Bot Composition (for 8-player game):
- 1 Tested Agent (external)
- 4 Basic NCP bots (SmartBaselineAgent) - rule-based
- 3 LLM NCP bots (LLMNPCAgent) - LLM-powered
"""
import json
import logging
import random
from typing import Any, Dict, Optional
from pydantic import BaseModel, HttpUrl

from .ncp_agent import SmartBaselineAgent
from .llm_npc_agent import LLMNPCAgent
from .models import ActionType

logger = logging.getLogger(__name__)


class NCPLocalHandler:
    """
    Local handler for NCP agents - avoids HTTP calls by running NCP logic directly.
    Each NCP agent instance maintains its own state.
    
    Supports both:
    - SmartBaselineAgent (rule-based)
    - LLMNPCAgent (LLM-powered)
    """
    
    def __init__(self, agent_id: str, npc_type: str = "baseline"):
        """
        Initialize NCP handler.
        
        Args:
            agent_id: Unique agent identifier
            npc_type: "baseline" for SmartBaselineAgent, "llm" for LLMNPCAgent
        """
        self.agent_id = agent_id
        self.npc_type = npc_type
        
        if npc_type == "llm":
            # Use OpenAI instead of Groq to avoid rate limiting
            self.agent = LLMNPCAgent(
                agent_id=agent_id,
                provider="openai",
                model="gpt-4o-mini"
            )
            logger.info(f"NCP LLM handler created: {agent_id} (OpenAI)")
        else:
            self.agent = SmartBaselineAgent(agent_id=agent_id)
            logger.info(f"NCP baseline handler created: {agent_id}")
    
    async def handle_message(self, message: Dict) -> Dict:
        """
        Process message and return NCP agent's response.
        Mimics HTTP endpoint behavior but runs locally.
        """
        message_type = message.get("type")
        
        try:
            if message_type == "game_start":
                return self.agent.handle_game_start(message)
            
            elif message_type == "night_action":
                response = self.agent.handle_night_action(message)
                if response and "action" in response and response['action']:
                    action = response['action']
                    logger.info(f"Player {self.agent_id} night action: {action.get('action_type')} → target={action.get('target_id')}")
                return response
            
            elif message_type == "speak":
                response = self.agent.handle_speak(message)
                if response and "speech" in response and response['speech']:
                    logger.info(f"Player {self.agent_id} speech: {response['speech']}")
                return response
            
            elif message_type == "vote":
                response = self.agent.handle_vote(message)
                if response and "vote" in response:
                    logger.info(f"Player {self.agent_id} vote: Player {response['vote']}")
                return response
            
            elif message_type == "vote_intention":
                return self.agent.handle_vote(message)
            
            elif message_type == "reaction":
                # Simple reaction - baseline agents can return a brief comment
                return {"reaction": "I see."}
            
            elif message_type == "bid_request":
                # Baseline agents return a variable bid (30-70 range)
                bid_value = random.randint(30, 70)
                priority = random.choice(["low", "medium", "high"])
                logger.debug(f"Player {self.agent_id} bid: {bid_value} - {priority}")
                return {"bid_value": bid_value, "priority": priority}
            
            elif message_type == "sheriff_election":
                return self.agent.handle_sheriff_election(message)
            
            elif message_type == "sheriff_summary":
                return self.agent.handle_sheriff_summary(message)
            
            elif message_type == "hunter_shoot":
                return self.agent.handle_hunter_shoot(message)
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                return {"status": "unknown_message_type"}
        
        except Exception as e:
            logger.error(f"NCP handler error for {self.agent_id}: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}


class AgentBeatsEvalRequest(BaseModel):
    """
    AgentBeats evaluation request format.
    
    Only one participant (the tested agent) is specified.
    The rest of the lobby is filled with NCP bots:
    - 4 baseline bots (SmartBaselineAgent)
    - 3 LLM bots (LLMNPCAgent)
    """
    participant: HttpUrl  # Single agent under test
    config: Dict[str, Any]  # Game configuration
    
    # Optional: specify which player slot the tested agent should take
    # If not specified, randomly assigned
    tested_player_id: Optional[int] = None
    
    # Optional: NCP bot configuration
    num_baseline_bots: int = 4  # Number of rule-based bots
    num_llm_bots: int = 3  # Number of LLM-powered bots


class AgentBeatsAdapter:
    """
    Adapter for AgentBeats platform integration.
    
    Key features:
    - Fills game with NCP bots (only 1 external agent tested)
    - Mixed NCP composition: 4 baseline + 3 LLM bots
    - Returns metrics only for the tested agent
    - Handles local NCP agent execution (no HTTP overhead)
    """
    
    def __init__(self):
        self.ncp_handlers: Dict[int, NCPLocalHandler] = {}
        self.tested_player_id: Optional[int] = None
        self.tested_agent_endpoint: Optional[str] = None
        self.ncp_bot_types: Dict[int, str] = {}  # player_id -> "baseline" or "llm"
    
    def setup_game_participants(
        self, 
        request: AgentBeatsEvalRequest,
        num_players: int
    ) -> Dict[int, str]:
        """
        Setup game participants: 1 tested agent + (num_players - 1) NCP bots.
        
        NCP bot composition:
        - request.num_baseline_bots of SmartBaselineAgent
        - request.num_llm_bots of LLMNPCAgent
        - Total must equal (num_players - 1)
        
        Args:
            request: AgentBeats evaluation request
            num_players: Total number of players in game
        
        Returns:
            player_endpoints: Mapping of player_id -> endpoint
        """
        import random
        
        num_npc_needed = num_players - 1
        num_baseline = request.num_baseline_bots
        num_llm = request.num_llm_bots
        
        # Validate bot counts
        if num_baseline + num_llm != num_npc_needed:
            logger.warning(
                f"NCP bot count mismatch: need {num_npc_needed}, "
                f"got {num_baseline} baseline + {num_llm} LLM"
            )
            # Auto-adjust: prioritize baseline bots
            if num_baseline + num_llm < num_npc_needed:
                num_baseline = num_npc_needed - num_llm
            else:
                num_llm = num_npc_needed - num_baseline
        
        player_endpoints = {}
        
        # Determine tested player slot
        if request.tested_player_id and 1 <= request.tested_player_id <= num_players:
            self.tested_player_id = request.tested_player_id
        else:
            # Random slot for tested agent
            self.tested_player_id = random.randint(1, num_players)
        
        self.tested_agent_endpoint = str(request.participant)
        
        logger.info(f"🎯 Tested agent assigned to Player {self.tested_player_id}")
        logger.info(f"   Endpoint: {self.tested_agent_endpoint}")
        logger.info(f"🤖 NCP bot composition: {num_baseline} baseline + {num_llm} LLM")
        
        # Create list of bot types to assign
        bot_types = ["baseline"] * num_baseline + ["llm"] * num_llm
        random.shuffle(bot_types)
        
        # Assign NCP handlers to other slots
        bot_index = 0
        for player_id in range(1, num_players + 1):
            if player_id == self.tested_player_id:
                # Real external agent
                player_endpoints[player_id] = self.tested_agent_endpoint
            else:
                # NCP bot
                bot_type = bot_types[bot_index]
                bot_index += 1
                
                ncp_id = f"ncp-{bot_type}-p{player_id}"
                player_endpoints[player_id] = f"ncp://{ncp_id}"
                
                # Create local handler
                self.ncp_handlers[player_id] = NCPLocalHandler(ncp_id, npc_type=bot_type)
                self.ncp_bot_types[player_id] = bot_type
                
                emoji = "🧠" if bot_type == "llm" else "🤖"
                logger.info(f"{emoji} NCP {bot_type} bot created for Player {player_id}")
        
        return player_endpoints
    
    def is_ncp_endpoint(self, endpoint: str) -> bool:
        """Check if endpoint is a local NCP bot."""
        return endpoint.startswith("ncp://")
    
    async def send_to_player(
        self,
        player_id: int,
        endpoint: str,
        message: Dict,
        external_messenger
    ) -> Dict:
        """
        Send message to player - route to NCP handler or external agent.
        
        Args:
            player_id: Player ID
            endpoint: Player endpoint
            message: Message to send
            external_messenger: Messenger instance for external HTTP calls
        
        Returns:
            Response from player
        """
        if self.is_ncp_endpoint(endpoint):
            # Local NCP bot
            if player_id not in self.ncp_handlers:
                logger.error(f"No NCP handler for player {player_id}")
                logger.error(f"Available handlers: {list(self.ncp_handlers.keys())}")
                logger.error(f"Tested player ID: {self.tested_player_id}")
                return {"status": "error", "error": "no_handler"}
            return await self.ncp_handlers[player_id].handle_message(message)
        else:
            # External agent - use HTTP
            message_json = json.dumps(message)
            response = await external_messenger.talk_to_agent(
                message=message_json,
                url=endpoint
            )
            return json.loads(response) if isinstance(response, str) else response
    
    def filter_results_for_tested_agent(
        self,
        full_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter game results to return only tested agent's metrics.
        
        This is what AgentBeats platform receives for evaluation.
        
        Args:
            full_results: Complete game results with all players
        
        Returns:
            Filtered results containing only tested agent's performance
        """
        if self.tested_player_id is None:
            logger.warning("No tested player ID set")
            return full_results
        
        tested_id_str = str(self.tested_player_id)
        
        # Extract tested agent's data
        tested_player_data = full_results.get("players", {}).get(tested_id_str, {})
        tested_metrics = tested_player_data.get("metrics", {})
        
        # Extract tested agent's role-specific metrics if available
        tested_role_metrics = full_results.get("role_metrics", {}).get(tested_id_str, {})
        
        # Get game outcome info
        winner = full_results.get("winner")
        tested_camp = tested_player_data.get("camp")
        tested_role = tested_player_data.get("role")
        tested_survived = tested_player_data.get("is_alive", False)
        
        # Determine if tested agent won
        won_game = (winner == tested_camp) if winner and tested_camp else False
        
        # Filter advanced metrics based on role
        advanced_metrics = full_results.get("advanced_metrics", {})
        filtered_advanced_metrics = {}
        
        # Only include manipulation metrics for werewolves
        if tested_role == "werewolf":
            filtered_advanced_metrics["manipulation_success_d1"] = advanced_metrics.get("manipulation_success_d1", 0.0)
            filtered_advanced_metrics["manipulation_success_d2"] = advanced_metrics.get("manipulation_success_d2", 0.0)
        
        # Only include auto_sabotage for good camp (when they accidentally help wolves)
        if tested_camp == "good":
            filtered_advanced_metrics["auto_sabotage"] = advanced_metrics.get("auto_sabotage", 0.0)
        
        # Include general game metrics (not role-specific)
        filtered_advanced_metrics["day1_wolf_eliminated"] = advanced_metrics.get("day1_wolf_eliminated", 0.0)
        filtered_advanced_metrics["vote_swings"] = advanced_metrics.get("vote_swings", [])
        
        filtered_results = {
            "status": "complete",
            "game_id": full_results.get("game_id"),
            "rounds": full_results.get("rounds"),
            
            # Tested agent info
            "tested_agent": {
                "player_id": self.tested_player_id,
                "agent_id": tested_player_data.get("agent_id"),
                "role": tested_role,
                "camp": tested_camp,
                "survived": tested_survived,
                "won_game": won_game
            },
            
            # Performance metrics (only for tested agent)
            "performance_metrics": tested_metrics,
            
            # Role-specific metrics
            "role_metrics": tested_role_metrics,
            
            # Game context (minimal info about game outcome)
            "game_outcome": {
                "winner": winner,
                "total_rounds": full_results.get("rounds"),
                "tested_agent_won": won_game
            },
            
            # Advanced metrics (filtered by role)
            "advanced_metrics": filtered_advanced_metrics
        }
        
        logger.info(f"✂️ Filtered results for Player {self.tested_player_id}")
        logger.info(f"   Role: {tested_role}, Camp: {tested_camp}")
        logger.info(f"   Won: {won_game}, Survived: {tested_survived}")
        logger.info(f"   Metrics: IRS={tested_metrics.get('irs')}, "
                   f"VRS={tested_metrics.get('vrs')}, "
                   f"MSS={tested_metrics.get('mss')}")
        
        return filtered_results
    
    def get_tested_player_id(self) -> Optional[int]:
        """Get the player ID of the tested agent."""
        return self.tested_player_id
    
    def cleanup(self):
        """Clean up NCP handlers."""
        # For LLM agents with httpx clients, we skip async cleanup to avoid
        # "Event loop is closed" errors. The httpx clients will be garbage
        # collected and their connections will be closed by the OS.
        # This is safe because we're shutting down the entire game session.
        
        logger.info(f"🧹 Cleaning up {len(self.ncp_handlers)} NCP handlers")
        
        for player_id, handler in self.ncp_handlers.items():
            # Reset baseline agents (synchronous cleanup)
            if hasattr(handler.agent, 'reset'):
                handler.agent.reset()
        
        self.ncp_handlers.clear()
        logger.info("🧹 NCP handlers cleaned up")
