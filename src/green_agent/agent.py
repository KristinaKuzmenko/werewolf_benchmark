"""
Werewolf Green Agent - Game orchestration logic.
Extracted from werewolf_judge.py to follow A2A template structure.
Full game logic integrated.
"""
import json
import logging
import math
import random
from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict
from datetime import datetime

from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from .messenger import Messenger
from .game_engine import WerewolfGameEngine
from .metrics import MetricsCalculator, AdvancedMetricsCalculator
from .models import GameState, Action, RoleType, Camp, ActionType, Phase
from .agentbeats_adapter import AgentBeatsAdapter, AgentBeatsEvalRequest

logger = logging.getLogger(__name__)
console = Console()


class EvalRequest(BaseModel):
    """
    Request format sent by the AgentBeats platform to green agents.
    Follows A2A standard structure.
    
    Supports two modes:
    1. Multi-agent tournament: participants dict with multiple agents
    2. AgentBeats single-eval: participant (singular) with one agent + NCP bots
    """
    participants: dict[str, HttpUrl] | dict[str, dict[str, Any]] | None = None  # Multi-agent tournament mode (can include agentbeats_id)
    participant: HttpUrl | None = None  # AgentBeats single-eval mode
    config: dict[str, Any]  # Game configuration
    tested_player_id: int | None = None  # Optional: specify player slot for tested agent
    agentbeats_id: str | None = None  # AgentBeats platform ID (top-level or in participants)


class WerewolfAgent:
    """
    Werewolf game orchestrator agent.
    Implements game logic following A2A patterns.
    
    Supports two evaluation modes:
    1. Tournament mode: Multiple external agents compete
    2. AgentBeats mode: Single agent tested against NCP bots
    """
    
    # Required roles for a valid game request
    required_roles: list[str] = []  # Empty = any participants accepted
    
    # Required config keys
    required_config_keys: list[str] = ["num_players"]
    
    def __init__(self):
        self.messenger = Messenger()
        self.active_games: dict[str, GameState] = {}
        self.player_actions: dict[str, dict[int, list[Action]]] = {}
        self.task_status: dict[str, str] = {}
        self.game_results: dict[str, Any] = {}
        
        # AgentBeats integration
        self.agentbeats_adapter: dict[str, AgentBeatsAdapter] = {}  # task_id -> adapter
        self.is_agentbeats_mode: dict[str, bool] = {}  # task_id -> bool
    
    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """
        Validate incoming assessment request.
        
        Supports:
        - Tournament mode: participants (plural) provided
        - AgentBeats mode: participant (singular) provided
        
        Returns:
            (is_valid, error_message)
        """
        # Check config keys
        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        
        num_players = request.config.get("num_players", 8)
        
        # Validate num_players is valid (8 or 12)
        if num_players not in [8, 12]:
            return False, f"Invalid num_players: {num_players}. Must be 8 or 12"
        
        # Check mode: tournament vs AgentBeats
        if request.participants is not None:
            # Check if this is AgentBeats mode (1 participant) or tournament mode (all participants)
            if len(request.participants) == 1:
                # AgentBeats mode: 1 agent + NPC bots
                logger.info(f"🎯 AgentBeats mode: 1 tested agent + {num_players - 1} NPC bots")
            elif len(request.participants) == num_players:
                # Tournament mode: all agents provided
                logger.info(f"🏆 Tournament mode: {num_players} agents")
            else:
                return False, f"Expected {num_players} players or 1 (AgentBeats mode), got {len(request.participants)}"
        elif request.participant is not None:
            # AgentBeats mode: single agent + NCP bots (alternative format)
            logger.info(f"🎯 AgentBeats mode: 1 tested agent + {num_players - 1} NPC bots")
        else:
            return False, "Must provide either 'participants' (tournament) or 'participant' (AgentBeats)"
        
        return True, "ok"
    
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Main agent logic - orchestrate a Werewolf game.
        
        Args:
            message: The incoming A2A message with game request
            updater: TaskUpdater for reporting progress and results
        """
        input_text = get_message_text(message)
        
        # Parse and validate request
        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return
        except json.JSONDecodeError as e:
            await updater.reject(new_agent_text_message(f"Invalid JSON: {e}"))
            return
        
        # Extract configuration
        config = request.config
        num_players = config.get("num_players", 8)
        enable_sheriff = config.get("enable_sheriff", True)
        num_games = config.get("num_tasks", config.get("num_games", 1))  # Support both num_tasks and num_games
        
        logger.info(f"📋 Evaluation config: {num_games} games, {num_players} players")
        
        # Determine mode: AgentBeats if single participant (or participant field), Tournament if all participants
        is_agentbeats = (
            request.participant is not None or 
            (request.participants is not None and len(request.participants) == 1)
        )
        
        # Store request info for creating adapters per game
        # For AgentBeats mode with participants (list with 1 element), extract the URL and agentbeats_id
        participant_url = None
        participant_agentbeats_id = None
        participant_name = "baseline-agent"  # Default name
        
        # Try to get agentbeats_id from multiple locations
        if request.agentbeats_id:
            # Top-level agentbeats_id (preferred)
            participant_agentbeats_id = request.agentbeats_id
        
        if request.participant is not None:
            participant_url = request.participant
        elif request.participants is not None and len(request.participants) == 1:
            # Get the first (and only) participant - can be URL or dict with endpoint/agentbeats_id
            participant_name = list(request.participants.keys())[0]  # Get the name/key
            participant_data = list(request.participants.values())[0]
            if isinstance(participant_data, dict):
                participant_url = participant_data.get("endpoint")
                # Override with nested agentbeats_id if present
                if not participant_agentbeats_id and participant_data.get("agentbeats_id"):
                    participant_agentbeats_id = participant_data.get("agentbeats_id")
            else:
                participant_url = participant_data
                
        tested_player_id_config = request.tested_player_id
        
        if is_agentbeats:
            logger.info(f"🤖 AgentBeats mode enabled")
            logger.info(f"   1 tested agent + {num_players - 1} NCP bots")
            logger.info(f"   Participant name: {participant_name}")
        else:
            logger.info(f"🏆 Tournament mode enabled")
            logger.info(f"   {len(request.participants)} competing agents")
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting {num_games} game(s) with {num_players} players...")
        )
        
        # Run multiple games and aggregate results
        all_game_results = []
        
        for game_num in range(1, num_games + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"🎮 GAME {game_num}/{num_games}")
            logger.info(f"{'='*80}")
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Running game {game_num}/{num_games}...")
            )
            
            # Create fresh adapter for each game (in AgentBeats mode)
            adapter = None
            if is_agentbeats:
                adapter = AgentBeatsAdapter()
                agentbeats_request = AgentBeatsEvalRequest(
                    participant=participant_url,
                    config=config,
                    tested_player_id=tested_player_id_config
                )
                player_endpoints = adapter.setup_game_participants(
                    agentbeats_request, 
                    num_players
                )
                # Name mapping for AgentBeats
                player_agent_names = {
                    adapter.tested_player_id: "tested_agent"
                }
                for pid in range(1, num_players + 1):
                    if pid != adapter.tested_player_id:
                        player_agent_names[pid] = f"ncp_bot_{pid}"
                
                logger.info(f"   Tested agent: Player {adapter.tested_player_id}")
            else:
                # Tournament mode: all external agents
                player_endpoints = {}
                player_agent_names = {}
                for i, (player_name, endpoint) in enumerate(request.participants.items(), 1):
                    player_endpoints[i] = str(endpoint)
                    player_agent_names[i] = player_name
            
            # Initialize game engine for this game
            try:
                engine = WerewolfGameEngine(
                    num_players=num_players,
                    enable_sheriff=enable_sheriff,
                    config=config
                )
                
                game_state = engine.initialize_game(player_endpoints)
                game_state.player_agent_names = player_agent_names
                
                # Use game_id as task tracking identifier
                task_id = game_state.game_id
                self.active_games[task_id] = game_state
                self.player_actions[task_id] = {i: [] for i in range(1, num_players + 1)}
                self.task_status[task_id] = "running"
                
                # Store AgentBeats adapter if in that mode
                if adapter:
                    self.agentbeats_adapter[task_id] = adapter
                    self.is_agentbeats_mode[task_id] = True
                else:
                    self.is_agentbeats_mode[task_id] = False
                
                logger.info(f"🎮 Game initialized: {game_state.game_id}")
                logger.info(f"  Players: {num_players}")
                logger.info(f"  Sheriff: {enable_sheriff}")
                logger.info(f"  Mode: {'AgentBeats' if adapter else 'Tournament'}")
                
                # Log tested agent role in AgentBeats mode
                if adapter:
                    tested_role = game_state.players[adapter.tested_player_id].role
                    logger.info(f"  🎭 Tested agent (Player {adapter.tested_player_id}) role: {tested_role.value}")
                
            except Exception as e:
                logger.error(f"Failed to initialize game {game_num}: {e}")
                continue
            
            # Run full game orchestration
            try:
                await self._orchestrate_game(task_id, engine, config, updater)
                
                # Store results for this game
                if task_id in self.game_results:
                    game_result = self.game_results[task_id].copy()
                    game_result["game_number"] = game_num
                    
                    # Verify tested_agent was added
                    if is_agentbeats and "tested_agent" not in game_result:
                        logger.error(f"❌ Game {game_num} (task {task_id}) missing tested_agent field!")
                    
                    all_game_results.append(game_result)
                    logger.info(f"✓ Game {game_num} completed")
                
                # Cleanup this game (do NOT delete game_results yet - needed for aggregation)
                if task_id in self.agentbeats_adapter:
                    self.agentbeats_adapter[task_id].cleanup()
                    del self.agentbeats_adapter[task_id]
                if task_id in self.active_games:
                    del self.active_games[task_id]
                if task_id in self.player_actions:
                    del self.player_actions[task_id]
                if task_id in self.task_status:
                    del self.task_status[task_id]
                if task_id in self.game_results:
                    del self.game_results[task_id]
                    
            except Exception as e:
                logger.error(f"Error in game {game_num} orchestration: {e}", exc_info=True)
                # Cleanup on error
                if task_id in self.agentbeats_adapter:
                    self.agentbeats_adapter[task_id].cleanup()
                    del self.agentbeats_adapter[task_id]
                continue
        
        # Aggregate results across all games
        if not all_game_results:
            await updater.failed(new_agent_text_message(f"All {num_games} games failed"))
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 AGGREGATING RESULTS FROM {len(all_game_results)} GAMES")
        logger.info(f"{'='*80}")
        
        # Debug: Check structure of all_game_results
        if all_game_results:
            first_result = all_game_results[0]
            logger.info(f"DEBUG: First game result keys: {list(first_result.keys())}")
            if "results" in first_result:
                logger.warning("⚠️ game_result already contains 'results' - possible double aggregation!")
        
        try:
            aggregated_results = self._aggregate_game_results(
                all_game_results, 
                is_agentbeats, 
                participant_agentbeats_id=participant_agentbeats_id,
                participant_name=participant_name
            )
            
            # AgentBeats aggregation returns dict with participants and results
            if is_agentbeats:
                logger.info("✂️ Results filtered for tested agent only")
            
            result_json = json.dumps(aggregated_results, indent=2)
            await updater.add_artifact(
                [Part(root=DataPart(kind="data", data=aggregated_results))]
            )
            
            logger.info(f"✅ Evaluation complete: {len(all_game_results)}/{num_games} games successful")
            
        except Exception as e:
            logger.error(f"Error aggregating results: {e}", exc_info=True)
            await updater.failed(new_agent_text_message(f"Failed to aggregate results: {e}"))
    
    async def _orchestrate_game(
        self,
        task_id: str,
        engine: WerewolfGameEngine,
        config: Dict[str, Any],
        updater: TaskUpdater
    ):
        """
        Main orchestration loop for the game.
        Manages rounds, phases, and player interactions.
        """
        game_state = self.active_games[task_id]
        
        try:
            # Send initial game state to all players
            await self._broadcast_game_start(task_id, game_state, updater)
            
            # Sheriff election on first day (if enabled)
            if engine.enable_sheriff:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message("Sheriff election begins")
                )
                await self._conduct_sheriff_election(task_id, engine, game_state, updater)
            
            # Main game loop
            while not game_state.winner:
                game_state.round_number += 1
                logger.info(f"=== Round {game_state.round_number} ===")
                
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Round {game_state.round_number} starting")
                )
                
                # Night phase
                game_state.phase = Phase.NIGHT
                await self._run_night_phase(task_id, engine, game_state, updater)
                
                # Check for winner after night
                game_state.winner = engine.check_victory_condition(game_state)
                if game_state.winner:
                    break
                
                # Day phase
                game_state.phase = Phase.DAY
                await self._run_day_phase(task_id, engine, game_state, updater)
                
                # Check for winner after day
                game_state.winner = engine.check_victory_condition(game_state)
                if game_state.winner:
                    break
                
                # Safety: max 20 rounds
                if game_state.round_number >= 20:
                    logger.warning("Max rounds reached, ending game")
                    # Determine winner by remaining players
                    good_alive = sum(1 for p in game_state.players.values()
                                   if p.is_alive and p.camp == Camp.GOOD)
                    wolf_alive = sum(1 for p in game_state.players.values()
                                   if p.is_alive and p.camp == Camp.WOLF)
                    game_state.winner = Camp.GOOD if good_alive > wolf_alive else Camp.WOLF
                    break
            
            # Finalize game and calculate metrics
            await self._finalize_game(task_id, engine, game_state, updater)
            
        except Exception as e:
            logger.error(f"Error orchestrating game {task_id}: {e}", exc_info=True)
            raise
    
    async def _broadcast_game_start(
        self,
        task_id: str,
        game_state: GameState,
        updater: TaskUpdater
    ):
        """Notify all players that the game is starting and send role info"""
        logger.info(f"📢 Broadcasting game start to {len(game_state.players)} players")
        
        # Pretty print game start
        table = Table(title="🐺 Werewolf Game Starting", box=box.ROUNDED)
        table.add_column("Player", style="cyan", justify="center")
        table.add_column("Role", style="yellow")
        table.add_column("Camp", style="green")
        
        for player_id, player in game_state.players.items():
            role_emoji = {
                RoleType.WEREWOLF: "🐺",
                RoleType.SEER: "🔮",
                RoleType.WITCH: "🧪",
                RoleType.HUNTER: "🎯",
                RoleType.GUARD: "🛡️",
                RoleType.VILLAGER: "👤"
            }.get(player.role, "❓")
            
            camp_color = "red" if player.camp == Camp.WOLF else "green"
            
            table.add_row(
                f"Player {player_id}",
                f"{role_emoji} {player.role.value}",
                f"[{camp_color}]{player.camp.value}[/{camp_color}]"
            )
        
        console.print(table)
        console.print(f"[bold blue]Game ID:[/bold blue] {game_state.game_id}")
        console.print(f"[bold blue]Total Players:[/bold blue] {len(game_state.players)}")
        console.print()
        
        for player_id, player in game_state.players.items():
            observation = {
                "type": "game_start",
                "game_id": game_state.game_id,
                "player_id": player_id,
                "role": player.role.value,
                "camp": player.camp.value,
                "total_players": len(game_state.players),
                "alive_players": game_state.alive_players.copy()
            }
            
            # Werewolves know each other
            if player.role == RoleType.WEREWOLF:
                observation["werewolf_team"] = [
                    pid for pid in game_state.alive_players
                    if game_state.players[pid].role == RoleType.WEREWOLF
                ]
            
            logger.info(f"  → Notifying Player {player_id} ({player.role.value}) at {player.endpoint}")
            await self._send_to_player(task_id, player_id, player.endpoint, observation, updater)
    
    async def _send_to_player(
        self,
        task_id: str,
        player_id: int,
        endpoint: str,
        message: Dict,
        updater: TaskUpdater
    ) -> Dict:
        """
        Send message to a player agent using messenger or local NCP handler.
        
        Args:
            task_id: Game task ID (to lookup AgentBeats adapter)
            player_id: Player ID (for NCP routing)
            endpoint: The agent's URL endpoint
            message: The message dict to send (will be JSON-serialized)
            updater: TaskUpdater for status updates
            
        Returns:
            Parsed response dict from the agent
        """
        msg_type = message.get('type', 'unknown')
        logger.debug(f"📤 Sending {msg_type} to Player {player_id} at {endpoint}")
        
        # Check if AgentBeats mode with NCP bots
        if self.is_agentbeats_mode.get(task_id, False):
            adapter = self.agentbeats_adapter.get(task_id)
            if adapter:
                # Use adapter for routing (handles NCP vs external)
                return await adapter.send_to_player(
                    player_id,
                    endpoint,
                    message,
                    self.messenger
                )
        
        # Standard tournament mode: all external agents
        try:
            # Convert message dict to JSON string for A2A protocol
            message_json = json.dumps(message)
            
            # Call messenger with correct arguments: (message: str, url: str)
            response_str = await self.messenger.talk_to_agent(
                message=message_json,
                url=endpoint
            )
            
            # Parse response string back to dict
            if response_str is None or response_str == "":
                logger.warning(f"📥 Empty response from {endpoint}")
                return {}
            
            try:
                result = json.loads(response_str)
            except json.JSONDecodeError:
                # Response might be plain text, wrap it
                logger.debug(f"📥 Non-JSON response from {endpoint}: {response_str[:100]}...")
                return {"text": response_str}
            
            if not isinstance(result, dict):
                logger.warning(f"📥 Invalid response type from {endpoint}: {type(result)}")
                return {}
            
            if 'action' in result and result.get('action'):
                action = result.get('action') or {}
                action_type = action.get('action_type', 'unknown') if isinstance(action, dict) else 'unknown'
                logger.debug(f"📥 Response from {endpoint}: {action_type}")
            else:
                logger.debug(f"📥 Response from {endpoint}: OK")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Error sending to {endpoint}: {e}", exc_info=True)
            return {}
    
    async def _conduct_sheriff_election(
        self,
        task_id: str,
        engine: WerewolfGameEngine,
        game_state: GameState,
        updater: TaskUpdater
    ):
        """Conduct sheriff election"""
        # Request votes from all players
        votes = {}
        for player_id in game_state.alive_players:
            player = game_state.players[player_id]
            
            vote_request = {
                "type": "sheriff_election",
                "candidates": game_state.alive_players.copy()
            }
            
            response = await self._send_to_player(task_id, player_id, player.endpoint, vote_request, updater)
            if response and "vote" in response:
                votes[player_id] = response["vote"]
        
        # Process election
        game_state = engine.elect_sheriff(game_state, votes)
        
        if game_state.sheriff_id:
            logger.info(f"Player {game_state.sheriff_id} elected as Sheriff")
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Player {game_state.sheriff_id} elected as Sheriff")
            )
    
    async def _run_night_phase(
        self,
        task_id: str,
        engine: WerewolfGameEngine,
        game_state: GameState,
        updater: TaskUpdater
    ):
        """Execute night phase - collect and process night actions"""
        night_actions = {}
        
        logger.info(f"🌙 Night phase: requesting actions from {len(game_state.alive_players)} alive players")
        
        # Collect actions from roles that act at night
        for player_id in game_state.alive_players:
            player = game_state.players[player_id]
            
            # Check if this role acts at night
            if player.role in [RoleType.WEREWOLF, RoleType.SEER, RoleType.GUARD, RoleType.WITCH]:
                logger.info(f"  → Requesting night action from Player {player_id} ({player.role.value})")
                
                observation = engine.create_observation_for_player(
                    player_id, game_state,
                    phase_specific_info={'werewolf_victim': None}
                )
                
                action_request = {
                    "type": "night_action",
                    "observation": observation.model_dump()
                }
                
                try:
                    response = await self._send_to_player(task_id, player_id, player.endpoint, action_request, updater)
                    
                    if response and "action" in response and response["action"] is not None:
                        action = Action(**response["action"])
                        night_actions[player_id] = action
                        logger.info(f"  ✓ Player {player_id} action: {action.action_type.value} → target={action.target_id}")
                        
                        # Track actions for evaluation
                        self.player_actions[task_id][player_id].append(action)
                    else:
                        logger.warning(f"  ✗ Player {player_id} returned no action: {response}")
                        
                except Exception as e:
                    logger.error(f"  ✗ Error getting action from Player {player_id}: {e}", exc_info=True)
            else:
                logger.debug(f"  ⊘ Player {player_id} ({player.role.value}) does not act at night")
        
        # Process night actions
        game_state, eliminated = engine.process_night_phase(game_state, night_actions)
        
        # Announce results
        if eliminated:
            console.print(f"[bold red]💀 Night {game_state.round_number}: Players {eliminated} were eliminated[/bold red]")
            result_msg = f"Night {game_state.round_number}: Players {eliminated} were eliminated"
        else:
            console.print(f"[bold green]✅ Night {game_state.round_number}: No one was eliminated[/bold green]")
            result_msg = f"Night {game_state.round_number}: No one was eliminated"
        
        await updater.update_status(TaskState.working, new_agent_text_message(result_msg))
        
        # Check for hunter death and handle shooting
        for eliminated_id in eliminated:
            if game_state.players[eliminated_id].role == RoleType.HUNTER:
                await self._handle_hunter_shot(task_id, engine, game_state, eliminated_id, updater)
    
    async def _run_day_phase(
        self,
        task_id: str,
        engine: WerewolfGameEngine,
        game_state: GameState,
        updater: TaskUpdater
    ):
        """Execute day phase with richer discussion, reactions, and voting"""
        
        # === Adaptive discussion controls ===
        MIN_DISCUSSION_ROUNDS = 2
        MAX_DISCUSSION_ROUNDS = 3
        ENABLE_BIDDING = True
        ENABLE_REACTIONS = True
        REACTIONS_PER_ROUND = 1

        discussion_rounds = min(
            MAX_DISCUSSION_ROUNDS,
            max(MIN_DISCUSSION_ROUNDS, math.ceil(len(game_state.alive_players) / 3))
        )

        latest_speeches: Dict[int, str] = {}
        all_speech_context: Dict[int, str] = {}
        speech_transcript: List[Dict[str, Any]] = []
        accusations: Dict[int, List[Dict[str, Any]]] = {}
        last_round_order: Dict[int, int] = {}

        logger.info(f"☀️ Day phase: {discussion_rounds} discussion rounds")

        # === Phase 1: Multi-round discussion with bidding ===
        for disc_round in range(1, discussion_rounds + 1):
            console.print(f"[cyan]💬 Discussion Round {disc_round}/{discussion_rounds}[/cyan]")

            if ENABLE_BIDDING:
                speaker_order = await self._get_bidding_order(
                    task_id,
                    game_state,
                    disc_round,
                    last_round_order,
                    updater
                )
                logger.info(f"💰 Bidding result: speaker_order={speaker_order}, len={len(speaker_order)}")
            else:
                speaker_order = game_state.alive_players.copy()
                if game_state.sheriff_id and game_state.sheriff_id in speaker_order:
                    speaker_order.remove(game_state.sheriff_id)
                    speaker_order.insert(0, game_state.sheriff_id)

            round_order_map: Dict[int, int] = {}

            for order_idx, player_id in enumerate(speaker_order, start=1):
                round_order_map[player_id] = order_idx

                player = game_state.players[player_id]
                observation = engine.create_observation_for_player(player_id, game_state)
                accusation_digest = self._build_accusation_digest(player_id, accusations)
                
                # Extract voting history from game_history
                vote_history = [
                    event for event in game_state.game_history
                    if event.get('type') == 'vote'
                ]
                
                # Extract elimination history
                elimination_history = [
                    event for event in game_state.game_history
                    if event.get('type') == 'elimination'
                ]

                speech_request = {
                    "type": "speak",
                    "observation": observation.model_dump(),
                    "discussion_round": disc_round,
                    "previous_speeches": all_speech_context,
                    "previous_speech_events": speech_transcript,  # All speeches, not just last 6
                    "vote_history": vote_history,  # Full voting history
                    "elimination_history": elimination_history,  # Who was eliminated when
                    "accusations_summary": accusation_digest,
                }

                response = await self._send_to_player(task_id, player_id, player.endpoint, speech_request, updater)

                if response and "speech" in response:
                    latest_speeches[player_id] = response["speech"]
                    all_speech_context[player_id] = response["speech"]

                    speech_event = {
                        'type': 'speech',
                        'round': game_state.round_number,
                        'phase': 'day',
                        'player_id': player_id,
                        'text': response["speech"],
                        'discussion_round': disc_round,
                        'order': order_idx,
                        'word_count': len(response["speech"].split()),
                        'char_len': len(response["speech"]),
                    }
                    game_state.game_history.append(speech_event)

                    speech_transcript.append({
                        'player_id': player_id,
                        'round': game_state.round_number,
                        'discussion_round': disc_round,
                        'text': response["speech"],
                        'timestamp': datetime.utcnow().isoformat()
                    })

                    if "private_thoughts" in response:
                        game_state.game_history.append({
                            'type': 'reasoning',
                            'round': game_state.round_number,
                            'phase': 'day',
                            'player_id': player_id,
                            'reasoning': response["private_thoughts"],
                            'suspicions': response.get("suspicions", {}),
                            'strategy': response.get("strategy", "unknown"),
                            'confidence': response.get("confidence", 0.5)
                        })

                    # Parse accusations/defenses
                    self._parse_accusations(
                        player_id, game_state, response["speech"], 
                        disc_round, order_idx, accusations
                    )

                    logger.info(f"[SPEECH] Player {player_id}: {response['speech']}")

                    if ENABLE_REACTIONS:
                        await self._request_reaction_feedback(
                            task_id,
                            engine,
                            game_state,
                            triggering_player=player_id,
                            discussion_round=disc_round,
                            speech_text=response["speech"],
                            reaction_slots=REACTIONS_PER_ROUND,
                            updater=updater
                        )

            last_round_order = round_order_map

        # === Phase 2: Rebuttals ===
        if accusations:
            accusation_counts = Counter()
            for entries in accusations.values():
                for entry in entries:
                    accusation_counts[entry['target']] += 1

            top_accused = [pid for pid, _ in accusation_counts.most_common(3)]
            if top_accused:
                console.print(f"[yellow]⚖️ Rebuttal phase: Players {top_accused} may respond[/yellow]")

            for accused_id in top_accused:
                if accused_id not in game_state.alive_players:
                    continue

                player = game_state.players[accused_id]
                accusation_threads = [
                    {"accuser": accuser, **entry}
                    for accuser, entries in accusations.items()
                    for entry in entries
                    if entry['target'] == accused_id
                ]

                rebuttal_request = {
                    "type": "speak",
                    "context": "rebuttal",
                    "accusations_against_you": accusation_threads,
                    "recent_speeches": speech_transcript[-3:],
                    "observation": engine.create_observation_for_player(
                        accused_id, game_state
                    ).model_dump(),
                }

                response = await self._send_to_player(task_id, accused_id, player.endpoint, rebuttal_request, updater)

                if response and "speech" in response:
                    game_state.game_history.append({
                        'type': 'speech',
                        'round': game_state.round_number,
                        'phase': 'rebuttal',
                        'player_id': accused_id,
                        'text': response["speech"],
                    })

        # === Phase 3: Voting ===
        vote_intentions_before = await self._collect_vote_intentions(
            task_id, game_state, "before_voting", updater
        )

        votes_round1, vote_meta_round1 = await self._collect_votes_with_metadata(
            task_id, game_state, round_num=1, updater=updater
        )
        await self._broadcast_vote_summary(
            task_id, game_state, votes_round1, vote_meta_round1, updater
        )

        sheriff_recommendation = None
        if game_state.sheriff_id and game_state.sheriff_id in game_state.alive_players:
            sheriff = game_state.players[game_state.sheriff_id]
            summary_request = {
                "type": "sheriff_summary",
                "votes": votes_round1,
            }
            response = await self._send_to_player(task_id, game_state.sheriff_id, sheriff.endpoint, summary_request, updater)
            if response and "recommendation" in response:
                sheriff_recommendation = response["recommendation"]
                game_state.game_history.append({
                    'type': 'sheriff_recommendation',
                    'round': game_state.round_number,
                    'sheriff_id': game_state.sheriff_id,
                    'recommended_target': sheriff_recommendation,
                    'previous_votes': votes_round1,
                })
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Sheriff recommends exiling Player {sheriff_recommendation}")
                )

        votes_round2, _ = await self._collect_votes_with_metadata(
            task_id, game_state, round_num=2, updater=updater
        )

        # Process day phase
        game_state, eliminated_id = engine.process_day_phase(
            game_state,
            latest_speeches,
            votes_round1,
            sheriff_recommendation,
            votes_round2
        )

        if eliminated_id:
            # Mark votes that led to elimination
            for event in game_state.game_history:
                if (event.get('type') == 'vote' and
                    event.get('round') == game_state.round_number and
                    event.get('target_id') == eliminated_id):
                    event['eliminated'] = True

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Day {game_state.round_number}: Player {eliminated_id} was exiled")
            )

            # Check for hunter death
            if game_state.players[eliminated_id].role == RoleType.HUNTER:
                await self._handle_hunter_shot(task_id, engine, game_state, eliminated_id, updater)
    
    def _parse_accusations(
        self,
        player_id: int,
        game_state: GameState,
        speech_text: str,
        disc_round: int,
        order_idx: int,
        accusations: Dict[int, List[Dict[str, Any]]]
    ):
        """Parse accusations and defenses from speech text"""
        speech_lower = speech_text.lower()
        
        for target_id in game_state.alive_players:
            if target_id == player_id:
                continue

            patterns = [
                f"player {target_id}",
                f"player{target_id}",
                f"#{target_id}",
                f"player {target_id},",
                f"player {target_id}.",
                f"no. {target_id}",
                f"number {target_id}",
            ]
            werewolf_keywords = [
                'werewolf', 'wolf', 'wolves',
                'suspicious', 'suspect', 'suspicion',
                'believe', 'think they', 'acting',
                'accuse', 'guilty', 'evil'
            ]
            villager_keywords = [
                'innocent', 'trust', 'villager',
                'good', 'honest', 'defend',
                'protect', 'believe in'
            ]

            target_mentioned = any(p in speech_lower for p in patterns)
            if not target_mentioned:
                continue

            werewolf_mentioned = any(w in speech_lower for w in werewolf_keywords)
            villager_mentioned = any(v in speech_lower for v in villager_keywords)

            if werewolf_mentioned:
                game_state.game_history.append({
                    'type': 'identity_claim',
                    'player_id': player_id,
                    'claimed_player_id': target_id,
                    'claimed_role': RoleType.WEREWOLF.value,
                    'round': game_state.round_number,
                    'discussion_round': disc_round
                })
                accusations.setdefault(player_id, []).append({
                    'target': target_id,
                    'discussion_round': disc_round,
                    'order': order_idx,
                    'round_number': game_state.round_number,
                    'timestamp': datetime.utcnow().isoformat()
                })
                logger.info(f"🎯 Accusation: Player {player_id} suspects Player {target_id} is a werewolf")
                break
            elif villager_mentioned:
                game_state.game_history.append({
                    'type': 'identity_claim',
                    'player_id': player_id,
                    'claimed_player_id': target_id,
                    'claimed_role': RoleType.VILLAGER.value,
                    'round': game_state.round_number,
                    'discussion_round': disc_round
                })
                logger.info(f"🛡️ Defense: Player {player_id} defends Player {target_id} as innocent")
                break
    
    async def _get_bidding_order(
        self,
        task_id: str,
        game_state: GameState,
        discussion_round: int,
        last_round_order: Optional[Dict[int, int]],
        updater: TaskUpdater
    ) -> List[int]:
        """Collect bids and return speaker order by urgency/novelty."""
        bids: Dict[int, float] = {}

        for player_id in game_state.alive_players:
            player = game_state.players[player_id]

            bid_request = {
                "type": "bid_request",
                "game_state": {
                    "round": game_state.round_number,
                    "alive_players": game_state.alive_players,
                    "sheriff_id": game_state.sheriff_id,
                },
                "context": f"Discussion round {discussion_round}",
                "round": game_state.round_number,
            }

            response = await self._send_to_player(task_id, player_id, player.endpoint, bid_request, updater)

            if response and "bid_value" in response:
                normalized = self._normalize_bid_value(
                    response.get("bid_value"),
                    response.get("priority")
                )
                if last_round_order and player_id in last_round_order:
                    penalty = max(0, 4 - last_round_order[player_id]) * 5
                    normalized = max(0.0, normalized - penalty)
                bids[player_id] = normalized
                logger.info(f"💰 Player {player_id} bid: raw={response.get('bid_value')}, priority={response.get('priority')}, normalized={normalized:.1f}")

                game_state.game_history.append({
                    'type': 'bid',
                    'round': game_state.round_number,
                    'discussion_round': discussion_round,
                    'player_id': player_id,
                    'bid_value': normalized,
                    'raw_bid': response.get("bid_value"),
                    'priority': response.get("priority"),
                    'bid_reasoning': response.get("bid_reasoning", ""),
                })
            else:
                bids[player_id] = 50.0

        sorted_players = sorted(bids.keys(), key=lambda x: bids[x], reverse=True)
        return sorted_players
    
    def _normalize_bid_value(self, raw_value: Any, priority_hint: Any) -> float:
        """Convert textual/structured bid data into a normalized numeric value."""
        tier_map = {
            "low": 25.0,
            "medium": 50.0,
            "med": 50.0,
            "high": 75.0,
            "urgent": 90.0,
        }

        if isinstance(raw_value, (int, float)):
            value = float(raw_value)
        elif isinstance(raw_value, str):
            lowered = raw_value.strip().lower()
            if lowered.replace('.', '', 1).isdigit():
                value = float(lowered)
            else:
                value = tier_map.get(lowered, 50.0)
        else:
            value = 50.0

        if isinstance(priority_hint, str):
            priority = priority_hint.strip().lower()
            value += tier_map.get(priority, 0.0) * 0.2

        return max(0.0, min(100.0, value))
    
    def _build_accusation_digest(
        self,
        player_id: int,
        accusations: Dict[int, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Summarize how often and recently a player has been accused."""
        total = 0
        recent_entries: List[Dict[str, Any]] = []
        for accuser, entries in accusations.items():
            relevant = [entry for entry in entries if entry['target'] == player_id]
            if not relevant:
                continue
            total += len(relevant)
            for entry in relevant:
                entry_with_accuser = entry.copy()
                entry_with_accuser['accuser'] = accuser
                recent_entries.append(entry_with_accuser)

        recent_entries.sort(key=lambda e: (e['round_number'], e['order']), reverse=True)
        return {
            "total_against_you": total,
            "recent": recent_entries[:3]
        }
    
    async def _request_reaction_feedback(
        self,
        task_id: str,
        engine: WerewolfGameEngine,
        game_state: GameState,
        triggering_player: int,
        discussion_round: int,
        speech_text: str,
        reaction_slots: int,
        updater: TaskUpdater
    ):
        """Ping a few listeners for quick reactions to keep everyone engaged."""
        listeners = [pid for pid in game_state.alive_players if pid != triggering_player]
        if not listeners:
            return
        sample_size = min(reaction_slots, len(listeners))
        responders = random.sample(listeners, sample_size)

        for responder_id in responders:
            responder = game_state.players[responder_id]
            reaction_request = {
                "type": "reaction",
                "context": "post_speech_feedback",
                "speaker_id": triggering_player,
                "speech_text": speech_text,
                "discussion_round": discussion_round,
                "observation": engine.create_observation_for_player(
                    responder_id, game_state
                ).model_dump(),
            }

            response = await self._send_to_player(task_id, responder_id, responder.endpoint, reaction_request, updater)
            if response and "reaction" in response:
                game_state.game_history.append({
                    'type': 'reaction',
                    'round': game_state.round_number,
                    'discussion_round': discussion_round,
                    'player_id': responder_id,
                    'target_id': triggering_player,
                    'text': response["reaction"]
                })
    
    async def _broadcast_vote_summary(
        self,
        task_id: str,
        game_state: GameState,
        votes: Dict[int, int],
        vote_meta: Dict[int, Dict[str, Any]],
        updater: TaskUpdater
    ):
        """Share the current tally with average confidence for transparency."""
        if not votes:
            return

        tally = Counter(votes.values())
        lines = []
        for target_id, count in tally.most_common():
            confidences = [
                vote_meta.get(pid, {}).get("confidence", 0.5)
                for pid, choice in votes.items()
                if choice == target_id
            ]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
            lines.append(f"P{target_id}: {count} votes (avg conf {avg_conf:.2f})")

        message = "; ".join(lines)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Current tally → {message}")
        )
    
    async def _collect_vote_intentions(
        self,
        task_id: str,
        game_state: GameState,
        phase: str,
        updater: TaskUpdater
    ) -> Dict[int, Dict]:
        """Collect vote intentions from players for vote swing tracking."""
        intentions = {}

        for player_id in game_state.alive_players:
            player = game_state.players[player_id]

            intention_request = {
                "type": "vote_intention",
                "phase": phase,
                "candidates": game_state.alive_players.copy()
            }

            try:
                response = await self._send_to_player(task_id, player_id, player.endpoint, intention_request, updater)

                if response and "target" in response:
                    intentions[player_id] = {
                        "target": response["target"],
                        "confidence": response.get("confidence", 0.5)
                    }
            except Exception as e:
                logger.debug(f"Could not get vote intention from player {player_id}: {e}")

        return intentions
    
    async def _collect_votes_with_metadata(
        self,
        task_id: str,
        game_state: GameState,
        round_num: int,
        updater: TaskUpdater
    ) -> tuple[Dict[int, int], Dict[int, Dict[str, Any]]]:
        """Collect votes from all alive players with metadata."""
        votes = {}
        metadata: Dict[int, Dict[str, Any]] = {}

        for player_id in game_state.alive_players:
            player = game_state.players[player_id]

            vote_request = {
                "type": "vote",
                "round": round_num,
                "candidates": game_state.alive_players.copy(),
                "current_votes": votes.copy()  # Show who has voted so far
            }

            response = await self._send_to_player(task_id, player_id, player.endpoint, vote_request, updater)

            if response and "vote" in response:
                votes[player_id] = response["vote"]
                confidence = response.get("confidence", 0.5)
                metadata[player_id] = {
                    "confidence": confidence,
                    "reasoning": response.get("reasoning")
                }
                
                logger.info(f"🗳️ Player {player_id} votes for Player {response['vote']}")

                # Log vote event
                game_state.game_history.append({
                    'type': 'vote',
                    'round': game_state.round_number,
                    'phase': f'vote_round_{round_num}',
                    'voting_round': round_num,
                    'voter_id': player_id,
                    'player_id': player_id,
                    'target_id': response["vote"],
                    'confidence': confidence,
                    'reasoning': response.get("reasoning")
                })

                action = Action(
                    action_type=ActionType.VOTE,
                    player_id=player_id,
                    target_id=response["vote"]
                )
                self.player_actions[task_id][player_id].append(action)

        return votes, metadata
    
    async def _handle_hunter_shot(
        self,
        task_id: str,
        engine: WerewolfGameEngine,
        game_state: GameState,
        hunter_id: int,
        updater: TaskUpdater
    ):
        """Handle hunter's shot when eliminated"""
        hunter = game_state.players[hunter_id]
        
        shot_request = {
            "type": "hunter_shoot",
            "targets": game_state.alive_players.copy()
        }
        
        response = await self._send_to_player(task_id, hunter_id, hunter.endpoint, shot_request, updater)
        
        if response and "target" in response:
            target = response["target"]
            game_state = engine.handle_hunter_death(game_state, hunter_id, target)
            
            # Track hunter action
            action = Action(
                action_type=ActionType.SHOOT,
                player_id=hunter_id,
                target_id=target
            )
            self.player_actions[task_id][hunter_id].append(action)
            
            logger.info(f"Hunter (Player {hunter_id}) shot Player {target}")
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Hunter (Player {hunter_id}) shot Player {target}")
            )
    
    async def _finalize_game(
        self,
        task_id: str,
        engine: WerewolfGameEngine,
        game_state: GameState,
        updater: TaskUpdater
    ):
        """Calculate metrics and produce final assessment"""
        logger.info(f"Finalizing game {task_id}")
        
        # Pretty print game over
        if game_state.winner:
            winner_color = "red" if game_state.winner == Camp.WOLF else "green"
            winner_text = f"[bold {winner_color}]🎮 GAME OVER - {game_state.winner.value.upper()} CAMP WINS![/bold {winner_color}]"
        else:
            winner_color = "yellow"
            winner_text = "[bold yellow]🎮 GAME OVER - NO WINNER (MAX ROUNDS)[/bold yellow]"
        
        console.print()
        console.print(Panel(
            winner_text,
            border_style=winner_color,
            box=box.DOUBLE
        ))
        console.print(f"[bold]Total Rounds:[/bold] {game_state.round_number}")
        console.print()
        
        # Update status
        self.task_status[task_id] = "calculating_metrics"
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Calculating metrics...")
        )
        
        # Ensure winner is determined
        if game_state.winner is None:
            logger.warning("No winner determined - calculating by remaining players")
            good_alive = sum(1 for p in game_state.players.values() 
                           if p.is_alive and p.camp == Camp.GOOD)
            wolf_alive = sum(1 for p in game_state.players.values() 
                           if p.is_alive and p.camp == Camp.WOLF)
            
            if good_alive > wolf_alive:
                game_state.winner = Camp.GOOD
            elif wolf_alive > good_alive:
                game_state.winner = Camp.WOLF
            else:
                game_state.winner = Camp.GOOD  # Tie defaults to good
            logger.info(f"Winner set to {game_state.winner.value}")
        
        # Calculate comprehensive metrics
        logger.info("Calculating comprehensive metrics (IRS/MSS require LLM API)...")
        calculator = MetricsCalculator(use_llm_metrics=True, use_batch_evaluation=True)
        
        try:
            game_result = await calculator.calculate_game_result(
                game_state,
                self.player_actions[task_id]
            )
            logger.info("✓ Metrics calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}", exc_info=True)
            raise
        
        # Calculate advanced metrics
        logger.info("Calculating advanced metrics...")
        try:
            advanced_calculator = AdvancedMetricsCalculator()
            
            # Prepare data
            wolf_ids = [
                pid for pid, player in game_state.players.items()
                if player.role == RoleType.WEREWOLF
            ]
            
            # Convert game_history to dict format
            history_dicts = []
            for event in game_state.game_history:
                if hasattr(event, 'dict'):
                    history_dicts.append(event.dict())
                elif isinstance(event, dict):
                    history_dicts.append(event)
            
            # Convert players to dict format
            players_dict = {}
            for pid, player in game_state.players.items():
                players_dict[pid] = {
                    "role": player.role.value,
                    "camp": player.camp.value,
                    "is_alive": player.is_alive
                }
            
            advanced_metrics = advanced_calculator.calculate_all_metrics(
                history_dicts,
                players_dict,
                wolf_ids
            )
            
            # Add to game_result
            game_result.advanced_metrics = advanced_metrics
            logger.info("✓ Advanced metrics calculated")
            
        except Exception as e:
            logger.warning(f"Could not calculate advanced metrics: {e}")
            advanced_metrics = {
                "manipulation_success_d1": 0.0,
                "manipulation_success_d2": 0.0,
                "auto_sabotage": 0.0,
                "day1_wolf_eliminated": 0.0,
                "vote_swings": []
            }
            game_result.advanced_metrics = advanced_metrics
        
        # Display metrics with Rich
        metrics_table = Table(title="📊 Player Performance Metrics", box=box.ROUNDED)
        metrics_table.add_column("Player", style="cyan", justify="center")
        metrics_table.add_column("Role", style="yellow")
        metrics_table.add_column("IRS", justify="right")
        metrics_table.add_column("VRS", justify="right")
        metrics_table.add_column("MSS", justify="right")
        metrics_table.add_column("SR", justify="right")
        metrics_table.add_column("Survived", justify="center")

        def _fmt_pct_optional(x: Optional[float]) -> str:
            return "N/A" if x is None else f"{x:.1%}"
        
        for player_id, player in game_state.players.items():
            player_metrics = game_result.player_metrics[player_id]
            role_emoji = {
                RoleType.WEREWOLF: "🐺",
                RoleType.SEER: "🔮",
                RoleType.WITCH: "🧪",
                RoleType.HUNTER: "🎯",
                RoleType.GUARD: "🛡️",
                RoleType.VILLAGER: "👤"
            }.get(player.role, "❓")
            
            survived = "✅" if player.is_alive else "❌"
            
            metrics_table.add_row(
                f"{player_id}",
                f"{role_emoji} {player.role.value}",
                _fmt_pct_optional(player_metrics.irs),
                f"{player_metrics.vrs:.1%}",
                f"{player_metrics.mss:.1%}",
                f"{player_metrics.sr:.1%}",
                survived
            )
        
        console.print(metrics_table)
        console.print()
        
        # Determine tested agent info (for AgentBeats mode)
        tested_player_id = None
        if task_id in self.agentbeats_adapter:
            tested_player_id = self.agentbeats_adapter[task_id].tested_player_id
            logger.info(f"🎯 Task {task_id}: Tested player ID = {tested_player_id}")
        
        # Store final results
        game_result_data = {
            "status": "complete",
            "game_id": game_state.game_id,
            "winner": game_state.winner.value if game_state.winner else None,
            "rounds": game_state.round_number,
            "players": {
                str(pid): {
                    "agent_id": game_state.player_agent_names.get(pid, f"player-{pid}"),
                    "role": player.role.value,
                    "camp": player.camp.value,
                    "is_alive": player.is_alive,
                    "metrics": game_result.player_metrics[pid].model_dump()
                }
                for pid, player in game_state.players.items()
            },
            "role_metrics": {
                str(pid): metrics.model_dump() if metrics else {}
                for pid, metrics in game_result.role_metrics.items()
            },
            "aggregate_metrics": {
                "avg_irs_good": game_result.avg_irs_good,
                "avg_irs_wolf": game_result.avg_irs_wolf,
                "avg_vrs_good": game_result.avg_vrs_good,
                "avg_vrs_wolf": game_result.avg_vrs_wolf,
                "avg_mss": game_result.avg_mss,
            },
            "advanced_metrics": advanced_metrics,
            "game_history": [
                event.dict() if hasattr(event, 'dict') else event
                for event in game_state.game_history
            ]
        }
        
        # Add tested_agent info for AgentBeats mode
        if tested_player_id and tested_player_id in game_state.players:
            tested_player = game_state.players[tested_player_id]
            winner_camp = game_state.winner
            tested_won = (tested_player.camp == Camp.GOOD and winner_camp == Camp.GOOD) or \
                        (tested_player.camp == Camp.WOLF and winner_camp == Camp.WOLF)
            
            game_result_data["tested_agent"] = {
                "player_id": tested_player_id,
                "agent_id": game_state.player_agent_names.get(tested_player_id, "tested_agent"),
                "role": tested_player.role.value,
                "camp": tested_player.camp.value,
                "survived": tested_player.is_alive,
                "won_game": tested_won
            }
            logger.info(f"✅ Added tested_agent info for task {task_id}, player {tested_player_id}")
        elif tested_player_id:
            logger.warning(f"⚠️ Task {task_id}: tested_player_id={tested_player_id} not found in game_state.players (keys: {list(game_state.players.keys())})")
        else:
            logger.warning(f"⚠️ Task {task_id}: No tested_player_id found (adapter exists: {task_id in self.agentbeats_adapter})")
        
        self.game_results[task_id] = game_result_data
        
        self.task_status[task_id] = "complete"
        logger.info(f"Game {task_id} finalized. Winner: {game_state.winner.value}")
    
    def _aggregate_game_results(self, all_game_results: List[Dict[str, Any]], is_agentbeats: bool, participant_agentbeats_id: str = None, participant_name: str = None) -> Dict[str, Any]:
        """
        Aggregate results from multiple games.
        
        Args:
            all_game_results: List of individual game results
            is_agentbeats: If True, aggregate only tested agent metrics
            participant_agentbeats_id: AgentBeats ID of the tested participant
            
        Returns:
            Aggregated results across all games
        """
        from collections import defaultdict
        
        num_games = len(all_game_results)
        
        if is_agentbeats:
            # AgentBeats mode: aggregate tested agent metrics only
            return self._aggregate_agentbeats_results(all_game_results, participant_agentbeats_id, participant_name)
        else:
            # Tournament mode: aggregate all players
            return self._aggregate_tournament_results(all_game_results)
    
    def _aggregate_agentbeats_results(self, all_game_results: List[Dict[str, Any]], participant_agentbeats_id: str = None, participant_name: str = None) -> Dict[str, Any]:
        """Aggregate results for AgentBeats mode (single tested agent)."""
        from collections import Counter
        
        num_games = len(all_game_results)
        logger.info(f"🔄 Aggregating {num_games} game results")
        
        # Debug: Check what we received
        if all_game_results:
            first_keys = list(all_game_results[0].keys())
            logger.info(f"📋 First game result keys: {first_keys}")
            if "results" in first_keys:
                logger.error("❌ ERROR: Received already-aggregated results! Expected individual game results.")
                return all_game_results[0]  # Return as-is if already aggregated
        
        # Find tested agent across games (should be consistent player_id or marked)
        tested_metrics = []
        roles_played = []
        games_won = 0
        games_survived = 0
        
        for game_result in all_game_results:
            # In AgentBeats results, there should be a tested_agent field
            if "tested_agent" in game_result:
                tested_agent = game_result["tested_agent"]
                player_id = tested_agent.get("player_id")
                
                # Get metrics from players[player_id]["metrics"]
                if player_id and "players" in game_result:
                    player_data = game_result["players"].get(str(player_id), {})
                    metrics = player_data.get("metrics", {})
                    if metrics:
                        tested_metrics.append(metrics)
                
                # Get role, survival, and win info from tested_agent
                roles_played.append(tested_agent.get("role"))
                
                if tested_agent.get("survived", False):
                    games_survived += 1
                
                if tested_agent.get("won_game", False):
                    games_won += 1
        
        # Aggregate metrics
        if not tested_metrics:
            logger.warning("No tested agent metrics found in game results")
            return {"error": "No tested agent data found"}
        
        # Calculate averages
        def avg_metric(metric_name):
            values = [m.get(metric_name, 0) for m in tested_metrics if m.get(metric_name) is not None]
            return sum(values) / len(values) if values else 0.0
        
        # Count roles
        role_counts = Counter(roles_played)
        
        # Build results in AgentBeats format
        results = []
        for i, game_result in enumerate(all_game_results, 1):
            if "tested_agent" in game_result:
                tested = game_result["tested_agent"]
                player_id = tested.get("player_id")
                
                # Get metrics for this game
                player_metrics = {}
                role_metrics = {}
                advanced_metrics = {}
                if player_id and "players" in game_result:
                    player_data = game_result["players"].get(str(player_id), {})
                    player_metrics = player_data.get("metrics", {})
                    
                    # Get role-specific metrics
                    if "role_metrics" in game_result:
                        role_metrics = game_result["role_metrics"].get(str(player_id), {})
                    
                    # Get advanced metrics
                    if "advanced_metrics" in game_result:
                        advanced_metrics = game_result["advanced_metrics"]
                
                game_result_entry = {
                    "game_number": game_result.get("game_number", i),
                    "role": tested.get("role"),
                    "survived": tested.get("survived", False),
                    "won": tested.get("won_game", False),
                    "rounds": game_result.get("rounds", 0),
                    "winner": game_result.get("winner"),
                    "metrics": {
                        # Core metrics
                        "irs": player_metrics.get("irs"),
                        "vrs": player_metrics.get("vrs"),
                        "mss": player_metrics.get("mss"),
                        "persuasion_score": player_metrics.get("persuasion_score"),
                        "deception_score": player_metrics.get("deception_score"),
                        
                        # Role-specific metrics
                        "seer_check_accuracy": role_metrics.get("seer_check_accuracy"),
                        "witch_heal_effectiveness": role_metrics.get("witch_heal_effectiveness"),
                        "witch_poison_effectiveness": role_metrics.get("witch_poison_effectiveness"),
                        "hunter_shot_accuracy": role_metrics.get("hunter_shot_accuracy"),
                        "guard_protection_success": role_metrics.get("guard_protection_success"),
                        
                        # Advanced social metrics (if available)
                        "manipulation_success_d1": advanced_metrics.get("manipulation_success_d1"),
                        "manipulation_success_d2": advanced_metrics.get("manipulation_success_d2"),
                        "auto_sabotage": advanced_metrics.get("auto_sabotage")
                    }
                }
                results.append(game_result_entry)
        
        # Calculate aggregate statistics across all games
        def avg_metric(metric_name):
            values = [
                g.get("metrics", {}).get(metric_name) 
                for g in results 
                if g.get("metrics", {}).get(metric_name) is not None
            ]
            return sum(values) / len(values) if values else None
        
        # Return single aggregated result for all games
        # Client will wrap this in results array
        return {
            "num_games": num_games,
            "games_completed": len(tested_metrics),
            "win_rate": games_won / num_games,
            "survival_rate": games_survived / num_games,
            
            # Average core metrics
            "average_irs": avg_metric("irs"),
            "average_vrs": avg_metric("vrs"),
            "average_mss": avg_metric("mss"),
            "average_persuasion": avg_metric("persuasion_score"),
            "average_deception": avg_metric("deception_score"),
            
            # Average role-specific metrics
            "average_seer_check_accuracy": avg_metric("seer_check_accuracy"),
            "average_witch_heal_effectiveness": avg_metric("witch_heal_effectiveness"),
            "average_witch_poison_effectiveness": avg_metric("witch_poison_effectiveness"),
            "average_hunter_shot_accuracy": avg_metric("hunter_shot_accuracy"),
            "average_guard_protection_success": avg_metric("guard_protection_success"),
            
            # Average advanced social metrics
            "average_manipulation_success_d1": avg_metric("manipulation_success_d1"),
            "average_manipulation_success_d2": avg_metric("manipulation_success_d2"),
            "average_auto_sabotage": avg_metric("auto_sabotage"),
            
            "roles_played": dict(role_counts),
            "games": results  # Individual game details
        }
    
    def _aggregate_tournament_results(self, all_game_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results for tournament mode (all players)."""
        # Tournament aggregation - track all players across games
        # This is more complex and would involve ELO calculations, etc.
        # For now, return summary
        
        return {
            "status": "complete",
            "mode": "tournament",
            "num_games": len(all_game_results),
            "games": all_game_results,
            "message": "Tournament mode aggregation - see individual game results"
        }