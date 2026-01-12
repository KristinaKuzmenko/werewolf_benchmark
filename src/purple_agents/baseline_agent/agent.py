"""
Baseline Werewolf Agent - Game logic implementation.
Simple rule-based strategy for different roles.
"""
import json
import logging
import random
from typing import Dict, List, Optional, Any

from a2a.server.tasks import TaskUpdater
from a2a.types import Message
from a2a.utils import get_message_text

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


class BaselineWerewolfAgent:
    """
    Baseline purple agent with simple strategies for each role.
    Serves as a comparison baseline for more advanced agents.
    """
    
    def __init__(self, agent_id: str = "baseline-agent"):
        self.agent_id = agent_id
        
        # Agent state (reset on game_start)
        self.player_id: Optional[int] = None
        self.role: Optional[str] = None
        self.camp: Optional[str] = None
        self.game_id: Optional[str] = None
        
        # Memory
        self.alive_players: List[int] = []
        self.suspected_werewolves: Dict[int, float] = {}
        self.known_roles: Dict[int, str] = {}
        self.werewolf_team: List[int] = []
        self.speeches_history: List[Dict] = []
    
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Process incoming A2A message and respond.
        
        Args:
            message: A2A Message with JSON-encoded game message
            updater: TaskUpdater for sending response
        """
        input_text = get_message_text(message)
        
        try:
            game_message = json.loads(input_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
            response = {"status": "error", "message": f"Invalid JSON: {e}"}
            await updater.add_artifact(
                [{"type": "text", "text": json.dumps(response)}]
            )
            return
        
        msg_type = game_message.get("type")
        logger.info(f"Agent {self.agent_id} received: {msg_type}")
        
        # Route to appropriate handler
        handlers = {
            "game_start": self._handle_game_start,
            "sheriff_election": self._handle_sheriff_election,
            "night_action": self._handle_night_action,
            "speak": self._handle_speak,
            "vote": self._handle_vote,
            "sheriff_summary": self._handle_sheriff_summary,
            "hunter_shoot": self._handle_hunter_shoot,
        }
        
        handler = handlers.get(msg_type)
        if handler:
            response = await handler(game_message)
        else:
            response = {"status": "acknowledged"}
        
        # Send response as JSON text
        response_text = json.dumps(response)
        await updater.add_artifact(
            [{"type": "text", "text": response_text}]
        )
    
    async def _handle_game_start(self, message: Dict) -> Dict:
        """Initialize agent state when game starts"""
        self.game_id = message.get("game_id")
        self.player_id = message.get("player_id")
        self.role = message.get("role")
        self.camp = message.get("camp")
        self.alive_players = message.get("alive_players", [])
        
        # Reset state
        self.suspected_werewolves = {}
        self.known_roles = {}
        self.speeches_history = []
        
        if "werewolf_team" in message:
            self.werewolf_team = message["werewolf_team"]
        else:
            self.werewolf_team = []
        
        # Initialize suspicion scores with some randomness
        for pid in self.alive_players:
            if pid != self.player_id and pid not in self.werewolf_team:
                self.suspected_werewolves[pid] = random.uniform(0.2, 0.6)
        
        logger.info(
            f"Agent {self.agent_id} (Player {self.player_id}): "
            f"Role={self.role}, Camp={self.camp}"
        )
        
        # Pretty print role assignment
        role_emoji = {
            "werewolf": "🐺",
            "seer": "🔮",
            "witch": "🧪",
            "hunter": "🎯",
            "guard": "🛡️",
            "villager": "👤"
        }.get(self.role, "❓")
        
        camp_color = "red" if self.camp == "wolf" else "green"
        console.print(
            f"[bold {camp_color}]{role_emoji} Player {self.player_id}: "
            f"{self.role.upper()} ({self.camp} camp)[/bold {camp_color}]"
        )
        
        return {"status": "initialized"}
    
    async def _handle_sheriff_election(self, message: Dict) -> Dict:
        """Vote for sheriff - simple strategy"""
        candidates = message.get("candidates", [])
        
        if not candidates:
            return {"vote": self.player_id or 0}
        
        # Don't vote for known werewolves
        safe_candidates = [
            c for c in candidates
            if c not in self.werewolf_team and
               self.known_roles.get(c) != "werewolf"
        ]
        
        if not safe_candidates:
            safe_candidates = candidates
        
        # Vote for someone with low suspicion
        vote = min(
            safe_candidates,
            key=lambda c: self.suspected_werewolves.get(c, 0.5)
        )
        
        return {"vote": vote}
    
    async def _handle_night_action(self, message: Dict) -> Dict:
        """Handle night actions based on role"""
        observation = message.get("observation", {})
        
        if self.role == "werewolf":
            return await self._werewolf_night_action(observation)
        elif self.role == "seer":
            return await self._seer_night_action(observation)
        elif self.role == "guard":
            return await self._guard_night_action(observation)
        elif self.role == "witch":
            return await self._witch_night_action(observation)
        
        return {"action": None}
    
    async def _werewolf_night_action(self, observation: Dict) -> Dict:
        """Werewolf: Kill a non-werewolf player"""
        alive = observation.get("alive_players", self.alive_players)
        
        # Kill non-werewolves
        targets = [p for p in alive if p not in self.werewolf_team]
        
        if not targets:
            return {"action": None}
        
        # Prioritize killing suspected gods (low suspicion = likely god)
        god_suspects = [
            p for p in targets
            if self.suspected_werewolves.get(p, 0.5) < 0.3
        ]
        
        target = random.choice(god_suspects if god_suspects else targets)
        
        return {
            "action": {
                "action_type": "kill",
                "player_id": self.player_id,
                "target_id": target
            }
        }
    
    async def _seer_night_action(self, observation: Dict) -> Dict:
        """Seer: Check most suspicious player"""
        alive = observation.get("alive_players", self.alive_players)
        
        # Check someone we don't know yet
        unknown = [
            p for p in alive
            if p != self.player_id and p not in self.known_roles
        ]
        
        if not unknown:
            return {"action": None}
        
        # Check most suspicious unknown player
        target = max(
            unknown,
            key=lambda p: self.suspected_werewolves.get(p, 0.5)
        )
        
        # Process result if available
        role_info = observation.get("role_specific_info", {})
        if "check_result" in role_info:
            checked_player = role_info.get("checked_player")
            is_werewolf = role_info.get("check_result")
            if checked_player is not None:
                self.known_roles[checked_player] = "werewolf" if is_werewolf else "villager"
                self.suspected_werewolves[checked_player] = 1.0 if is_werewolf else 0.0
        
        return {
            "action": {
                "action_type": "check",
                "player_id": self.player_id,
                "target_id": target
            }
        }
    
    async def _guard_night_action(self, observation: Dict) -> Dict:
        """Guard: Protect a likely target"""
        alive = observation.get("alive_players", self.alive_players)
        role_info = observation.get("role_specific_info", {})
        last_protected = role_info.get("last_protected")
        
        # Don't protect self or last protected (game rules)
        valid_targets = [
            p for p in alive
            if p != self.player_id and p != last_protected
        ]
        
        if not valid_targets:
            return {"action": None}
        
        # Protect someone with low suspicion (likely god)
        target = min(
            valid_targets,
            key=lambda p: self.suspected_werewolves.get(p, 0.5)
        )
        
        return {
            "action": {
                "action_type": "protect",
                "player_id": self.player_id,
                "target_id": target
            }
        }
    
    async def _witch_night_action(self, observation: Dict) -> Dict:
        """Witch: Use potions strategically"""
        role_info = observation.get("role_specific_info", {})
        victim = role_info.get("werewolf_victim")
        heal_available = role_info.get("heal_available", False)
        poison_available = role_info.get("poison_available", False)
        
        # Save first night victim if it's not self
        if heal_available and victim and victim != self.player_id:
            # Don't save suspected werewolves
            if self.suspected_werewolves.get(victim, 0.5) < 0.7:
                return {
                    "action": {
                        "action_type": "heal",
                        "player_id": self.player_id,
                        "target_id": victim
                    }
                }
        
        # Poison most suspicious player
        if poison_available:
            alive = observation.get("alive_players", self.alive_players)
            suspects = [
                p for p in alive
                if p != self.player_id and self.suspected_werewolves.get(p, 0) > 0.8
            ]
            
            if suspects:
                target = max(suspects, key=lambda p: self.suspected_werewolves.get(p, 0))
                return {
                    "action": {
                        "action_type": "poison",
                        "player_id": self.player_id,
                        "target_id": target
                    }
                }
        
        return {"action": None}
    
    async def _handle_speak(self, message: Dict) -> Dict:
        """Make a speech during day phase"""
        observation = message.get("observation", {})
        self.alive_players = observation.get("alive_players", self.alive_players)
        
        # Simple speech generation based on role
        if self.role == "werewolf":
            speech = self._werewolf_speech()
        elif self.role == "seer":
            speech = self._seer_speech()
        else:
            speech = self._villager_speech()
        
        self.speeches_history.append({
            "player_id": self.player_id,
            "speech": speech
        })
        
        return {"speech": speech}
    
    def _werewolf_speech(self) -> str:
        """Werewolf tries to appear innocent"""
        non_wolf = self._get_random_non_wolf()
        templates = [
            f"I think we should look carefully at Player {non_wolf}. They've been very quiet.",
            "We need to work together to find the werewolves. I'm a villager.",
            f"I'm suspicious of Player {non_wolf}'s voting pattern.",
            f"Has anyone noticed Player {non_wolf} acting strange?",
        ]
        return random.choice(templates)
    
    def _seer_speech(self) -> str:
        """Seer may reveal information strategically"""
        werewolves_known = [
            p for p, role in self.known_roles.items()
            if role == "werewolf"
        ]
        
        if werewolves_known and random.random() > 0.5:
            target = werewolves_known[0]
            return f"I checked Player {target} and they are a werewolf! I am the Seer."
        
        return "I'm observing everyone's behavior carefully."
    
    def _villager_speech(self) -> str:
        """Villager shares observations"""
        # Build suspicion if not already done
        if not self.suspected_werewolves and self.alive_players:
            for pid in self.alive_players:
                if pid != self.player_id and pid not in self.werewolf_team:
                    self.suspected_werewolves[pid] = random.uniform(0.1, 0.9)
        
        if self.suspected_werewolves:
            most_suspected = max(
                self.suspected_werewolves.items(),
                key=lambda x: x[1]
            )[0]
            
            templates = [
                f"I suspect Player {most_suspected} is a werewolf based on their behavior.",
                f"Player {most_suspected} has been acting suspicious. I think they're a werewolf.",
                f"I believe Player {most_suspected} is a wolf. We should vote them out.",
            ]
            return random.choice(templates)
        
        return "I'm a villager trying to help find the werewolves."
    
    async def _handle_vote(self, message: Dict) -> Dict:
        """Vote for who to exile"""
        candidates = message.get("candidates", [])
        round_num = message.get("round", 1)
        
        if not candidates:
            return {"vote": self.player_id or 0}
        
        # Vote for most suspicious player
        if self.role == "werewolf":
            # Werewolves vote for non-werewolves
            targets = [c for c in candidates if c not in self.werewolf_team]
            if targets:
                # Vote for least suspicious (to avoid attention)
                vote = min(targets, key=lambda c: self.suspected_werewolves.get(c, 0.5))
            else:
                vote = random.choice(candidates)
        else:
            # Good camp votes for most suspicious
            vote = max(
                candidates,
                key=lambda c: self.suspected_werewolves.get(c, 0)
            )
        
        # Update suspicion based on voting patterns
        self._update_suspicions(candidates, vote)
        
        return {"vote": vote}
    
    async def _handle_sheriff_summary(self, message: Dict) -> Dict:
        """Sheriff makes recommendation"""
        votes = message.get("votes", {})
        
        # Count votes
        vote_counts: Dict[int, int] = {}
        for voter, target in votes.items():
            vote_counts[target] = vote_counts.get(target, 0) + 1
        
        # Recommend most voted if they're suspicious, otherwise most suspicious
        if vote_counts:
            most_voted = max(vote_counts, key=lambda k: vote_counts[k])
            
            if self.suspected_werewolves.get(most_voted, 0) > 0.5:
                recommendation = most_voted
            else:
                # Override with most suspicious
                if self.suspected_werewolves:
                    recommendation = max(
                        self.suspected_werewolves.items(),
                        key=lambda x: x[1]
                    )[0]
                else:
                    recommendation = most_voted
        else:
            if self.suspected_werewolves:
                recommendation = max(
                    self.suspected_werewolves.items(),
                    key=lambda x: x[1]
                )[0]
            else:
                recommendation = self.player_id or 0
        
        return {"recommendation": recommendation}
    
    async def _handle_hunter_shoot(self, message: Dict) -> Dict:
        """Hunter shoots when eliminated"""
        targets = message.get("targets", [])
        
        if not targets:
            return {"target": self.player_id or 0}
        
        # Shoot most suspicious player
        if self.suspected_werewolves:
            target = max(
                [t for t in targets if t in self.suspected_werewolves],
                key=lambda t: self.suspected_werewolves.get(t, 0),
                default=random.choice(targets)
            )
        else:
            target = random.choice(targets)
        
        return {"target": target}
    
    def _get_random_non_wolf(self) -> int:
        """Get random player that's not in werewolf team"""
        candidates = [
            p for p in self.alive_players
            if p != self.player_id and p not in self.werewolf_team
        ]
        return random.choice(candidates) if candidates else (self.player_id or 0)
    
    def _update_suspicions(self, candidates: List[int], my_vote: int):
        """Simple suspicion update based on voting"""
        for candidate in candidates:
            if candidate != my_vote:
                self.suspected_werewolves[candidate] = min(
                    self.suspected_werewolves.get(candidate, 0.1) + 0.05,
                    1.0
                )
