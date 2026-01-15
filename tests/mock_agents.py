"""
Mock agents for testing metric calculation.
These agents have predetermined behaviors to validate evaluation correctness.
"""
import random
from typing import Dict, Any, Optional


class MockAgentBase:
    """Base class for mock agents."""
    
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.role = None
        self.game_state = {}
    
    def handle_game_start(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize game state."""
        self.role = message.get("role")
        self.game_state = message.get("observation", {})
        return {"status": "ready"}
    
    def handle_night_action(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Default night action - random target."""
        alive_players = message.get("observation", {}).get("alive_players", [])
        if alive_players:
            target = random.choice([p for p in alive_players if p != self.player_id])
            return {"target_id": target}
        return {}
    
    def handle_speak(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Generate speech."""
        return {"speech": "Generic statement."}
    
    def handle_vote(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Cast vote."""
        candidates = message.get("candidates", [])
        if candidates:
            return {"vote": random.choice(candidates)}
        return {}


class PerfectSeerMock(MockAgentBase):
    """
    Perfect Seer agent that always checks suspected wolves.
    Expected metrics:
    - seer_check_accuracy > 0.8 (checks wolves preferentially)
    - IRS > 0.7 (knows who wolves are)
    - win_rate > 0.6 (helps village win)
    """
    
    def __init__(self, player_id: int, wolf_ids: list = None):
        super().__init__(player_id)
        self.wolf_ids = wolf_ids or [2, 3]  # Known wolf IDs
        self.checked_players = []
    
    def handle_night_action(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Seer checks wolves preferentially."""
        if self.role != "seer":
            return super().handle_night_action(message)
        
        alive_players = message.get("observation", {}).get("alive_players", [])
        
        # Check unchecked wolves first
        unchecked_wolves = [w for w in self.wolf_ids if w in alive_players and w not in self.checked_players]
        if unchecked_wolves:
            target = unchecked_wolves[0]
            self.checked_players.append(target)
            return {"action_type": "check", "target_id": target}
        
        # Then check other players
        unchecked = [p for p in alive_players if p != self.player_id and p not in self.checked_players]
        if unchecked:
            target = unchecked[0]
            self.checked_players.append(target)
            return {"action_type": "check", "target_id": target}
        
        return {}
    
    def handle_speak(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Speak with evidence about wolves."""
        if self.checked_players:
            suspects = [p for p in self.checked_players if p in self.wolf_ids]
            if suspects:
                return {"speech": f"I suspect Player {suspects[0]} is a werewolf based on their voting pattern."}
        return {"speech": "We need to carefully analyze voting patterns."}
    
    def handle_vote(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Vote for known wolves."""
        candidates = message.get("candidates", [])
        
        # Vote for wolves if they're candidates
        wolf_candidates = [c for c in candidates if c in self.wolf_ids]
        if wolf_candidates:
            return {"vote": wolf_candidates[0]}
        
        # Otherwise random
        if candidates:
            return {"vote": random.choice(candidates)}
        return {}


class RandomAgentMock(MockAgentBase):
    """
    Random agent that makes random decisions.
    Expected metrics:
    - IRS ≈ 0.5 (random guessing baseline)
    - VRS ≈ 0.5 (random votes)
    - MSS ≈ 0.3-0.5 (generic statements)
    - win_rate ≈ 0.4-0.6 (baseline)
    """
    
    def handle_speak(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Generic random speech."""
        templates = [
            "I'm not sure who to suspect.",
            "We should be careful.",
            "Someone seems suspicious.",
            "Let's think about this."
        ]
        return {"speech": random.choice(templates)}
    
    def handle_vote(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Completely random vote."""
        candidates = message.get("candidates", [])
        if candidates:
            return {"vote": random.choice(candidates)}
        return {}


class SaboteurAgentMock(MockAgentBase):
    """
    Saboteur agent that intentionally makes bad decisions.
    Expected metrics:
    - IRS < 0.3 (wrong identification)
    - VRS < 0.2 (counter-productive votes)
    - MSS < 0.3 (poor quality speech)
    - win_rate < 0.3 (loses frequently)
    """
    
    def __init__(self, player_id: int, wolf_ids: list = None):
        super().__init__(player_id)
        self.wolf_ids = wolf_ids or [2, 3]
    
    def handle_speak(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Contradictory and illogical speech."""
        return {"speech": "I trust everyone. No one is suspicious."}
    
    def handle_vote(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Vote for villagers (bad strategy if you're villager)."""
        candidates = message.get("candidates", [])
        
        # If we know wolves, vote for non-wolves (bad!)
        if self.role in ["villager", "seer", "witch", "guard", "hunter"]:
            non_wolves = [c for c in candidates if c not in self.wolf_ids]
            if non_wolves:
                return {"vote": random.choice(non_wolves)}
        
        # Random otherwise
        if candidates:
            return {"vote": random.choice(candidates)}
        return {}
    
    def handle_night_action(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Waste abilities."""
        if self.role == "seer":
            # Check known villagers (wasteful)
            alive_players = message.get("observation", {}).get("alive_players", [])
            non_wolves = [p for p in alive_players if p not in self.wolf_ids and p != self.player_id]
            if non_wolves:
                return {"action_type": "check", "target_id": random.choice(non_wolves)}
        
        return super().handle_night_action(message)


class AggressiveWolfMock(MockAgentBase):
    """
    Aggressive wolf that draws suspicion.
    Expected metrics:
    - deception_score < 0.3 (caught early)
    - survival_rate < 0.4 (eliminated often)
    """
    
    def __init__(self, player_id: int, wolf_teammate_id: int):
        super().__init__(player_id)
        self.wolf_teammate = wolf_teammate_id
    
    def handle_speak(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Aggressive accusatory speech."""
        return {"speech": "I think we should eliminate someone quickly without discussion!"}
    
    def handle_vote(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Vote opposite to majority (draws suspicion)."""
        candidates = message.get("candidates", [])
        current_votes = message.get("current_votes", {})
        
        # Find least voted candidate (against majority)
        if candidates and current_votes:
            vote_counts = {c: sum(1 for v in current_votes.values() if v == c) for c in candidates}
            least_voted = min(vote_counts, key=vote_counts.get)
            return {"vote": least_voted}
        
        if candidates:
            return {"vote": random.choice(candidates)}
        return {}
