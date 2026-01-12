"""
Smart Baseline Agent for Werewolf Benchmark.

A strong rule-based agent using:
- Bayesian probability tracking
- Voting pattern analysis
- Claim consistency tracking
- Role-specific optimal strategies

Designed to be a challenging opponent for LLM agents.
"""

import random
import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class RoleType(str, Enum):
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    GUARD = "guard"
    VILLAGER = "villager"


class Camp(str, Enum):
    GOOD = "good"
    WOLF = "wolf"


@dataclass
class PlayerBelief:
    """Beliefs about a single player."""
    wolf_probability: float = 0.3  # Prior probability
    claims: List[str] = field(default_factory=list)  # Role claims
    accusations: List[int] = field(default_factory=list)  # Who they accused
    defenses: List[int] = field(default_factory=list)  # Who they defended
    votes: List[int] = field(default_factory=list)  # Vote history
    was_accused_by: List[int] = field(default_factory=list)
    was_defended_by: List[int] = field(default_factory=list)
    speech_count: int = 0
    suspicious_behaviors: int = 0
    helpful_behaviors: int = 0


class SmartBaselineAgent:
    """
    Strong baseline agent with advanced game theory strategies.
    """
    
    def __init__(self, agent_id: str = "smart-baseline"):
        self.agent_id = agent_id
        self.reset()
    
    def reset(self):
        """Reset state for new game."""
        self.player_id: Optional[int] = None
        self.role: Optional[RoleType] = None
        self.camp: Optional[Camp] = None
        self.game_id: Optional[str] = None
        
        # Game state
        self.alive_players: Set[int] = set()
        self.dead_players: Set[int] = set()
        self.round_number: int = 0
        
        # Knowledge (for special roles)
        self.wolf_team: List[int] = []  # Known if werewolf
        self.seer_results: Dict[int, bool] = {}  # player_id -> is_wolf
        self.last_protected: Optional[int] = None  # For guard
        self.witch_heal_used: bool = False
        self.witch_poison_used: bool = False
        
        # Beliefs about other players
        self.beliefs: Dict[int, PlayerBelief] = {}
        
        # History tracking
        self.voting_history: List[Dict[int, int]] = []  # round -> {voter: target}
        self.elimination_history: List[Tuple[int, str, str]] = []  # (player_id, role, phase)
        self.speech_history: List[Dict] = []
        self.night_kills: List[int] = []
        
        # Derived analysis
        self.voting_similarity: Dict[Tuple[int, int], int] = defaultdict(int)
        self.role_claimants: Dict[str, List[int]] = defaultdict(list)
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def handle_game_start(self, message: Dict) -> Dict:
        """Initialize agent state."""
        self.reset()
        
        self.game_id = message.get("game_id")
        self.player_id = message.get("player_id")
        self.role = RoleType(message.get("role", "villager"))
        self.camp = Camp(message.get("camp", "good"))
        self.alive_players = set(message.get("alive_players", []))
        
        # Wolves know each other
        if "werewolf_team" in message:
            self.wolf_team = message["werewolf_team"]
        
        # Initialize beliefs about all players
        num_players = len(self.alive_players)
        num_wolves = len(self.wolf_team) if self.wolf_team else max(2, num_players // 4)
        prior_wolf_prob = num_wolves / num_players
        
        for pid in self.alive_players:
            if pid == self.player_id:
                continue
            if pid in self.wolf_team:
                # I know they're wolves
                self.beliefs[pid] = PlayerBelief(wolf_probability=1.0)
            else:
                self.beliefs[pid] = PlayerBelief(wolf_probability=prior_wolf_prob)
        
        logger.info(f"SmartBaseline {self.player_id}: Role={self.role.value}, Camp={self.camp.value}")
        
        return {"status": "initialized"}
    
    # =========================================================================
    # BAYESIAN UPDATES
    # =========================================================================
    
    def update_on_elimination(self, player_id: int, role: str, phase: str):
        """Update beliefs when someone is eliminated."""
        if player_id in self.alive_players:
            self.alive_players.remove(player_id)
        self.dead_players.add(player_id)
        
        was_wolf = role == "werewolf"
        self.elimination_history.append((player_id, role, phase))
        
        if player_id in self.beliefs:
            del self.beliefs[player_id]
        
        # Update beliefs based on who accused/defended the eliminated
        for pid, belief in self.beliefs.items():
            if player_id in belief.accusations:
                # They accused this player
                if was_wolf:
                    # Good accusation! Less suspicious
                    belief.wolf_probability *= 0.7
                    belief.helpful_behaviors += 1
                else:
                    # Bad accusation - slightly more suspicious
                    belief.wolf_probability *= 1.1
                    belief.suspicious_behaviors += 1
            
            if player_id in belief.defenses:
                # They defended this player
                if was_wolf:
                    # Defended a wolf! More suspicious
                    belief.wolf_probability *= 1.4
                    belief.suspicious_behaviors += 2
                else:
                    # Defended innocent - slightly less suspicious
                    belief.wolf_probability *= 0.95
        
        # Normalize probabilities
        self._normalize_probabilities()
    
    def update_on_vote(self, voter_id: int, target_id: int, round_num: int):
        """Update beliefs based on voting."""
        if voter_id == self.player_id or voter_id not in self.beliefs:
            return
        
        self.beliefs[voter_id].votes.append(target_id)
        
        # Track voting similarity between players
        for other_id, other_belief in self.beliefs.items():
            if other_id == voter_id:
                continue
            if other_belief.votes and other_belief.votes[-1] == target_id:
                # Both voted for same target this round
                pair = tuple(sorted([voter_id, other_id]))
                self.voting_similarity[pair] += 1
                
                # High similarity = potential wolf team
                if self.voting_similarity[pair] >= 3:
                    self.beliefs[voter_id].wolf_probability *= 1.1
                    if other_id in self.beliefs:
                        self.beliefs[other_id].wolf_probability *= 1.1
    
    def update_on_speech(self, player_id: int, speech: str, round_num: int):
        """Analyze speech for information."""
        if player_id == self.player_id or player_id not in self.beliefs:
            return
        
        belief = self.beliefs[player_id]
        belief.speech_count += 1
        speech_lower = speech.lower()
        
        # Track role claims
        role_patterns = {
            'seer': [r'\bi am (?:the )?seer\b', r'\bi\'m (?:the )?seer\b', r'\bseer here\b'],
            'witch': [r'\bi am (?:the )?witch\b', r'\bi\'m (?:the )?witch\b'],
            'hunter': [r'\bi am (?:the )?hunter\b', r'\bi\'m (?:the )?hunter\b'],
            'guard': [r'\bi am (?:the )?guard\b', r'\bi\'m (?:the )?guard\b'],
            'villager': [r'\bi am (?:a )?villager\b', r'\bjust a villager\b'],
        }
        
        for role, patterns in role_patterns.items():
            for pattern in patterns:
                if re.search(pattern, speech_lower):
                    belief.claims.append(role)
                    self.role_claimants[role].append(player_id)
                    
                    # Multiple seer claims = someone lying
                    if role == 'seer' and len(self.role_claimants['seer']) > 1:
                        for claimant in self.role_claimants['seer']:
                            if claimant in self.beliefs:
                                self.beliefs[claimant].wolf_probability *= 1.3
                                self.beliefs[claimant].suspicious_behaviors += 1
        
        # Track accusations
        accusation_patterns = [
            r'player\s*(\d+)\s*is\s*(?:a\s*)?(?:wolf|werewolf|suspicious)',
            r'suspect\s*player\s*(\d+)',
            r'vote\s*(?:for\s*)?player\s*(\d+)',
            r'player\s*(\d+)\s*(?:seems?|looks?)\s*suspicious',
        ]
        
        for pattern in accusation_patterns:
            matches = re.findall(pattern, speech_lower)
            for match in matches:
                try:
                    target = int(match)
                    if target != player_id and target in self.alive_players:
                        belief.accusations.append(target)
                        if target in self.beliefs:
                            self.beliefs[target].was_accused_by.append(player_id)
                except ValueError:
                    pass
        
        # Track defenses
        defense_patterns = [
            r'player\s*(\d+)\s*is\s*(?:not\s*(?:a\s*)?wolf|innocent|good|trustworthy)',
            r'trust\s*player\s*(\d+)',
            r'player\s*(\d+)\s*is\s*(?:definitely\s*)?(?:on\s*)?(?:our|the\s*good)\s*side',
        ]
        
        for pattern in defense_patterns:
            matches = re.findall(pattern, speech_lower)
            for match in matches:
                try:
                    target = int(match)
                    if target != player_id and target in self.alive_players:
                        belief.defenses.append(target)
                        if target in self.beliefs:
                            self.beliefs[target].was_defended_by.append(player_id)
                except ValueError:
                    pass
        
        # Seer check claims
        check_pattern = r'(?:i\s*)?checked\s*player\s*(\d+).*(?:is|was|are)\s*(?:a\s*)?(\w+)'
        check_matches = re.findall(check_pattern, speech_lower)
        for target_str, result in check_matches:
            try:
                target = int(target_str)
                is_wolf_claim = 'wolf' in result.lower()
                
                # If claimed seer check contradicts known info, liar!
                if target in self.seer_results:
                    if is_wolf_claim != self.seer_results[target]:
                        belief.wolf_probability *= 1.5
                        belief.suspicious_behaviors += 2
            except ValueError:
                pass
        
        self.speech_history.append({
            'player_id': player_id,
            'round': round_num,
            'speech': speech
        })
    
    def update_on_night_kill(self, victim_id: int):
        """Update beliefs based on night kill target."""
        self.night_kills.append(victim_id)
        
        if victim_id in self.beliefs:
            del self.beliefs[victim_id]
        
        if victim_id in self.alive_players:
            self.alive_players.remove(victim_id)
        self.dead_players.add(victim_id)
        
        # Wolves don't kill wolves (usually)
        # Anyone who strongly defended victim might be good
        for pid, belief in self.beliefs.items():
            if victim_id in belief.defenses:
                # Defended someone who got killed = probably good
                belief.wolf_probability *= 0.85
    
    def _normalize_probabilities(self):
        """Keep probabilities in valid range."""
        for belief in self.beliefs.values():
            belief.wolf_probability = max(0.05, min(0.95, belief.wolf_probability))
    
    # =========================================================================
    # DECISION MAKING
    # =========================================================================
    
    def get_most_suspicious(self, exclude: Set[int] = None) -> Optional[int]:
        """Get the most suspicious player based on beliefs."""
        exclude = exclude or set()
        candidates = {
            pid: belief.wolf_probability 
            for pid, belief in self.beliefs.items()
            if pid in self.alive_players and pid not in exclude
        }
        
        if not candidates:
            return None
        
        return max(candidates, key=candidates.get)
    
    def get_least_suspicious(self, exclude: Set[int] = None) -> Optional[int]:
        """Get the least suspicious player (likely god role)."""
        exclude = exclude or set()
        candidates = {
            pid: belief.wolf_probability 
            for pid, belief in self.beliefs.items()
            if pid in self.alive_players and pid not in exclude
        }
        
        if not candidates:
            return None
        
        return min(candidates, key=candidates.get)
    
    # =========================================================================
    # ROLE-SPECIFIC STRATEGIES
    # =========================================================================
    
    def _villager_vote_strategy(self) -> int:
        """Villager votes for most suspicious."""
        target = self.get_most_suspicious()
        return target if target else random.choice(list(self.alive_players - {self.player_id}))
    
    def _werewolf_vote_strategy(self, current_votes: Dict[int, int] = None) -> int:
        """
        Werewolf voting strategy:
        - Don't vote together with other wolves (avoid detection)
        - Vote for most suspicious good player
        - If under pressure, might bus a fellow wolf
        """
        current_votes = current_votes or {}
        
        # Get targets other wolves are voting for
        wolf_targets = set()
        for wolf_id in self.wolf_team:
            if wolf_id != self.player_id and wolf_id in current_votes:
                wolf_targets.add(current_votes[wolf_id])
        
        # Non-wolf candidates
        candidates = [
            pid for pid in self.alive_players 
            if pid != self.player_id and pid not in self.wolf_team
        ]
        
        if not candidates:
            # All remaining are wolves - vote for someone to avoid suspicion
            candidates = [pid for pid in self.alive_players if pid != self.player_id]
        
        # Prefer targets other wolves aren't voting for (split votes = less suspicious)
        preferred = [c for c in candidates if c not in wolf_targets]
        if preferred:
            candidates = preferred
        
        # Among candidates, pick least suspicious (to blend in)
        # Or most accused by others (to follow crowd)
        candidate_scores = {}
        for pid in candidates:
            if pid in self.beliefs:
                # Lower wolf_probability = better target (looks less like coordinated attack)
                score = 1 - self.beliefs[pid].wolf_probability
                # Bonus if others are accusing them
                score += len(self.beliefs[pid].was_accused_by) * 0.2
                candidate_scores[pid] = score
            else:
                candidate_scores[pid] = 0.5
        
        if candidate_scores:
            return max(candidate_scores, key=candidate_scores.get)
        
        return random.choice(candidates) if candidates else self.player_id
    
    def _werewolf_kill_strategy(self) -> int:
        """
        Choose kill target:
        - Prioritize suspected seer/witch/hunter
        - Kill influential players
        - Kill those who accuse wolves most
        """
        candidates = [
            pid for pid in self.alive_players 
            if pid not in self.wolf_team and pid != self.player_id
        ]
        
        if not candidates:
            return None
        
        threat_scores = {}
        for pid in candidates:
            score = 0
            belief = self.beliefs.get(pid, PlayerBelief())
            
            # Suspected god roles (low wolf probability but active)
            if belief.wolf_probability < 0.3 and belief.speech_count > 1:
                score += 50  # Likely god role
            
            # Seer claimants are HIGH priority
            if pid in self.role_claimants.get('seer', []):
                score += 100
            
            # Those who accuse wolves
            accusations_against_wolves = sum(
                1 for acc in belief.accusations if acc in self.wolf_team
            )
            score += accusations_against_wolves * 30
            
            # Influential players (many follow their votes)
            score += belief.helpful_behaviors * 10
            
            # Low suspicion = dangerous (trusted by village)
            score += (1 - belief.wolf_probability) * 20
            
            threat_scores[pid] = score
        
        return max(threat_scores, key=threat_scores.get)
    
    def _seer_check_strategy(self) -> int:
        """Choose who to check: most suspicious unchecked player."""
        candidates = [
            pid for pid in self.alive_players 
            if pid != self.player_id and pid not in self.seer_results
        ]
        
        if not candidates:
            return None
        
        # Check most suspicious first
        return max(
            candidates, 
            key=lambda p: self.beliefs.get(p, PlayerBelief()).wolf_probability
        )
    
    def _guard_protect_strategy(self) -> int:
        """
        Choose who to protect:
        - Suspected seer/important roles
        - Low suspicion players (probably good)
        - Not self, not last protected
        """
        candidates = [
            pid for pid in self.alive_players 
            if pid != self.player_id and pid != self.last_protected
        ]
        
        if not candidates:
            return None
        
        value_scores = {}
        for pid in candidates:
            score = 0
            belief = self.beliefs.get(pid, PlayerBelief())
            
            # Seer claimants are valuable
            if pid in self.role_claimants.get('seer', []):
                score += 100
            
            # Low wolf probability = good player = protect
            score += (1 - belief.wolf_probability) * 50
            
            # Active helpful players
            score += belief.helpful_behaviors * 20
            
            value_scores[pid] = score
        
        return max(value_scores, key=value_scores.get)
    
    def _witch_strategy(self, victim_id: Optional[int], heal_available: bool, poison_available: bool) -> Dict:
        """
        Witch decision:
        - Heal: save victim if they're likely good
        - Poison: kill most suspicious player
        """
        action = None
        
        # Heal decision
        if heal_available and victim_id and victim_id != self.player_id:
            victim_belief = self.beliefs.get(victim_id, PlayerBelief())
            
            # Save if low wolf probability
            if victim_belief.wolf_probability < 0.5:
                # Check if victim is valuable (seer claimant, etc.)
                is_valuable = (
                    victim_id in self.role_claimants.get('seer', []) or
                    victim_belief.helpful_behaviors >= 2
                )
                
                if is_valuable or victim_belief.wolf_probability < 0.3:
                    action = {
                        "action_type": "heal",
                        "player_id": self.player_id,
                        "target_id": victim_id
                    }
                    return {"action": action}
        
        # Poison decision
        if poison_available:
            # Poison if very confident someone is wolf
            most_sus = self.get_most_suspicious()
            if most_sus:
                belief = self.beliefs.get(most_sus, PlayerBelief())
                # Only poison if very confident
                if belief.wolf_probability > 0.75:
                    action = {
                        "action_type": "poison",
                        "player_id": self.player_id,
                        "target_id": most_sus
                    }
                    return {"action": action}
        
        return {"action": None}
    
    def _hunter_shoot_strategy(self, available_targets: List[int]) -> int:
        """Hunter shoots most suspicious when dying."""
        if not available_targets:
            return None
        
        candidates = [t for t in available_targets if t in self.beliefs]
        if not candidates:
            return random.choice(available_targets)
        
        return max(candidates, key=lambda p: self.beliefs[p].wolf_probability)
    
    # =========================================================================
    # SPEECH GENERATION
    # =========================================================================
    
    def _generate_speech(self) -> str:
        """Generate speech based on role and game state."""
        
        if self.role == RoleType.WEREWOLF:
            return self._wolf_speech()
        elif self.role == RoleType.SEER:
            return self._seer_speech()
        else:
            return self._villager_speech()
    
    def _villager_speech(self) -> str:
        """Generic good player speech."""
        most_sus = self.get_most_suspicious()
        
        if not most_sus:
            templates = [
                "We need to analyze voting patterns carefully.",
                "Let's not rush to judgment. Who has been acting suspicious?",
                "I'm watching everyone's behavior closely.",
            ]
            return random.choice(templates)
        
        belief = self.beliefs.get(most_sus, PlayerBelief())
        confidence = "highly" if belief.wolf_probability > 0.7 else "somewhat"
        
        reasons = []
        if belief.suspicious_behaviors > 0:
            reasons.append("inconsistent behavior")
        if len(belief.defenses) > len(belief.accusations):
            reasons.append("defending suspicious players")
        if not reasons:
            reasons.append("overall behavior pattern")
        
        templates = [
            f"I'm {confidence} suspicious of Player {most_sus}. Reason: {reasons[0]}.",
            f"Player {most_sus} seems suspicious to me. I suggest we vote for them.",
            f"Based on my analysis, Player {most_sus} might be a werewolf. {reasons[0].capitalize()}.",
            f"I think we should focus on Player {most_sus}. Their {reasons[0]} is concerning.",
        ]
        
        return random.choice(templates)
    
    def _wolf_speech(self) -> str:
        """Wolf tries to appear innocent and deflect."""
        # Find a non-wolf to accuse
        candidates = [
            pid for pid in self.alive_players 
            if pid not in self.wolf_team and pid != self.player_id
        ]
        
        if not candidates:
            return "I'm just a villager trying to help find the wolves."
        
        # Accuse someone who is already somewhat suspicious
        target = None
        for pid in candidates:
            belief = self.beliefs.get(pid, PlayerBelief())
            if belief.was_accused_by:  # Others already accusing them
                target = pid
                break
        
        if not target:
            # Pick someone with moderate suspicion
            target = random.choice(candidates)
        
        templates = [
            f"I've noticed Player {target} has been acting strange. We should watch them.",
            f"Player {target}'s voting pattern seems suspicious to me.",
            f"Has anyone else noticed Player {target}? Something feels off about them.",
            f"I'm a villager and I think Player {target} might be a wolf.",
            f"We should consider voting for Player {target}. Their behavior is suspicious.",
        ]
        
        return random.choice(templates)
    
    def _seer_speech(self) -> str:
        """Seer decides whether to reveal information."""
        wolves_found = [pid for pid, is_wolf in self.seer_results.items() if is_wolf]
        innocents_found = [pid for pid, is_wolf in self.seer_results.items() if not is_wolf]
        
        # Reveal conditions:
        # 1. Found wolf AND (late game OR under pressure)
        # 2. About to die
        # 3. Critical game state
        
        alive_count = len(self.alive_players)
        should_reveal = False
        
        if wolves_found:
            # Check if any found wolf is still alive
            alive_wolves = [w for w in wolves_found if w in self.alive_players]
            
            if alive_wolves:
                # Late game - reveal
                if alive_count <= 5:
                    should_reveal = True
                
                # Wolf about to escape - reveal
                if self.beliefs.get(alive_wolves[0], PlayerBelief()).wolf_probability < 0.4:
                    should_reveal = True
        
        if should_reveal and wolves_found:
            alive_wolf = next((w for w in wolves_found if w in self.alive_players), None)
            if alive_wolf:
                return f"I am the Seer! I checked Player {alive_wolf} and they are a WEREWOLF! Vote them out!"
        
        # If found innocents, might softly defend
        if innocents_found:
            alive_innocent = next((i for i in innocents_found if i in self.alive_players), None)
            if alive_innocent and random.random() < 0.3:
                return f"I have information that Player {alive_innocent} is trustworthy. We should focus elsewhere."
        
        # Default to normal villager speech
        return self._villager_speech()
    
    # =========================================================================
    # MESSAGE HANDLERS
    # =========================================================================
    
    def handle_night_action(self, message: Dict) -> Dict:
        """Handle night action based on role."""
        observation = message.get("observation", {})
        role_info = observation.get("role_specific_info", {})
        
        self.round_number = observation.get("round_number", self.round_number)
        self.alive_players = set(observation.get("alive_players", self.alive_players))
        
        action = None
        
        if self.role == RoleType.WEREWOLF:
            target = self._werewolf_kill_strategy()
            if target:
                action = {
                    "action_type": "kill",
                    "player_id": self.player_id,
                    "target_id": target
                }
        
        elif self.role == RoleType.SEER:
            # Process previous check result if available
            if "check_result" in role_info:
                checked_player = role_info.get("checked_player")
                is_wolf = role_info.get("check_result")
                if checked_player is not None:
                    self.seer_results[checked_player] = is_wolf
                    if checked_player in self.beliefs:
                        self.beliefs[checked_player].wolf_probability = 1.0 if is_wolf else 0.05
            
            target = self._seer_check_strategy()
            if target:
                action = {
                    "action_type": "check",
                    "player_id": self.player_id,
                    "target_id": target
                }
        
        elif self.role == RoleType.GUARD:
            target = self._guard_protect_strategy()
            if target:
                action = {
                    "action_type": "protect",
                    "player_id": self.player_id,
                    "target_id": target
                }
                self.last_protected = target
        
        elif self.role == RoleType.WITCH:
            victim = role_info.get("werewolf_victim")
            heal_available = role_info.get("heal_available", not self.witch_heal_used)
            poison_available = role_info.get("poison_available", not self.witch_poison_used)
            
            result = self._witch_strategy(victim, heal_available, poison_available)
            if result.get("action"):
                action = result["action"]
                if action["action_type"] == "heal":
                    self.witch_heal_used = True
                elif action["action_type"] == "poison":
                    self.witch_poison_used = True
        
        return {"action": action}
    
    def handle_speak(self, message: Dict) -> Dict:
        """Generate speech."""
        observation = message.get("observation", {})
        self.round_number = observation.get("round_number", self.round_number)
        self.alive_players = set(observation.get("alive_players", self.alive_players))
        
        # Process any new information from observation
        for event in observation.get("recent_events", []):
            self._process_event(event)
        
        speech = self._generate_speech()
        return {"speech": speech}
    
    def handle_vote(self, message: Dict) -> Dict:
        """Cast vote."""
        candidates = message.get("candidates", [])
        observation = message.get("observation", {})
        current_votes = message.get("current_votes", {})
        
        self.round_number = observation.get("round_number", self.round_number)
        
        if not candidates:
            return {"vote": self.player_id}
        
        # Process current votes to update beliefs
        for voter, target in current_votes.items():
            try:
                self.update_on_vote(int(voter), int(target), self.round_number)
            except (ValueError, TypeError):
                pass
        
        # Choose vote based on role
        if self.role == RoleType.WEREWOLF:
            vote = self._werewolf_vote_strategy(current_votes)
        else:
            vote = self._villager_vote_strategy()
        
        # Ensure vote is valid
        if vote not in candidates:
            vote = random.choice(candidates)
        
        return {"vote": vote}
    
    def handle_sheriff_election(self, message: Dict) -> Dict:
        """Vote for sheriff."""
        candidates = message.get("candidates", [])
        
        if not candidates:
            return {"vote": self.player_id}
        
        # Vote for least suspicious (probably good and capable)
        vote = self.get_least_suspicious(exclude=set(self.wolf_team))
        
        if vote not in candidates:
            vote = random.choice(candidates)
        
        return {"vote": vote}
    
    def handle_sheriff_summary(self, message: Dict) -> Dict:
        """Sheriff makes recommendation."""
        votes = message.get("votes", {})
        
        # Count votes
        vote_counts = defaultdict(int)
        for voter, target in votes.items():
            vote_counts[target] += 1
        
        # Recommend most voted if suspicious, else most suspicious overall
        if vote_counts:
            most_voted = max(vote_counts, key=vote_counts.get)
            belief = self.beliefs.get(most_voted, PlayerBelief())
            
            if belief.wolf_probability > 0.4:
                return {"recommendation": most_voted}
        
        # Fall back to most suspicious
        most_sus = self.get_most_suspicious()
        return {"recommendation": most_sus or self.player_id}
    
    def handle_hunter_shoot(self, message: Dict) -> Dict:
        """Hunter shoots when dying."""
        targets = message.get("targets", [])
        
        if not targets:
            return {"target": self.player_id}
        
        target = self._hunter_shoot_strategy(targets)
        return {"target": target if target else random.choice(targets)}
    
    def _process_event(self, event: Dict):
        """Process game event to update beliefs."""
        event_type = event.get("type")
        
        if event_type == "speech":
            self.update_on_speech(
                event.get("player_id"),
                event.get("speech", event.get("text", "")),
                event.get("round", self.round_number)
            )
        
        elif event_type == "vote":
            self.update_on_vote(
                event.get("voter_id"),
                event.get("target_id"),
                event.get("round", self.round_number)
            )
        
        elif event_type == "elimination":
            self.update_on_elimination(
                event.get("player_id"),
                event.get("role", "unknown"),
                event.get("phase", "day")
            )
        
        elif event_type == "night_kill":
            self.update_on_night_kill(event.get("victim_id"))