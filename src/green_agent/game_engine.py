"""
Game engine for Werewolf - handles game flow, rules, and state management.
"""
import uuid
import random
from typing import Dict, List, Optional, Tuple
from .models import (
    GameState, PlayerInfo, Action, ActionType, Phase, Camp,
    RoleType, NightActions, DayActions, Observation
)
from .roles import create_role, get_camp_for_role


class WerewolfGameEngine:
    """
    Manages the complete Werewolf game flow following official rules.
    """
    
    def __init__(
        self, 
        num_players: int = 8,
        enable_sheriff: bool = True,
        config: Optional[Dict] = None
    ):
        """
        Initialize game engine with configuration.
        
        Args:
            num_players: Total number of players (8 or 12)
            enable_sheriff: Whether to enable sheriff election
            config: Additional game configuration
        """
        self.num_players = num_players
        self.enable_sheriff = enable_sheriff
        self.config = config or {}
        
        # Role distribution from config
        if 'roles' in self.config:
            # Convert role names to RoleType enums
            self.role_distribution = {}
            for role_name, count in self.config['roles'].items():
                try:
                    role_type = RoleType(role_name)
                    self.role_distribution[role_type] = count
                except ValueError:
                    logger.warning(f"Unknown role: {role_name}")
            
            # Validate total matches num_players
            total = sum(self.role_distribution.values())
            if total != num_players:
                raise ValueError(f"Role count ({total}) doesn't match num_players ({num_players})")
        else:
            # Fallback to defaults
            if num_players == 8:
                self.role_distribution = {
                    RoleType.WEREWOLF: 2,
                    RoleType.VILLAGER: 3,
                    RoleType.SEER: 1,
                    RoleType.WITCH: 1,
                    RoleType.GUARD: 1,
                }
            else:  # 12 players
                self.role_distribution = {
                    RoleType.WEREWOLF: 4,
                    RoleType.VILLAGER: 4,
                    RoleType.SEER: 1,
                    RoleType.WITCH: 1,
                    RoleType.HUNTER: 1,
                    RoleType.GUARD: 1,
                }
    
    def initialize_game(self, player_endpoints: Dict[int, str]) -> GameState:
        """
        Initialize a new game with role assignments.
        
        Args:
            player_endpoints: Mapping of player_id to A2A endpoint
            
        Returns:
            Initial game state
        """
        game_id = str(uuid.uuid4())
        
        # Assign roles randomly
        roles = []
        for role_type, count in self.role_distribution.items():
            roles.extend([role_type] * count)
        
        random.shuffle(roles)
        
        # Create player info
        players = {}
        for player_id in range(1, self.num_players + 1):
            role = roles[player_id - 1]
            camp = get_camp_for_role(role)
            
            players[player_id] = PlayerInfo(
                player_id=player_id,
                role=role,
                camp=camp,
                is_alive=True,
                is_sheriff=False,
                endpoint=player_endpoints[player_id]
            )
        
        game_state = GameState(
            game_id=game_id,
            round_number=0,
            phase=Phase.NIGHT,
            players=players,
            alive_players=list(range(1, self.num_players + 1)),
            eliminated_players=[],
            sheriff_id=None,
            game_history=[],
            witch_heal_used=False,
            witch_poison_used=False,
            last_protected=None,
            winner=None
        )
        
        return game_state
    
    def process_night_phase(
        self, 
        game_state: GameState,
        actions: Dict[int, Action]
    ) -> Tuple[GameState, List[int]]:
        """
        Process all night actions and determine who dies.
        
        Args:
            game_state: Current game state
            actions: Actions taken by players {player_id: action}
            
        Returns:
            Tuple of (updated game state, list of eliminated player IDs)
        """
        night_actions = NightActions()
        eliminated = []
        
        # 1. Werewolves choose victim
        werewolf_votes = {}
        for player_id, action in actions.items():
            player = game_state.players[player_id]
            if player.role == RoleType.WEREWOLF and action.action_type == ActionType.KILL:
                werewolf_votes[action.target_id] = werewolf_votes.get(action.target_id, 0) + 1
        
        if werewolf_votes:
            night_actions.werewolf_target = max(werewolf_votes, key=werewolf_votes.get)
        
        # 2. Guard protects someone
        for player_id, action in actions.items():
            player = game_state.players[player_id]
            if player.role == RoleType.GUARD and action.action_type == ActionType.PROTECT:
                # Validate cannot protect same person twice in a row
                if action.target_id != game_state.last_protected:
                    night_actions.guard_target = action.target_id
                    night_actions.guard_id = player_id  # Track who made the protection
                    game_state.last_protected = action.target_id
        
        # 3. Seer checks identity
        for player_id, action in actions.items():
            player = game_state.players[player_id]
            if player.role == RoleType.SEER and action.action_type == ActionType.CHECK:
                night_actions.seer_target = action.target_id
                
                # Log the check result
                target_player = game_state.players[action.target_id]
                game_state.game_history.append({
                    'type': 'seer_check',
                    'round': game_state.round_number,
                    'seer_id': player_id,
                    'target_id': action.target_id,
                    'result': 'werewolf' if target_player.camp == Camp.WOLF else 'good'
                })
        
        # 4. Determine if werewolf kill is successful
        victim_id = night_actions.werewolf_target
        if victim_id and victim_id != night_actions.guard_target:
            eliminated.append(victim_id)
        elif victim_id == night_actions.guard_target:
            # Protection was successful
            game_state.game_history.append({
                'type': 'protection_saved',
                'round': game_state.round_number,
                'player_id': victim_id,
                'guard_id': night_actions.guard_id  # Track who made the save
            })
        
        # 5. Witch actions (if role exists in game)
        witch_id = None
        for player_id, player in game_state.players.items():
            if player.role == RoleType.WITCH and player.is_alive:
                witch_id = player_id
                break
        
        if witch_id and witch_id in actions:
            action = actions[witch_id]
            
            # Heal potion
            if action.action_type == ActionType.HEAL and not game_state.witch_heal_used:
                if victim_id in eliminated:
                    eliminated.remove(victim_id)
                    game_state.witch_heal_used = True
                    night_actions.witch_heal = True
                    
                    game_state.game_history.append({
                        'type': 'witch_heal',
                        'round': game_state.round_number,
                        'saved_player': victim_id
                    })
            
            # Poison potion
            elif action.action_type == ActionType.POISON and not game_state.witch_poison_used:
                if action.target_id and action.target_id not in eliminated:
                    eliminated.append(action.target_id)
                    game_state.witch_poison_used = True
                    night_actions.witch_poison_target = action.target_id
                    
                    game_state.game_history.append({
                        'type': 'witch_poison',
                        'round': game_state.round_number,
                        'poisoned_player': action.target_id
                    })
        
        # Update game state
        for player_id in eliminated:
            game_state.players[player_id].is_alive = False
            game_state.alive_players.remove(player_id)
            game_state.eliminated_players.append(player_id)
            
            game_state.game_history.append({
                'type': 'elimination',
                'phase': 'night',
                'round': game_state.round_number,
                'player_id': player_id
            })
        
        return game_state, eliminated
    
    def process_day_phase(
        self,
        game_state: GameState,
        speeches: Dict[int, str],
        votes_round1: Dict[int, int],
        sheriff_recommendation: Optional[int],
        votes_round2: Dict[int, int]
    ) -> Tuple[GameState, Optional[int]]:
        """
        Process day phase: speeches, voting, and elimination.
        
        Args:
            game_state: Current game state
            speeches: Player speeches {player_id: speech}
            votes_round1: First voting round {voter_id: target_id}
            sheriff_recommendation: Sheriff's recommended exile target
            votes_round2: Second voting round after sheriff summary
            
        Returns:
            Tuple of (updated game state, eliminated player ID)
        """
        # Note: Speeches are logged in werewolf_judge.py with more detail (phase, reasoning, etc)
        
        # Process first voting round
        vote_counts_1 = {}
        for voter_id, target_id in votes_round1.items():
            weight = 1.5 if game_state.players[voter_id].is_sheriff else 1.0
            vote_counts_1[target_id] = vote_counts_1.get(target_id, 0) + weight
            
            game_state.game_history.append({
                'type': 'vote',
                'round': game_state.round_number,
                'voting_round': 1,
                'voter_id': voter_id,
                'target_id': target_id,
                'weight': weight,
                'is_final': False  # First round is not final
            })
        
        # Sheriff recommendation
        if game_state.sheriff_id and sheriff_recommendation:
            game_state.game_history.append({
                'type': 'sheriff_recommendation',
                'round': game_state.round_number,
                'sheriff_id': game_state.sheriff_id,
                'recommended_target': sheriff_recommendation,
                'previous_votes': votes_round1
            })
        
        # Process second voting round
        vote_counts_2 = {}
        for voter_id, target_id in votes_round2.items():
            weight = 1.5 if game_state.players[voter_id].is_sheriff else 1.0
            vote_counts_2[target_id] = vote_counts_2.get(target_id, 0) + weight
            
            game_state.game_history.append({
                'type': 'vote',
                'round': game_state.round_number,
                'voting_round': 2,
                'voter_id': voter_id,
                'target_id': target_id,
                'weight': weight,
                'is_final': True  # Second round is final
            })
        
        # Determine who is eliminated
        eliminated_id = None
        if vote_counts_2:
            max_votes = max(vote_counts_2.values())
            candidates = [pid for pid, votes in vote_counts_2.items() if votes == max_votes]
            
            if len(candidates) == 1:
                eliminated_id = candidates[0]
            else:
                # Tie - random selection after debate
                eliminated_id = random.choice(candidates)
        
        # Update vote events with elimination result
        if eliminated_id:
            for event in game_state.game_history:
                if (event.get('type') == 'vote' and 
                    event.get('round') == game_state.round_number and
                    event.get('voting_round') == 2):
                    event['eliminated'] = event.get('target_id') == eliminated_id
        
        # Update game state if someone is eliminated
        if eliminated_id:
            game_state.players[eliminated_id].is_alive = False
            game_state.alive_players.remove(eliminated_id)
            game_state.eliminated_players.append(eliminated_id)
            
            game_state.game_history.append({
                'type': 'elimination',
                'phase': 'day',
                'round': game_state.round_number,
                'player_id': eliminated_id,
                'vote_count': vote_counts_2.get(eliminated_id, 0)
            })
        
        return game_state, eliminated_id
    
    def handle_hunter_death(
        self,
        game_state: GameState,
        hunter_id: int,
        shot_target: int
    ) -> GameState:
        """Handle hunter shooting when eliminated"""
        if shot_target in game_state.alive_players:
            game_state.players[shot_target].is_alive = False
            game_state.alive_players.remove(shot_target)
            game_state.eliminated_players.append(shot_target)
            
            game_state.game_history.append({
                'type': 'hunter_shot',
                'round': game_state.round_number,
                'hunter_id': hunter_id,
                'shot_target': shot_target
            })
        
        return game_state
    
    def elect_sheriff(
        self,
        game_state: GameState,
        votes: Dict[int, int]
    ) -> GameState:
        """
        Conduct sheriff election.
        
        Args:
            game_state: Current game state
            votes: Election votes {voter_id: candidate_id}
            
        Returns:
            Updated game state with sheriff elected
        """
        vote_counts = {}
        for voter_id, candidate_id in votes.items():
            vote_counts[candidate_id] = vote_counts.get(candidate_id, 0) + 1
        
        if vote_counts:
            max_votes = max(vote_counts.values())
            candidates = [pid for pid, votes in vote_counts.items() if votes == max_votes]
            
            # Random selection if tie
            sheriff_id = random.choice(candidates)
            
            game_state.players[sheriff_id].is_sheriff = True
            game_state.sheriff_id = sheriff_id
            
            game_state.game_history.append({
                'type': 'sheriff_election',
                'round': game_state.round_number,
                'sheriff_id': sheriff_id,
                'votes': votes
            })
        
        return game_state
    
    def check_victory_condition(self, game_state: GameState) -> Optional[Camp]:
        """
        Check if any camp has won the game.
        
        Victory conditions:
        - Good camp: All werewolves eliminated
        - Wolf camp: Werewolves >= good camp members
        
        Returns:
            Winning camp or None if game continues
        """
        alive_werewolves = sum(
            1 for pid in game_state.alive_players
            if game_state.players[pid].camp == Camp.WOLF
        )
        
        alive_good = sum(
            1 for pid in game_state.alive_players
            if game_state.players[pid].camp == Camp.GOOD
        )
        
        if alive_werewolves == 0:
            return Camp.GOOD
        elif alive_werewolves >= alive_good:
            return Camp.WOLF
        
        return None
    
    def create_observation_for_player(
        self,
        player_id: int,
        game_state: GameState,
        phase_specific_info: Optional[Dict] = None
    ) -> Observation:
        """
        Create observation for a specific player based on their role and game state.
        
        Args:
            player_id: Player to create observation for
            game_state: Current game state
            phase_specific_info: Additional phase-specific information
            
        Returns:
            Observation tailored for the player
        """
        player = game_state.players[player_id]
        role_specific_info = {}
        
        # Werewolves know each other
        if player.role == RoleType.WEREWOLF:
            role_specific_info['werewolf_team'] = [
                pid for pid in game_state.alive_players
                if game_state.players[pid].role == RoleType.WEREWOLF
            ]
        
        # Seer gets check results
        if player.role == RoleType.SEER and phase_specific_info:
            if 'seer_result' in phase_specific_info:
                role_specific_info['seer_result'] = phase_specific_info['seer_result']
        
        # Guard gets feedback on successful protections
        if player.role == RoleType.GUARD:
            recent_saves = [
                e for e in game_state.game_history
                if e.get('type') == 'protection_saved' and e.get('guard_id') == player_id
            ]
            if recent_saves:
                last_save = recent_saves[-1]
                role_specific_info['protection_feedback'] = {
                    'success': True,
                    'saved_player': last_save.get('player_id'),
                    'round': last_save.get('round')
                }
        
        # Witch is informed of the victim
        if player.role == RoleType.WITCH and phase_specific_info:
            if 'werewolf_victim' in phase_specific_info:
                role_specific_info['werewolf_victim'] = phase_specific_info['werewolf_victim']
                role_specific_info['heal_available'] = not game_state.witch_heal_used
                role_specific_info['poison_available'] = not game_state.witch_poison_used
        
        observation = Observation(
            phase=game_state.phase,
            round_number=game_state.round_number,
            alive_players=game_state.alive_players.copy(),
            eliminated_players=game_state.eliminated_players.copy(),
            sheriff_id=game_state.sheriff_id,
            role_specific_info=role_specific_info
        )
        
        if phase_specific_info:
            observation.night_result = phase_specific_info.get('night_result')
            observation.votes = phase_specific_info.get('votes')
        
        return observation