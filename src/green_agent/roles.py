"""
Role definitions and role-specific logic for the Werewolf game.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from .models import (
    RoleType, Camp, PlayerInfo, Action, ActionType, 
    GameState, RoleMetrics
)


class Role(ABC):
    """Abstract base class for all roles"""
    
    def __init__(self, role_type: RoleType, camp: Camp):
        self.role_type = role_type
        self.camp = camp
    
    @abstractmethod
    def can_act_at_night(self) -> bool:
        """Whether this role can take action during night phase"""
        pass
    
    @abstractmethod
    def get_night_action_type(self) -> Optional[ActionType]:
        """What type of action this role takes at night"""
        pass
    
    @abstractmethod
    def validate_action(self, action: Action, game_state: GameState) -> bool:
        """Validate if the action is legal for this role"""
        pass
    
    @abstractmethod
    def evaluate_performance(
        self, 
        player_id: int, 
        game_state: GameState, 
        actions_taken: List[Action]
    ) -> RoleMetrics:
        """Evaluate this role's performance"""
        pass


class Werewolf(Role):
    """Werewolf role - kills villagers at night"""
    
    def __init__(self):
        super().__init__(RoleType.WEREWOLF, Camp.WOLF)
    
    def can_act_at_night(self) -> bool:
        return True
    
    def get_night_action_type(self) -> Optional[ActionType]:
        return ActionType.KILL
    
    def validate_action(self, action: Action, game_state: GameState) -> bool:
        if action.action_type != ActionType.KILL:
            return False
        
        # Can only kill alive non-werewolf players
        if action.target_id not in game_state.alive_players:
            return False
        
        target = game_state.players[action.target_id]
        return target.camp != Camp.WOLF
    
    def evaluate_performance(
        self, 
        player_id: int, 
        game_state: GameState, 
        actions_taken: List[Action]
    ) -> RoleMetrics:
        """Evaluate werewolf survival and contribution"""
        alpha = 0.6  # Weight: 60% survival duration, 40% binary survival
        survived = game_state.players[player_id].is_alive
        total_rounds = max(game_state.round_number, 1)
        
        # Find elimination round
        survival_rounds = total_rounds
        for event in game_state.game_history:
            if (event.get('type') == 'elimination' and 
                event.get('player_id') == player_id):
                survival_rounds = event.get('round', total_rounds)
                break
        
        # WereWolf-Plus formula: α·(survival_rounds/total) + (1-α)·I(survived)
        survival_score = (
            alpha * (survival_rounds / total_rounds) +
            (1 - alpha) * (1.0 if survived else 0.0)
        )
        
        return RoleMetrics(werewolf_survival=survival_score)


class Villager(Role):
    """Basic villager role - no special abilities"""
    
    def __init__(self):
        super().__init__(RoleType.VILLAGER, Camp.GOOD)
    
    def can_act_at_night(self) -> bool:
        return False
    
    def get_night_action_type(self) -> Optional[ActionType]:
        return None
    
    def validate_action(self, action: Action, game_state: GameState) -> bool:
        # Villagers can only vote during the day
        return action.action_type in [ActionType.VOTE, ActionType.SPEAK, ActionType.ELECT_SHERIFF]
    
    def evaluate_performance(
        self, 
        player_id: int, 
        game_state: GameState, 
        actions_taken: List[Action]
    ) -> RoleMetrics:
        # Villagers evaluated through player-level metrics only
        return RoleMetrics()


class Seer(Role):
    """Seer role - checks one player's identity each night"""
    
    def __init__(self):
        super().__init__(RoleType.SEER, Camp.GOOD)
    
    def can_act_at_night(self) -> bool:
        return True
    
    def get_night_action_type(self) -> Optional[ActionType]:
        return ActionType.CHECK
    
    def validate_action(self, action: Action, game_state: GameState) -> bool:
        if action.action_type != ActionType.CHECK:
            return False
        
        # Can only check alive players (except self)
        return (
            action.target_id in game_state.alive_players and
            action.target_id != action.player_id
        )
    
    def evaluate_performance(
        self, 
        player_id: int, 
        game_state: GameState, 
        actions_taken: List[Action]
    ) -> RoleMetrics:
        """Evaluate seer's accuracy in identifying werewolves"""
        check_actions = [a for a in actions_taken if a.action_type == ActionType.CHECK]
        
        werewolves_checked = sum(
            1 for action in check_actions
            if game_state.players[action.target_id].role == RoleType.WEREWOLF
        )
        
        
        total_checks = len(check_actions)
        seer_accuracy = werewolves_checked / max(total_checks, 1)
        
        return RoleMetrics(seer_accuracy=seer_accuracy)


class Witch(Role):
    """Witch role - has one heal and one poison potion"""
    
    def __init__(self):
        super().__init__(RoleType.WITCH, Camp.GOOD)
    
    def can_act_at_night(self) -> bool:
        return True
    
    def get_night_action_type(self) -> Optional[ActionType]:
        return ActionType.HEAL  # Or POISON
    
    def validate_action(self, action: Action, game_state: GameState) -> bool:
        if action.action_type == ActionType.HEAL:
            return not game_state.witch_heal_used
        elif action.action_type == ActionType.POISON:
            return (
                not game_state.witch_poison_used and
                action.target_id in game_state.alive_players
            )
        return False
    
    def evaluate_performance(
        self, 
        player_id: int, 
        game_state: GameState, 
        actions_taken: List[Action]
    ) -> RoleMetrics:
        """Evaluate witch's effectiveness with potions"""
        heal_actions = [a for a in actions_taken if a.action_type == ActionType.HEAL]
        poison_actions = [a for a in actions_taken if a.action_type == ActionType.POISON]
        
        correct_uses = 0
        total_uses = 0
        
        # Check if heals saved good camp members
        for action in heal_actions:
            total_uses += 1
            if action.target_id and game_state.players[action.target_id].camp == Camp.GOOD:
                correct_uses += 1
        
        # Check if poisons killed werewolves
        for action in poison_actions:
            total_uses += 1
            if action.target_id and game_state.players[action.target_id].camp == Camp.WOLF:
                correct_uses += 1
        
        witch_effectiveness = correct_uses / max(total_uses, 1)
        
        return RoleMetrics(witch_effectiveness=witch_effectiveness)


class Sheriff:
    """Sheriff position metrics (not a separate role).

    Measures how effectively the sheriff influences other players' votes.
    """

    @staticmethod
    def evaluate_performance(
        sheriff_id: int,
        game_state: GameState,
    ) -> float:
        """Compute sheriff influence on revotes.

        WereWolf-Plus style:
        SheriffInfluence = #(changed_vote AND revote == recommendation) / #total_vote_opportunities

        Notes about this codebase's history schema:
        - Sheriff recommendation is stored as event type 'sheriff_recommendation'
          with fields: round, sheriff_id, recommended_target, previous_votes
        - Votes are stored as event type 'vote' with field 'voting_round' (1 or 2)
        """
        recommendations = [
            e for e in game_state.game_history
            if e.get('type') == 'sheriff_recommendation' and e.get('sheriff_id') == sheriff_id
        ]

        if not recommendations:
            # Fallback: older/alternate logging (if present)
            recommendations = [
                e for e in game_state.game_history
                if e.get('type') == 'sheriff_summary' and e.get('sheriff_id') == sheriff_id
            ]

        if not recommendations:
            return 0.5

        successful_influences = 0
        total_opportunities = 0

        for rec in recommendations:
            round_num = rec.get('round')
            recommended_target = rec.get('recommended_target') or rec.get('recommendation')
            if recommended_target is None or round_num is None:
                continue

            votes_r1 = rec.get('previous_votes')
            if not isinstance(votes_r1, dict):
                votes_r1 = {
                    e.get('voter_id'): e.get('target_id')
                    for e in game_state.game_history
                    if e.get('type') == 'vote'
                    and e.get('round') == round_num
                    and (e.get('voting_round') == 1 or e.get('vote_round') == 1)
                }

            votes_r2 = {
                e.get('voter_id'): e.get('target_id')
                for e in game_state.game_history
                if e.get('type') == 'vote'
                and e.get('round') == round_num
                and (e.get('voting_round') == 2 or e.get('vote_round') == 2)
            }

            for voter_id, r2_target in votes_r2.items():
                if voter_id is None or voter_id == sheriff_id:
                    continue
                total_opportunities += 1

                r1_target = votes_r1.get(voter_id)
                if r1_target != recommended_target and r2_target == recommended_target:
                    successful_influences += 1

        if total_opportunities == 0:
            return 0.5

        return successful_influences / total_opportunities


class Hunter(Role):
    """Hunter role - shoots one player when eliminated"""
    
    def __init__(self):
        super().__init__(RoleType.HUNTER, Camp.GOOD)
    
    def can_act_at_night(self) -> bool:
        return False
    
    def get_night_action_type(self) -> Optional[ActionType]:
        return None
    
    def validate_action(self, action: Action, game_state: GameState) -> bool:
        # Hunter can only shoot when dying
        if action.action_type != ActionType.SHOOT:
            return False
        
        return action.target_id in game_state.alive_players
    
    def evaluate_performance(
        self, 
        player_id: int, 
        game_state: GameState, 
        actions_taken: List[Action]
    ) -> RoleMetrics:
        """Evaluate if hunter shot a werewolf"""
        shoot_actions = [a for a in actions_taken if a.action_type == ActionType.SHOOT]
        
        werewolves_shot = sum(
            1 for action in shoot_actions
            if action.target_id and game_state.players[action.target_id].camp == Camp.WOLF
        )
        
        hunter_accuracy = werewolves_shot / max(len(shoot_actions), 1)
        
        return RoleMetrics(hunter_accuracy=hunter_accuracy)


class Guard(Role):
    """Guard role - protects one player each night (not same twice in a row)"""
    
    def __init__(self):
        super().__init__(RoleType.GUARD, Camp.GOOD)
    
    def can_act_at_night(self) -> bool:
        return True
    
    def get_night_action_type(self) -> Optional[ActionType]:
        return ActionType.PROTECT
    
    def validate_action(self, action: Action, game_state: GameState) -> bool:
        if action.action_type != ActionType.PROTECT:
            return False
        
        # Cannot protect same player consecutively
        if action.target_id == game_state.last_protected:
            return False
        
        return action.target_id in game_state.alive_players
    
    def evaluate_performance(
        self, 
        player_id: int, 
        game_state: GameState, 
        actions_taken: List[Action]
    ) -> RoleMetrics:
        """Evaluate guard's protection effectiveness"""
        alpha = 0.5  # Weight between protecting good camp and preventing elimination
        
        protect_actions = [a for a in actions_taken if a.action_type == ActionType.PROTECT]
        
        good_camp_protected = sum(
            1 for action in protect_actions
            if action.target_id and game_state.players[action.target_id].camp == Camp.GOOD
        )
        
        # Check if protected player was actually attacked
        successful_saves = 0
        for action in protect_actions:
            # This would need to be checked against night kill attempts
            # For now, simplified calculation
            for event in game_state.game_history:
                if (event.get('type') == 'protection_saved' and 
                    event.get('player_id') == action.target_id):
                    successful_saves += 1
        
        total_protections = len(protect_actions)
        
        if total_protections == 0:
            guard_effectiveness = 0.0
        else:
            guard_effectiveness = (
                alpha * (good_camp_protected / total_protections) +
                (1 - alpha) * (successful_saves / total_protections)
            )
        
        return RoleMetrics(guard_effectiveness=guard_effectiveness)


# Role factory
def create_role(role_type: RoleType) -> Role:
    """Factory function to create role instances"""
    role_map = {
        RoleType.WEREWOLF: Werewolf,
        RoleType.VILLAGER: Villager,
        RoleType.SEER: Seer,
        RoleType.WITCH: Witch,
        RoleType.HUNTER: Hunter,
        RoleType.GUARD: Guard,
    }
    
    role_class = role_map.get(role_type)
    if not role_class:
        raise ValueError(f"Unknown role type: {role_type}")
    
    return role_class()


def get_camp_for_role(role_type: RoleType) -> Camp:
    """Get the camp affiliation for a role type"""
    if role_type == RoleType.WEREWOLF:
        return Camp.WOLF
    return Camp.GOOD