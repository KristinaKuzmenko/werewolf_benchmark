"""
Deterministic Metrics Calculator

Calculates metrics that don't require LLM evaluation:
- SR (Survival Rate)
- Role-specific metrics (Seer, Guard, Witch, etc.)
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..models import (
    GameState, PlayerMetrics, RoleMetrics, Action, ActionType,
    RoleType, Camp, GameResult
)
from ..roles import create_role, Sheriff

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for metric calculation results with confidence"""
    value: float
    confidence: float  # 0.0-1.0 based on sample size
    explanation: str


def validate_game_history(game_history: List[Dict]) -> List[str]:
    """
    Validate game history for completeness.
    
    Returns:
        List of warning messages about missing event types
    """
    warnings = []
    
    required_events = ['speech', 'vote', 'elimination']
    optional_events = ['identity_claim', 'seer_check', 'protection_saved']
    
    event_types = {event.get('type') for event in game_history}
    
    # Check required events
    for req_type in required_events:
        if req_type not in event_types:
            warnings.append(f"Missing required event type: {req_type}")
    
    # Check optional events (info only)
    missing_optional = [opt for opt in optional_events if opt not in event_types]
    if missing_optional:
        warnings.append(
            f"Missing optional event types: {', '.join(missing_optional)} "
            f"(may affect advanced metrics)"
        )
    
    # Check event counts
    speech_count = sum(1 for e in game_history if e.get('type') == 'speech')
    vote_count = sum(1 for e in game_history if e.get('type') == 'vote')
    
    if speech_count == 0:
        warnings.append("No speech events found (IRS/MSS will be unavailable)")
    elif speech_count < 10:
        warnings.append(f"Very few speeches ({speech_count}) - metrics may be unreliable")
    
    return warnings


class DeterministicMetricsCalculator:
    """Calculate deterministic (non-LLM) metrics for Werewolf game evaluation"""
    
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Weight for various metric calculations (default 0.5)
        """
        self.alpha = alpha
    
    def calculate_player_metrics(
        self,
        player_id: int,
        game_state: GameState,
        all_games_history: List[GameState]
    ) -> PlayerMetrics:
        """
        Calculate deterministic player-level metrics across multiple games.
        
        Metrics:
        - SR (Survival Rate): Overall survival across games
        """
        metrics = PlayerMetrics()
        
        # Calculate SR - Survival Rate
        all_survived = 0
        
        for game in all_games_history:
            player = game.players.get(player_id)
            if player and player.is_alive:
                all_survived += 1
        
        metrics.total_games = len(all_games_history)
        metrics.sr = all_survived / max(len(all_games_history), 1)
        
        return metrics

    def _extract_role_actions(
        self,
        player_id: int,
        role_type: RoleType,
        game_history: List[Dict],
    ) -> List[Action]:
        """Extract role actions from the actual event types recorded in game_history."""
        actions: List[Action] = []

        for event in game_history:
            event_type = event.get('type')

            # Seer checks
            if role_type == RoleType.SEER and event_type == 'seer_check':
                if event.get('seer_id') == player_id:
                    target_id = event.get('target_id')
                    if target_id is not None:
                        actions.append(
                            Action(
                                action_type=ActionType.CHECK,
                                player_id=player_id,
                                target_id=target_id,
                            )
                        )

            # Witch actions (history doesn't track witch_id; assume single witch per game)
            elif role_type == RoleType.WITCH and event_type == 'witch_heal':
                target_id = event.get('saved_player')
                if target_id is not None:
                    actions.append(
                        Action(
                            action_type=ActionType.HEAL,
                            player_id=player_id,
                            target_id=target_id,
                        )
                    )
            elif role_type == RoleType.WITCH and event_type == 'witch_poison':
                target_id = event.get('poisoned_player')
                if target_id is not None:
                    actions.append(
                        Action(
                            action_type=ActionType.POISON,
                            player_id=player_id,
                            target_id=target_id,
                        )
                    )

            # Guard protections (only successful saves are explicitly logged)
            elif role_type == RoleType.GUARD and event_type == 'protection_saved':
                if event.get('guard_id') == player_id:
                    target_id = event.get('player_id')
                    if target_id is not None:
                        actions.append(
                            Action(
                                action_type=ActionType.PROTECT,
                                player_id=player_id,
                                target_id=target_id,
                            )
                        )

            # Hunter shot
            elif role_type == RoleType.HUNTER and event_type == 'hunter_shot':
                if event.get('hunter_id') == player_id:
                    target_id = event.get('shot_target')
                    if target_id is not None:
                        actions.append(
                            Action(
                                action_type=ActionType.SHOOT,
                                player_id=player_id,
                                target_id=target_id,
                            )
                        )

        return actions
    
    def calculate_sr_with_confidence(
        self,
        player_id: int,
        all_games_history: List[GameState]
    ) -> MetricResult:
        """
        Calculate Survival Rate with confidence score.
        
        Confidence based on sample size:
        - < 5 games: 0.3
        - 5-10 games: 0.6
        - 10-20 games: 0.8
        - 20+ games: 1.0
        """
        survived_count = 0
        total_games = len(all_games_history)
        
        for game in all_games_history:
            player = game.players.get(player_id)
            if player and player.is_alive:
                survived_count += 1
        
        sr = survived_count / max(total_games, 1) if total_games > 0 else 0.0
        
        # Calculate confidence based on sample size
        if total_games < 5:
            confidence = 0.3
        elif total_games < 10:
            confidence = 0.6
        elif total_games < 20:
            confidence = 0.8
        else:
            confidence = 1.0
        
        explanation = (
            f"Survived {survived_count}/{total_games} games. "
            f"{'Low' if confidence < 0.5 else 'Medium' if confidence < 0.8 else 'High'} "
            f"confidence due to sample size."
        )
        
        return MetricResult(
            value=sr,
            confidence=confidence,
            explanation=explanation
        )
    
    def calculate_role_metrics(
        self,
        player_id: int,
        role_type: RoleType,
        game_state: GameState,
        actions_taken: List[Action]
    ) -> RoleMetrics:
        """Calculate role-specific metrics"""
        role = create_role(role_type)
        return role.evaluate_performance(player_id, game_state, actions_taken)
    
    def calculate_sheriff_metrics(
        self,
        sheriff_id: int,
        game_state: GameState
    ) -> float:
        """Calculate sheriff influence metric (vote-persuasion effectiveness)."""
        return Sheriff.evaluate_performance(sheriff_id=sheriff_id, game_state=game_state)
