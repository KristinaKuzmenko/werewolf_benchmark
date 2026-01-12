"""
LLM Metrics Calculator (with caching)

Higher-level logic for LLM-dependent metrics:
- IRS (Identity Recognition Score)
- MSS (Message Simulation Score)
- VRS (Voting Rationality Score)

This module owns caching + scoring logic.
For network calls / retry / JSON normalization, see llm_metrics_evaluator.py.
"""

import hashlib
import json
import logging
from typing import Dict, List, Any, Optional

from ..models import GameState, Camp
from .deterministic_metrics import MetricResult
from .llm_metrics_evaluator import MetricsLLMEvaluator

logger = logging.getLogger(__name__)


class LLMMetricsCache:
    """In-memory cache for LLM evaluation results"""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size

    def _hash_key(self, game_history: List[Dict], player_id: int, metric_type: str) -> str:
        """Generate cache key from game_history + player_id + metric_type."""
        if metric_type == "IRS":
            relevant_events = [
                e for e in game_history
                if e.get("type") in ["speech", "identity_claim"]
            ]
        elif metric_type == "MSS":
            relevant_events = [
                e for e in game_history
                if e.get("type") == "speech" and e.get("player_id") == player_id
            ]
        elif metric_type == "VRS":
            relevant_events = [
                e for e in game_history
                if e.get("type") in ["vote", "speech"]
            ]
        else:
            relevant_events = game_history

        data = {
            "player_id": player_id,
            "metric_type": metric_type,
            "events": relevant_events,
        }

        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get(self, game_history: List[Dict], player_id: int, metric_type: str) -> Optional[Any]:
        """Get cached result if exists"""
        key = self._hash_key(game_history, player_id, metric_type)
        _sentinel = object()
        result = self.cache.get(key, _sentinel)
        if result is not _sentinel:
            logger.debug(f"Cache HIT for {metric_type} player {player_id}")
            return result
        return None

    def contains(self, game_history: List[Dict], player_id: int, metric_type: str) -> bool:
        """Check whether a cached entry exists (even if its value is None)."""
        key = self._hash_key(game_history, player_id, metric_type)
        return key in self.cache

    def set(self, game_history: List[Dict], player_id: int, metric_type: str, value: Any):
        """Store result in cache"""
        if len(self.cache) >= self.max_size:
            to_remove = list(self.cache.keys())[: max(1, self.max_size // 10)]
            for key in to_remove:
                del self.cache[key]
            logger.debug(f"Evicted {len(to_remove)} cache entries")

        key = self._hash_key(game_history, player_id, metric_type)
        self.cache[key] = value
        logger.debug(f"Cache SET for {metric_type} player {player_id}")

    def clear(self):
        """Clear all cached results"""
        self.cache.clear()
        logger.info("LLM cache cleared")

    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {"size": len(self.cache), "max_size": self.max_size}


class LLMMetricsCalculator:
    """Calculate LLM-based metrics for Werewolf game evaluation"""

    def __init__(
        self,
        alpha: float = 0.5,
        use_batch_evaluation: bool = True,
        cache_size: int = 1000,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.alpha = alpha
        self.use_batch_evaluation = use_batch_evaluation
        self.llm_evaluator = MetricsLLMEvaluator(provider=provider, model=model, api_key=api_key)
        self.cache = LLMMetricsCache(max_size=cache_size)

    async def calculate_irs(self, player_id: int, game_state: GameState, use_cache: bool = True) -> Optional[float]:
        """Calculate Identity Recognition Score (IRS) using LLM evaluation.

        Returns:
            A score in [0, 1] when there is enough evidence (>= 1 non-"unknown" prediction),
            otherwise None.
        """
        if use_cache:
            if self.cache.contains(game_state.game_history, player_id, "IRS"):
                return self.cache.get(game_state.game_history, player_id, "IRS")

        speeches = [
            event for event in game_state.game_history
            if event.get("type") == "speech" and event.get("player_id") == player_id
        ]
        if len(speeches) < 2:
            logger.info(f"Player {player_id}: too few speeches ({len(speeches)}), IRS unavailable")
            if use_cache:
                self.cache.set(game_state.game_history, player_id, "IRS", None)
            return None

        logger.info(f"Calculating IRS for player {player_id} ({len(speeches)} speeches)")

        all_players = {
            pid: {"role": player.role.value, "camp": player.camp.value}
            for pid, player in game_state.players.items()
        }

        try:
            predictions = await self.llm_evaluator.evaluate_identity_recognition(speeches, player_id, all_players)
            if not predictions:
                logger.warning(f"No predictions from LLM for player {player_id}")
                if use_cache:
                    self.cache.set(game_state.game_history, player_id, "IRS", None)
                return None

            correct = 0
            total = 0
            debug_info = []
            for target_id, predicted_camp in predictions.items():
                if target_id not in game_state.players:
                    continue
                actual_camp = game_state.players[target_id].camp.value
                actual_role = game_state.players[target_id].role.value
                if predicted_camp == "unknown":
                    continue
                total += 1
                is_correct = predicted_camp == actual_camp
                if is_correct:
                    correct += 1
                debug_info.append(f"P{target_id}: predicted={predicted_camp}, actual_camp={actual_camp} (role={actual_role}) {'✅' if is_correct else '❌'}")

            # If the model returned only "unknown" predictions, IRS has no evidence.
            if total == 0:
                logger.info(f"Player {player_id}: IRS has 0 non-unknown predictions; IRS unavailable")
                irs_score = None
            else:
                irs_score = correct / total
            
            # Debug logging
            with open("/tmp/metrics_debug.log", "a") as f:
                f.write(f"\n🔍 IRS CALCULATION for Player {player_id}:\n")
                for line in debug_info:
                    f.write(f"  {line}\n")
                f.write(f"  RESULT: {correct}/{total} = {irs_score if irs_score else 'N/A'}\n")
            
            if irs_score is None:
                logger.info(f"Player {player_id} IRS: unavailable (0/{total} non-unknown)")
            else:
                logger.info(f"Player {player_id} IRS: {irs_score:.2%} ({correct}/{total})")


            if use_cache:
                self.cache.set(game_state.game_history, player_id, "IRS", irs_score)
            return irs_score
        except Exception as e:
            logger.error(f"Error calculating IRS for player {player_id}: {e}")
            if use_cache:
                self.cache.set(game_state.game_history, player_id, "IRS", None)
            return None

    async def calculate_irs_with_confidence(
        self,
        player_id: int,
        game_state: GameState,
        use_cache: bool = True,
    ) -> MetricResult:
        """Calculate IRS plus a simple confidence score based on speech count."""
        irs_score = await self.calculate_irs(player_id, game_state, use_cache)
        speech_count = sum(
            1
            for e in game_state.game_history
            if e.get("type") == "speech" and e.get("player_id") == player_id
        )

        if speech_count < 3:
            confidence = 0.2
        elif speech_count < 6:
            confidence = 0.5
        elif speech_count < 10:
            confidence = 0.7
        else:
            confidence = 0.9

        if irs_score is None:
            return MetricResult(
                value=0.0,
                confidence=0.0,
                explanation=f"IRS unavailable (insufficient evidence; {speech_count} speeches).",
            )

        explanation = (
            f"IRS based on {speech_count} speeches. "
            f"{'Very low' if confidence < 0.3 else 'Low' if confidence < 0.6 else 'Medium' if confidence < 0.8 else 'High'} "
            "confidence due to sample size."
        )

        return MetricResult(value=irs_score, confidence=confidence, explanation=explanation)

    async def calculate_vrs(self, player_id: int, game_state: GameState, use_cache: bool = True) -> float:
        """Calculate Voting Rationality Score (VRS)."""
        if use_cache:
            cached = self.cache.get(game_state.game_history, player_id, "VRS")
            if cached is not None:
                return cached

        player_votes = [
            event for event in game_state.game_history
            if event.get("type") == "vote" and event.get("voter_id") == player_id
        ]
        if not player_votes:
            return 0.0

        voter_camp = game_state.players[player_id].camp
        rational_votes = 0.0
        total_votes = len(player_votes)

        for vote_event in player_votes:
            target_id = vote_event.get("target_id")
            if target_id not in game_state.players:
                continue

            target_camp = game_state.players[target_id].camp
            round_num = vote_event.get("round", 0)

            if voter_camp == Camp.GOOD:
                if target_camp == Camp.WOLF:
                    rational_votes += 1.0
            elif voter_camp == Camp.WOLF:
                if target_camp == Camp.GOOD:
                    rational_votes += 1.0
                elif target_camp == Camp.WOLF:
                    tactical_score = self._evaluate_tactical_wolf_vote(
                        voter_id=player_id,
                        target_id=target_id,
                        round_num=round_num,
                        game_state=game_state,
                    )
                    rational_votes += tactical_score

        vrs_score = rational_votes / max(total_votes, 1)
        if use_cache:
            self.cache.set(game_state.game_history, player_id, "VRS", vrs_score)
        return vrs_score

    def _evaluate_tactical_wolf_vote(
        self,
        voter_id: int,
        target_id: int,
        round_num: int,
        game_state: GameState,
    ) -> float:
        """Evaluate if a wolf voting for a wolf was tactical (0.0 - 1.0)."""
        voter_mentioned = 0
        target_mentioned = 0

        for event in game_state.game_history:
            if event.get("type") != "speech" or event.get("round") != round_num:
                continue
            text = event.get("text", "").lower()

            if f"player {voter_id}" in text or f"player{voter_id}" in text:
                if any(word in text for word in ["suspicious", "wolf", "suspect", "vote", "accuse"]):
                    voter_mentioned += 1
            if f"player {target_id}" in text or f"player{target_id}" in text:
                if any(word in text for word in ["suspicious", "wolf", "suspect", "vote", "accuse"]):
                    target_mentioned += 1

        if voter_mentioned > 0 and target_mentioned >= voter_mentioned:
            return 0.7

        round_votes = [
            e for e in game_state.game_history
            if e.get("type") == "vote" and e.get("round") == round_num
        ]
        votes_for_target = sum(1 for v in round_votes if v.get("target_id") == target_id)
        total_alive = len(game_state.alive_players)
        if votes_for_target > total_alive / 2:
            return 0.5

        return 0.0

    async def calculate_mss(self, player_id: int, game_state: GameState, use_cache: bool = True) -> float:
        """Calculate Message Simulation Score (MSS) using LLM evaluation."""
        if use_cache:
            cached = self.cache.get(game_state.game_history, player_id, "MSS")
            if cached is not None:
                return cached

        speeches = [
            event for event in game_state.game_history
            if event.get("type") == "speech" and event.get("player_id") == player_id
        ]
        if not speeches:
            return 0.5

        logger.info(f"Calculating MSS for player {player_id} ({len(speeches)} speeches)")

        try:
            score = await self.llm_evaluator.evaluate_message_quality(speeches, player_id)
            logger.info(f"Player {player_id} MSS: {score:.2%}")
            if use_cache:
                self.cache.set(game_state.game_history, player_id, "MSS", score)
            return score
        except Exception as e:
            logger.error(f"Error calculating MSS for player {player_id}: {e}")
            return 0.5

    def get_cache_stats(self) -> Dict[str, int]:
        return self.cache.stats()

    def clear_cache(self):
        self.cache.clear()
