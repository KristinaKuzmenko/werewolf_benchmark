"""
Evaluation metrics for player and role performance in Werewolf game.
Based on Werewolf-Bench and WereWolf-Plus paper metrics.

DEPRECATED: This is the legacy unified metrics calculator.
New architecture:
- deterministic_metrics.py: SR, role-specific (no LLM)
- llm_metrics_calculator.py: IRS, MSS, VRS (with LLM + caching)
- advanced_metrics.py: manipulation, persuasion (game-level)

This file maintained for backward compatibility.
"""
import os
import logging
from typing import Dict, List, Optional
import asyncio
from ..models import (
    GameState, PlayerMetrics, RoleMetrics, Action, ActionType,
    RoleType, Camp, GameResult
)
from .deterministic_metrics import (
    DeterministicMetricsCalculator,
    validate_game_history,
    MetricResult
)
from .llm_metrics_calculator import LLMMetricsCalculator
from .advanced_metrics import AdvancedMetricsCalculator, calculate_game_advanced_metrics


class MetricsCalculator:
    """
    UNIFIED metrics calculator (backward compatibility layer).
    
    Delegates to specialized calculators:
    - DeterministicMetricsCalculator: SR, role metrics
    - LLMMetricsCalculator: IRS, MSS, VRS (with caching)
    - AdvancedMetricsCalculator: manipulation, persuasion, vote swings
    """
    
    def __init__(self, alpha: float = 0.5, use_llm_metrics: bool = True, use_batch_evaluation: bool = True):
        """
        Args:
            alpha: Weight for various metric calculations (default 0.5)
            use_llm_metrics: Whether to use LLM for IRS/MSS (requires API key)
            use_batch_evaluation: Use optimized batch API calls (1 call instead of 24)
        """
        self.alpha = alpha
        self.use_llm_metrics = use_llm_metrics
        self.use_batch_evaluation = use_batch_evaluation
        self._validated_games = set()  # Track games already validated
        
        # Initialize specialized calculators
        self.deterministic = DeterministicMetricsCalculator(alpha=alpha)
        self.llm = LLMMetricsCalculator(
            alpha=alpha,
            use_batch_evaluation=use_batch_evaluation
        ) if use_llm_metrics else None
        self.advanced = AdvancedMetricsCalculator()
        
        # Legacy compatibility
        self.llm_evaluator = self.llm.llm_evaluator if self.llm else None
    
    def calculate_player_metrics(
        self,
        player_id: int,
        game_state: GameState,
        all_games_history: List[GameState]
    ) -> PlayerMetrics:
        """
        Calculate deterministic player-level metrics across multiple games.
        Delegates to DeterministicMetricsCalculator.
        
        Metrics:
        - SR (Survival Rate): Overall survival
        """
        # Validate game history (once per game, not per player)
        game_id = id(game_state.game_history)  # Use object id as unique game identifier
        if game_id not in self._validated_games:
            warnings = validate_game_history(game_state.game_history)
            if warnings:
                logger = logging.getLogger(__name__)
                # Only log critical warnings, not optional event warnings
                for warning in warnings:
                    if "Missing required" in warning or "No speech events" in warning:
                        logger.warning(f"Game history validation: {warning}")
                    else:
                        logger.debug(f"Game history validation: {warning}")
            self._validated_games.add(game_id)
        
        # Delegate to deterministic calculator
        return self.deterministic.calculate_player_metrics(
            player_id, game_state, all_games_history
        )
    
    async def calculate_irs(
        self,
        player_id: int,
        game_state: GameState
    ) -> Optional[float]:
        """
        Calculate Identity Recognition Score (IRS) using LLM evaluation.
        Delegates to LLMMetricsCalculator with caching.
        
        IRS measures how well a player correctly identifies other players' camps.
        """
        if not self.use_llm_metrics or not self.llm:
            return None
        
        # Collect all speeches from this player
        speeches = [
            event for event in game_state.game_history
            if event.get('type') == 'speech' and event.get('player_id') == player_id
        ]
        
        if not speeches:
            return None
        
        logger = logging.getLogger(__name__)
        # Delegate to LLM calculator (with caching)
        return await self.llm.calculate_irs(player_id, game_state, use_cache=True)
    
    async def calculate_vrs(
        self,
        player_id: int,
        game_state: GameState
    ) -> float:
        """
        Calculate Voting Rationality Score (VRS) with tactical voting support.
        Delegates to LLMMetricsCalculator with caching.
        
        VRS measures if a player votes rationally based on their camp:
        - Good camp: rational to vote for suspected wolves
        - Wolf camp: rational to vote for good players OR tactically for wolves
        
        For wolves, voting for fellow wolf can be rational if:
        - Under suspicion (self-preservation)
        - Fellow wolf is already doomed (vote with majority)
        """
        if not self.llm:
            return 0.0
        
        # Delegate to LLM calculator (with caching)
        return await self.llm.calculate_vrs(player_id, game_state, use_cache=True)
    
    async def calculate_mss(
        self,
        player_id: int,
        game_state: GameState
    ) -> float:
        """
        Calculate Message Simulation Score (MSS) using LLM evaluation.
        Delegates to LLMMetricsCalculator with caching.
        
        MSS measures how human-like and realistic the player's messages are.
        """
        if not self.use_llm_metrics or not self.llm:
            return 0.5  # Neutral default
        
        # Delegate to LLM calculator (with caching)
        return await self.llm.calculate_mss(player_id, game_state, use_cache=True)
    
    def calculate_role_metrics(
        self,
        player_id: int,
        role_type: RoleType,
        game_state: GameState,
        actions_taken: List[Action]
    ) -> RoleMetrics:
        """Calculate role-specific metrics. Delegates to DeterministicMetricsCalculator."""
        return self.deterministic.calculate_role_metrics(
            player_id, role_type, game_state, actions_taken
        )
    
    def calculate_sheriff_metrics(
        self,
        sheriff_id: int,
        game_state: GameState
    ) -> float:
        """Calculate sheriff influence metric. Delegates to DeterministicMetricsCalculator."""
        return self.deterministic.calculate_sheriff_metrics(sheriff_id, game_state)
    
    async def calculate_game_result(
        self,
        game_state: GameState,
        player_actions: Dict[int, List[Action]]
    ) -> GameResult:
        """Calculate complete game result with all metrics (async for LLM calls)"""
        
        player_metrics_dict = {}
        role_metrics_dict = {}
        
        logger = logging.getLogger(__name__)
        import time
        start_time = time.time()
        
        # Calculate basic metrics first
        for player_id, player in game_state.players.items():
            # Calculate player-level metrics (simplified for single game)
            player_metrics = self.calculate_player_metrics(
                player_id, 
                game_state, 
                [game_state]  # Single game for now
            )
            player_metrics_dict[player_id] = player_metrics
            
            # Calculate role-specific metrics
            actions = player_actions.get(player_id, [])
            role_metrics = self.calculate_role_metrics(
                player_id,
                player.role,
                game_state,
                actions
            )
            role_metrics_dict[player_id] = role_metrics
            
            # Add sheriff metrics if applicable
            if player.is_sheriff:
                sheriff_influence = self.calculate_sheriff_metrics(player_id, game_state)
                role_metrics.sheriff_influence = sheriff_influence
        
        # Execute LLM metrics
        if self.use_llm_metrics and self.llm_evaluator:
            if self.use_batch_evaluation:
                # OPTIMIZED: Single batch API call for all players
                with open("/tmp/metrics_debug.log", "a") as f:
                    f.write("✅ ENTERING BATCH EVALUATION BLOCK\n")
                
                # Prepare data
                all_players = {
                    pid: {
                        'role': player.role.value,
                        'camp': player.camp.value
                    }
                    for pid, player in game_state.players.items()
                }
                
                try:
                    logger.info(f"🔍 Calling batch evaluation for {len(all_players)} players...")
                    batch_results = await self.llm_evaluator.evaluate_all_players_batch(
                        game_state.game_history,
                        all_players
                    )
                    logger.info(f"📊 Batch results received: {len(batch_results)} players evaluated")
                    
                    # Apply batch results
                    for player_id, results in batch_results.items():
                        if player_id in player_metrics_dict:
                            # IRS from batch
                            irs_predictions = results.get("irs", {})
                            player_speeches_count = sum(
                                1
                                for e in game_state.game_history
                                if e.get('type') == 'speech' and e.get('player_id') == player_id
                            )

                            if player_speeches_count < 2:
                                player_metrics_dict[player_id].irs = None
                            elif not irs_predictions:
                                player_metrics_dict[player_id].irs = None
                            else:
                                # Score only on non-unknown predictions
                                non_unknown = {
                                    tid: pred for tid, pred in irs_predictions.items() if pred != "unknown"
                                }
                                if not non_unknown:
                                    player_metrics_dict[player_id].irs = None
                                else:
                                    correct = 0
                                    debug_info = []
                                    for target_id, pred_camp in non_unknown.items():
                                        if target_id in game_state.players:
                                            actual_camp = game_state.players[target_id].camp.value
                                            actual_role = game_state.players[target_id].role.value
                                            is_correct = pred_camp == actual_camp
                                            if is_correct:
                                                correct += 1
                                            debug_info.append(
                                                f"P{target_id}: predicted={pred_camp}, actual_camp={actual_camp} (role={actual_role}) {'✅' if is_correct else '❌'}"
                                            )
                                    
                                    # Debug logging
                                    with open("/tmp/metrics_debug.log", "a") as f:
                                        f.write(f"\n🔍 IRS CALCULATION for Player {player_id}:\n")
                                        for line in debug_info:
                                            f.write(f"  {line}\n")
                                        f.write(f"  RESULT: {correct}/{len(non_unknown)} = {correct / len(non_unknown):.2%}\n")
                                    
                                    player_metrics_dict[player_id].irs = correct / len(non_unknown)
                            
                            # MSS from batch
                            player_metrics_dict[player_id].mss = results.get("mss", 0.5)

                    # If some players are missing from batch_results, mark IRS unavailable for sparse speech.
                    for player_id in game_state.players.keys():
                        if player_id in batch_results:
                            continue
                        player_speeches_count = sum(
                            1
                            for e in game_state.game_history
                            if e.get('type') == 'speech' and e.get('player_id') == player_id
                        )
                        if player_speeches_count < 2:
                            player_metrics_dict[player_id].irs = None
                    
                    # VRS still needs individual calculation (based on voting logic)
                    vrs_tasks = []
                    for player_id in game_state.players.keys():
                        vrs_tasks.append((player_id, self.calculate_vrs(player_id, game_state)))
                    
                    logger.info("📊 Calculating VRS (Voting Rationality) for all players...")
                    vrs_results = await asyncio.gather(*[task for _, task in vrs_tasks], return_exceptions=True)
                    
                    for i, (player_id, _) in enumerate(vrs_tasks):
                        if isinstance(vrs_results[i], Exception):
                            logger.error(f"VRS failed for player {player_id}: {vrs_results[i]}")
                            vrs_results[i] = 0.5
                        player_metrics_dict[player_id].vrs = vrs_results[i]
                    
                    total_time = time.time() - start_time
                    logger.info("="*80)
                    logger.info(f"✅ Batch metrics calculated in {total_time:.1f}s")
                    logger.info(f"   🚀 Speedup: ~{(24 * 3) / max(total_time, 1):.1f}x faster than individual calls")
                    logger.info("="*80)
                    
                except Exception as e:
                    logger.error(f"❌ Batch evaluation failed: {e}", exc_info=True)
                    logger.warning("⚠️  Falling back to individual API calls (this will be slower)")
                    self.use_batch_evaluation = False  # Fallback
            
            if not self.use_batch_evaluation:
                # FALLBACK: Original individual calls
                irs_tasks = []
                vrs_tasks = []
                mss_tasks = []
                
                for player_id in game_state.players.keys():
                    irs_tasks.append((player_id, self.calculate_irs(player_id, game_state)))
                    vrs_tasks.append((player_id, self.calculate_vrs(player_id, game_state)))
                    mss_tasks.append((player_id, self.calculate_mss(player_id, game_state)))
                
                total_api_calls = len(irs_tasks) + len(vrs_tasks) + len(mss_tasks)
                logger.info("="*80)
                logger.info(f"🔄 Individual LLM metrics calculation:")
                logger.info(f"   • Total OpenAI API calls: {total_api_calls}")
                logger.info(f"   ⏱️  Expected time: ~{total_api_calls * 2}-{total_api_calls * 5} seconds")
                logger.info("="*80)
                
                logger.info("📊 Stage 1/3: Calculating IRS...")
                irs_results = await asyncio.gather(*[task for _, task in irs_tasks], return_exceptions=True)
                
                logger.info("📊 Stage 2/3: Calculating VRS...")
                vrs_results = await asyncio.gather(*[task for _, task in vrs_tasks], return_exceptions=True)
                
                logger.info("📊 Stage 3/3: Calculating MSS...")
                mss_results = await asyncio.gather(*[task for _, task in mss_tasks], return_exceptions=True)
                
                # Apply results
                for i, (player_id, _) in enumerate(irs_tasks):
                    if isinstance(irs_results[i], Exception):
                        irs_results[i] = None
                    if isinstance(vrs_results[i], Exception):
                        vrs_results[i] = 0.0
                    if isinstance(mss_results[i], Exception):
                        mss_results[i] = 0.5
                    
                    player_metrics_dict[player_id].irs = irs_results[i]
                    player_metrics_dict[player_id].vrs = vrs_results[i]
                    player_metrics_dict[player_id].mss = mss_results[i]
                
                total_time = time.time() - start_time
                logger.info(f"✅ All metrics calculated in {total_time:.1f}s")
        else:
            # No LLM metrics
            for player_id in player_metrics_dict.keys():
                player_metrics_dict[player_id].irs = None
                player_metrics_dict[player_id].vrs = 0.5
                player_metrics_dict[player_id].mss = 0.5
        
        # Log final metrics
        def _fmt_pct_optional(x: Optional[float]) -> str:
            return "N/A" if x is None else f"{x:.2%}"

        for player_id, player_metrics in player_metrics_dict.items():
            logger.info(
                f"Player {player_id}: IRS={_fmt_pct_optional(player_metrics.irs)}, VRS={player_metrics.vrs:.2%}, MSS={player_metrics.mss:.2%}"
            )
        
        # Calculate persuasion and deception scores for all players (AFTER LLM metrics)
        logger.info("📊 Calculating persuasion and deception scores...")
        persuasion_scores = {}
        deception_scores = {}
        all_speeches = [e for e in game_state.game_history if e.get('type') == 'speech']
        all_votes = [e for e in game_state.game_history if e.get('type') == 'vote']
        
        logger.info(f"Found {len(all_speeches)} speeches and {len(all_votes)} votes")
        
        # Calculate persuasion for all players
        for player_id, player in game_state.players.items():
            try:
                player_speeches = [s for s in all_speeches if s.get('player_id') == player_id]
                logger.info(f"Player {player_id} made {len(player_speeches)} speeches")
                
                if player_speeches:  # Only calculate if player made speeches
                    persuasion_result = await self.advanced.calculate_persuasion_score_v2(
                        player_id=player_id,
                        player_speeches=player_speeches,
                        all_speeches=all_speeches,
                        all_votes=all_votes,
                        players={pid: {'camp': p.camp.value, 'role': p.role.value} for pid, p in game_state.players.items()}
                    )
                    persuasion_scores[player_id] = persuasion_result.value
                    logger.info(f"✓ Player {player_id} persuasion: {persuasion_result.value:.2%} (confidence: {persuasion_result.confidence:.2%})")
                else:
                    persuasion_scores[player_id] = 0.0
                    logger.info(f"✓ Player {player_id} persuasion: 0.0 (no speeches)")
            except Exception as e:
                logger.error(f"❌ Error calculating persuasion for player {player_id}: {e}", exc_info=True)
                persuasion_scores[player_id] = 0.0
        
        # Calculate deception for wolves
        for player_id, player in game_state.players.items():
            if player.camp == Camp.WOLF:
                try:
                    player_speeches = [s for s in all_speeches if s.get('player_id') == player_id]
                    deception_result = await self.advanced.calculate_deception_quality_v2(
                        player_id=player_id,
                        player_speeches=player_speeches,
                        all_speeches=all_speeches,
                        all_votes=all_votes,
                        game_history=game_state.game_history,
                        players={pid: {'camp': p.camp.value, 'role': p.role.value} for pid, p in game_state.players.items()},
                        llm_evaluator=self.llm_evaluator if self.use_llm_metrics else None
                    )
                    deception_scores[player_id] = deception_result.value
                    logger.info(f"✓ Player {player_id} (wolf) deception: {deception_result.value:.2%} (confidence: {deception_result.confidence:.2%})")
                except Exception as e:
                    logger.error(f"❌ Error calculating deception for player {player_id}: {e}", exc_info=True)
                    deception_scores[player_id] = 0.5
            else:
                deception_scores[player_id] = 0.0  # Non-wolves get 0
        
        logger.info(f"✓ Calculated persuasion for {len(persuasion_scores)} players, deception for {len(deception_scores)} players")
        
        # Add persuasion and deception scores to player metrics
        for player_id in player_metrics_dict.keys():
            player_metrics_dict[player_id].persuasion_score = persuasion_scores.get(player_id, 0.0)
            player_metrics_dict[player_id].deception_score = deception_scores.get(player_id, 0.0)
            logger.info(f"Player {player_id}: Persuasion={persuasion_scores.get(player_id, 0.0):.2%}, Deception={deception_scores.get(player_id, 0.0):.2%}")
        
        # Calculate aggregate metrics by camp
        good_irs = []
        good_vrs = []
        wolf_irs = []
        wolf_vrs = []
        all_mss = []
        
        for player_id, metrics in player_metrics_dict.items():
            camp = game_state.players[player_id].camp
            
            if camp == Camp.GOOD:
                if metrics.irs is not None:
                    good_irs.append(metrics.irs)
                good_vrs.append(metrics.vrs)
            else:
                if metrics.irs is not None:
                    wolf_irs.append(metrics.irs)
                wolf_vrs.append(metrics.vrs)
            
            all_mss.append(metrics.mss)
        
        # Calculate advanced metrics (manipulation, deception, etc.)
        advanced_metrics = {}
        try:
            # Get all speeches and votes from game history
            all_speeches = [e for e in game_state.game_history if e.get('type') == 'speech']
            all_votes = [e for e in game_state.game_history if e.get('type') == 'vote']
            
            # Calculate game-level manipulation metrics
            manipulation_metrics = self.advanced.calculate_manipulation_success(
                game_state.game_history,
                {pid: {'camp': p.camp.value, 'role': p.role.value} for pid, p in game_state.players.items()},
                [pid for pid, p in game_state.players.items() if p.camp == Camp.WOLF]
            )
            advanced_metrics.update(manipulation_metrics)
            
            # Calculate auto-sabotage
            advanced_metrics['auto_sabotage'] = self.advanced.calculate_auto_sabotage(
                game_state.game_history,
                {pid: {'camp': p.camp.value, 'role': p.role.value} for pid, p in game_state.players.items()},
                [pid for pid, p in game_state.players.items() if p.camp == Camp.WOLF]
            )
            
            # Calculate day-1 wolf elimination
            advanced_metrics['day1_wolf_eliminated'] = self.advanced.calculate_day1_wolf_elimination(
                game_state.game_history,
                {pid: {'camp': p.camp.value, 'role': p.role.value} for pid, p in game_state.players.items()},
                [pid for pid, p in game_state.players.items() if p.camp == Camp.WOLF]
            )
            
            logger.info(f"Advanced metrics: Manipulation D1={advanced_metrics.get('manipulation_success_d1', 0):.1%}, "
                       f"D2={advanced_metrics.get('manipulation_success_d2', 0):.1%}, "
                       f"Auto-sabotage={advanced_metrics.get('auto_sabotage', 0)}")
            
            # Add deception scores for wolves
            advanced_metrics['deception_scores'] = deception_scores
        except Exception as e:
            logger.warning(f"Error calculating advanced metrics: {e}")
            advanced_metrics = {
                'manipulation_success_d1': 0.0,
                'manipulation_success_d2': 0.0,
                'auto_sabotage': 0.0,
                'day1_wolf_eliminated': 0.0
            }
        
        return GameResult(
            game_id=game_state.game_id,
            winner=game_state.winner,
            total_rounds=game_state.round_number,
            player_metrics=player_metrics_dict,
            role_metrics=role_metrics_dict,
            game_log=[event for event in game_state.game_history],
            avg_irs_good=sum(good_irs) / len(good_irs) if good_irs else 0.0,
            avg_irs_wolf=sum(wolf_irs) / len(wolf_irs) if wolf_irs else 0.0,
            avg_mss=sum(all_mss) / len(all_mss) if all_mss else 0.0,
            avg_vrs_good=sum(good_vrs) / len(good_vrs) if good_vrs else 0.0,
            avg_vrs_wolf=sum(wolf_vrs) / len(wolf_vrs) if wolf_vrs else 0.0,
            advanced_metrics=advanced_metrics
        )
    
    def _extract_role_score(self, role_type: RoleType, metrics: RoleMetrics) -> float:
        """Extract the relevant score for a specific role"""
        role_score_map = {
            RoleType.SEER: metrics.seer_accuracy,
            RoleType.WITCH: metrics.witch_effectiveness,
            RoleType.HUNTER: metrics.hunter_accuracy,
            RoleType.GUARD: metrics.guard_effectiveness,
            RoleType.WEREWOLF: metrics.werewolf_survival,
        }
        return role_score_map.get(role_type)
    
    # Helper methods removed - now handled by specialized calculators

    
    def format_metrics_report(self, game_result: GameResult) -> str:
        """Format a human-readable metrics report"""
        report = []
        report.append(f"=== Game Result: {game_result.game_id} ===")
        report.append(f"Winner: {game_result.winner}")
        report.append(f"Total Rounds: {game_result.total_rounds}")
        report.append("")
        
        # Aggregate summary
        report.append("=== Performance Summary ===")
        report.append(f"Good Camp Reasoning (IRS): {game_result.avg_irs_good:.2%}")
        report.append(f"Wolf Camp Reasoning (IRS): {game_result.avg_irs_wolf:.2%}")
        report.append(f"Good Camp Voting (VRS): {game_result.avg_vrs_good:.2%}")
        report.append(f"Wolf Camp Voting (VRS): {game_result.avg_vrs_wolf:.2%}")
        report.append(f"Overall Speech Realism (MSS): {game_result.avg_mss:.2%}")
        report.append("")
        
        report.append("=== Player Metrics ===")

        def _fmt_pct_optional(x: Optional[float]) -> str:
            return "N/A" if x is None else f"{x:.2%}"

        for player_id, metrics in game_result.player_metrics.items():
            report.append(f"\nPlayer {player_id}:")
            # New Werewolf-Bench style metrics
            report.append(f"  IRS (Identity Recognition Score): {_fmt_pct_optional(metrics.irs)}")
            report.append(f"  VRS (Voting Rationality Score): {metrics.vrs:.2%}")
            report.append(f"  MSS (Message Simulation Score): {metrics.mss:.2%}")
            report.append(f"  SR (Survival Rate): {metrics.sr:.2%}")
            # Legacy metrics
            
        
        report.append("\n=== Role-Specific Metrics ===")
        for player_id, metrics in game_result.role_metrics.items():
            report.append(f"\nPlayer {player_id}:")
            
            if metrics.seer_accuracy is not None:
                report.append(f"  Seer Accuracy: {metrics.seer_accuracy:.2%}")
            if metrics.witch_effectiveness is not None:
                report.append(f"  Witch Effectiveness: {metrics.witch_effectiveness:.2%}")
            if metrics.hunter_accuracy is not None:
                report.append(f"  Hunter Accuracy: {metrics.hunter_accuracy:.2%}")
            if metrics.guard_effectiveness is not None:
                report.append(f"  Guard Effectiveness: {metrics.guard_effectiveness:.2%}")
            if metrics.sheriff_influence is not None:
                report.append(f"  Sheriff Influence: {metrics.sheriff_influence:.2%}")
            if metrics.werewolf_survival is not None:
                report.append(f"  Werewolf Survival: {metrics.werewolf_survival * 100:.2f}%")
        
        return "\n".join(report)