"""
Pydantic models for Werewolf game state and A2A protocol messages.
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class RoleType(str, Enum):
    """Player roles in the game"""
    WEREWOLF = "werewolf"
    VILLAGER = "villager"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    GUARD = "guard"


class Camp(str, Enum):
    """Player camps"""
    GOOD = "good"
    WOLF = "wolf"


class Phase(str, Enum):
    """Game phases"""
    NIGHT = "night"
    DAY = "day"
    GAME_OVER = "game_over"


class ActionType(str, Enum):
    """Types of actions players can take"""
    KILL = "kill"
    PROTECT = "protect"
    CHECK = "check"
    HEAL = "heal"
    POISON = "poison"
    SHOOT = "shoot"
    VOTE = "vote"
    SPEAK = "speak"
    ELECT_SHERIFF = "elect_sheriff"


class PlayerInfo(BaseModel):
    """Information about a player"""
    player_id: int
    role: RoleType
    camp: Camp
    is_alive: bool = True
    is_sheriff: bool = False
    endpoint: str  # A2A endpoint for this player


class Action(BaseModel):
    """Action taken by a player"""
    action_type: ActionType
    player_id: int
    target_id: Optional[int] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Enhanced reasoning (for logging and UI)
    private_thoughts: Optional[str] = None
    suspicions: Optional[Dict[int, float]] = None  # player_id -> suspicion (0-1)
    strategy: Optional[str] = None  # e.g., "reveal_seer", "deflect", "accuse"
    confidence: Optional[float] = None  # 0-1
    
    # Bidding system
    bid_value: Optional[int] = None  # 0-100, urgency to speak
    bid_reasoning: Optional[str] = None


class Observation(BaseModel):
    """Observation/information given to a player"""
    phase: Phase
    round_number: int
    night_result: Optional[str] = None
    alive_players: List[int]
    eliminated_players: List[int]
    speaker_order: Optional[List[int]] = None
    votes: Optional[Dict[int, int]] = None  # voter_id -> target_id
    sheriff_id: Optional[int] = None
    role_specific_info: Dict[str, Any] = Field(default_factory=dict)


class GameState(BaseModel):
    """Complete game state"""
    game_id: str
    round_number: int
    phase: Phase
    players: Dict[int, PlayerInfo]
    alive_players: List[int]
    eliminated_players: List[int]
    sheriff_id: Optional[int] = None
    game_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Agent name mapping for leaderboard
    player_agent_names: Dict[int, str] = Field(default_factory=dict)
    
    # Role-specific state
    witch_heal_used: bool = False
    witch_poison_used: bool = False
    last_protected: Optional[int] = None  # For guard consecutive protection rule
    
    # Victory condition
    winner: Optional[Camp] = None


class NightActions(BaseModel):
    """Actions that occur during the night"""
    werewolf_target: Optional[int] = None
    guard_target: Optional[int] = None
    guard_id: Optional[int] = None  # ID of guard who made protection
    seer_target: Optional[int] = None
    witch_heal: Optional[bool] = None
    witch_poison_target: Optional[int] = None


class DayActions(BaseModel):
    """Actions that occur during the day"""
    speeches: Dict[int, str] = Field(default_factory=dict)
    votes: Dict[int, int] = Field(default_factory=dict)  # voter_id -> target_id
    sheriff_recommendation: Optional[int] = None
    hunter_shoot: Optional[int] = None


# A2A Protocol Models
class AgentCard(BaseModel):
    """A2A Agent Card"""
    agent_id: str
    name: str
    description: str
    version: str
    capabilities: List[str]
    endpoint: str


class TaskRequest(BaseModel):
    """A2A Task Request"""
    task_id: str
    participants: Dict[str, str]  # role_name -> endpoint
    config: Dict[str, Any]


class TaskUpdate(BaseModel):
    """A2A Task Update"""
    task_id: str
    update_type: str
    content: Any
    timestamp: str


class TaskResult(BaseModel):
    """A2A Task Result (Artifact)"""
    task_id: str
    result_type: str = "assessment_result"
    metrics: Dict[str, Any]
    game_summary: Dict[str, Any]
    player_performances: Dict[str, Dict[str, Any]]  # Changed from float to Any


# Evaluation Models
class RoleMetrics(BaseModel):
    """Role-specific performance metrics"""
    seer_accuracy: float = 0.0  # investigated werewolf / total werewolves
    witch_effectiveness: float = 0.0  # correct potions / total used
    hunter_accuracy: float = 0.0  # shot werewolf / total shots
    guard_effectiveness: float = 0.0  # successful protections
    sheriff_influence: float = 0.0  # vote changes following recommendation
    werewolf_survival: float = 0.0  # survival score


class PlayerMetrics(BaseModel):
    """Player-level performance metrics"""
    # Werewolf-Bench style metrics
    irs: Optional[float] = None  # Identity Recognition Score (LLM-based reasoning); None if not enough evidence
    vrs: float = 0.0  # Voting Rationality Score
    mss: float = 0.0  # Message Simulation Score (human-likeness)
    sr: float = 0.0   # Survival Rate
    
    # Advanced social metrics
    persuasion_score: float = 0.0  # Persuasion effectiveness
    deception_score: float = 0.0   # Deception quality (wolves only)

    games_survived: int = 0
    total_games: int = 0


class GameResult(BaseModel):
    """Complete game result with all metrics"""
    game_id: str
    winner: Camp
    total_rounds: int
    player_metrics: Dict[int, PlayerMetrics]
    role_metrics: Dict[int, RoleMetrics]
    game_log: List[Dict[str, Any]]
    
    # Aggregate metrics
    avg_irs_good: float = 0.0  # Average IRS for good camp
    avg_irs_wolf: float = 0.0  # Average IRS for wolf camp
    avg_mss: float = 0.0        # Average MSS across all players
    avg_vrs_good: float = 0.0  # Average VRS for good camp
    avg_vrs_wolf: float = 0.0  # Average VRS for wolf camp
    
    # Advanced metrics (Foaster.ai inspired)
    advanced_metrics: Dict[str, Any] = {}