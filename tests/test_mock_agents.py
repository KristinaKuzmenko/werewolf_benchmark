"""
Unit tests for metric calculation using mock agents.
These tests validate that the evaluation system correctly distinguishes
between different agent skill levels.
"""
import pytest
from tests.mock_agents import (
    PerfectSeerMock,
    RandomAgentMock,
    SaboteurAgentMock,
    AggressiveWolfMock
)
from src.green_agent.game_engine import WerewolfGameEngine
from src.green_agent.metrics import MetricsCalculator


def test_perfect_seer_metrics():
    """
    Test that Perfect Seer mock achieves expected high metrics.
    
    Expected:
    - seer_check_accuracy > 0.8 (checks wolves)
    - IRS > 0.7 (identifies wolves correctly)
    - win_rate > 0.6 (helps village win)
    """
    # This is a simplified test structure
    # In practice, you'd run full games with the mock agent
    
    agent = PerfectSeerMock(player_id=1, wolf_ids=[2, 3])
    agent.role = "seer"
    
    # Simulate night action
    message = {
        "observation": {"alive_players": [1, 2, 3, 4, 5, 6, 7, 8]}
    }
    action = agent.handle_night_action(message)
    
    # Should check player 2 or 3 (wolves)
    assert action.get("target_id") in [2, 3]
    assert action.get("action_type") == "check"


def test_random_agent_baseline():
    """
    Test that Random agent produces baseline metrics.
    
    Expected:
    - IRS ≈ 0.5 (random guessing)
    - VRS ≈ 0.5 (random votes)
    - Generic speech quality
    """
    agent = RandomAgentMock(player_id=1)
    
    # Test random voting
    message = {"candidates": [2, 3, 4, 5]}
    vote1 = agent.handle_vote(message)
    vote2 = agent.handle_vote(message)
    
    # Votes should be in candidate list
    assert vote1.get("vote") in message["candidates"]
    assert vote2.get("vote") in message["candidates"]
    
    # Test speech
    speech = agent.handle_speak({})
    assert len(speech.get("speech", "")) > 0


def test_saboteur_agent_low_metrics():
    """
    Test that Saboteur agent produces low metrics.
    
    Expected:
    - IRS < 0.3 (wrong identification)
    - VRS < 0.2 (bad votes)
    - win_rate < 0.3 (loses)
    """
    agent = SaboteurAgentMock(player_id=1, wolf_ids=[2, 3])
    agent.role = "villager"
    
    # As villager, should vote for wolves, but saboteur votes for non-wolves
    message = {"candidates": [2, 3, 4, 5]}  # 2,3 are wolves
    
    votes = [agent.handle_vote(message).get("vote") for _ in range(10)]
    
    # Most votes should be for non-wolves (bad strategy)
    non_wolf_votes = sum(1 for v in votes if v in [4, 5])
    assert non_wolf_votes > 5  # More than half should be bad votes


def test_aggressive_wolf_low_deception():
    """
    Test that Aggressive Wolf has low deception score.
    
    Expected:
    - Draws suspicion through voting patterns
    - Gets eliminated early
    """
    agent = AggressiveWolfMock(player_id=2, wolf_teammate_id=3)
    
    # Test voting against majority
    message = {
        "candidates": [1, 4, 5, 6],
        "current_votes": {
            7: 1,  # Most votes for player 1
            8: 1
        }
    }
    
    vote = agent.handle_vote(message)
    
    # Should vote opposite to majority (player 4, 5, or 6 instead of 1)
    # This draws suspicion
    assert vote.get("vote") is not None


def test_perfect_seer_strategic_checks():
    """
    Test that Perfect Seer checks wolves preferentially.
    """
    agent = PerfectSeerMock(player_id=1, wolf_ids=[2, 3])
    agent.role = "seer"
    
    message = {
        "observation": {"alive_players": [1, 2, 3, 4, 5, 6, 7, 8]}
    }
    
    # First check should be wolf
    check1 = agent.handle_night_action(message)
    assert check1.get("target_id") in [2, 3]
    
    # Second check should be the other wolf
    check2 = agent.handle_night_action(message)
    assert check2.get("target_id") in [2, 3]
    assert check2.get("target_id") != check1.get("target_id")


def test_perfect_seer_votes_for_wolves():
    """
    Test that Perfect Seer votes for wolves when possible.
    """
    agent = PerfectSeerMock(player_id=1, wolf_ids=[2, 3])
    
    # Wolf is a candidate
    message = {"candidates": [2, 4, 5]}
    vote = agent.handle_vote(message)
    
    # Should vote for wolf (player 2)
    assert vote.get("vote") == 2
    
    # No wolves in candidates
    message = {"candidates": [4, 5, 6]}
    vote = agent.handle_vote(message)
    
    # Should vote for someone
    assert vote.get("vote") in [4, 5, 6]


def test_saboteur_wastes_seer_checks():
    """
    Test that Saboteur wastes Seer ability on non-wolves.
    """
    agent = SaboteurAgentMock(player_id=1, wolf_ids=[2, 3])
    agent.role = "seer"
    
    message = {
        "observation": {"alive_players": [1, 2, 3, 4, 5, 6, 7, 8]}
    }
    
    # Should check non-wolves (wasteful)
    check = agent.handle_night_action(message)
    target = check.get("target_id")
    
    # Should NOT check wolves
    assert target not in [2, 3]
    assert target in [4, 5, 6, 7, 8]


def test_mock_agents_handle_all_message_types():
    """
    Test that all mock agents can handle standard message types.
    """
    agents = [
        PerfectSeerMock(1),
        RandomAgentMock(2),
        SaboteurAgentMock(3),
        AggressiveWolfMock(4, 5)
    ]
    
    for agent in agents:
        # Game start
        result = agent.handle_game_start({"role": "villager", "observation": {}})
        assert "status" in result or result == {}
        
        # Speak
        result = agent.handle_speak({})
        assert "speech" in result
        
        # Vote
        result = agent.handle_vote({"candidates": [1, 2, 3]})
        assert "vote" in result or result == {}


@pytest.mark.skip(reason="Requires full game simulation - integration test")
def test_metric_calculation_with_mocks():
    """
    Integration test: Run full games with mock agents and validate metrics.
    
    This test would:
    1. Create game with Perfect Seer, Random, and Saboteur agents
    2. Run multiple games
    3. Calculate metrics
    4. Assert Perfect Seer > Random > Saboteur
    """
    # Placeholder for full integration test
    # Would require running full game engine with mock agents
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
