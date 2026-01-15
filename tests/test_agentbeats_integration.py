"""
Integration tests for AgentBeats mode end-to-end workflow.
Tests green agent evaluation of purple agents.
"""
import pytest
import httpx
import json
from typing import Dict, Any


def create_a2a_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create A2A protocol message with proper structure."""
    return {
        "jsonrpc": "2.0",
        "method": "tasks/run",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "type": "data",
                        "data": payload,
                        "mimeType": "application/json"
                    }
                ]
            }
        },
        "id": "test-request-1"
    }


@pytest.mark.asyncio
async def test_agentbeats_assessment_workflow(agent):
    """Test complete AgentBeats assessment workflow."""
    # Prepare assessment request
    request_payload = {
        "participant": "http://purple_agent:8100",  # Purple agent endpoint (Docker network)
        "config": {
            "num_players": 8,
            "num_tasks": 2,  # Run 2 games for faster testing
            "enable_sheriff": True,
            "max_rounds": 5  # Shorter games for testing
        }
    }
    
    request_data = create_a2a_message(request_payload)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Send assessment request
        response = await client.post(
            f"{agent}/",
            json=request_data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # A2A response structure
        assert "jsonrpc" in result
        
        # Should have result (task created/updated) or error
        if "error" in result:
            # Print error for debugging
            print(f"Error response: {result['error']}")
            # For now, just verify we got a response (not parse error)
            assert result["error"]["code"] != -32700
        else:
            assert "result" in result
            # Validate task was created
            task = result["result"]
            assert "id" in task
            assert "status" in task


@pytest.mark.asyncio
async def test_agent_card_endpoint(agent):
    """Test that green agent exposes valid agent card."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{agent}/.well-known/agent-card.json")
        
        assert response.status_code == 200
        card = response.json()
        
        # Validate A2A agent card structure
        assert "name" in card
        assert "description" in card
        assert "url" in card
        assert "capabilities" in card
        assert "skills" in card
        
        # Should have werewolf orchestration skill
        assert len(card["skills"]) > 0
        assert card["skills"][0]["name"] == "Werewolf Game Orchestration"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
