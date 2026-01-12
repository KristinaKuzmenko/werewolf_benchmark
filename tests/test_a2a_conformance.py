"""
A2A Conformance tests for Werewolf Green Agent.
Validates that agent follows A2A protocol specification.
"""
from typing import Any
import pytest
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


# A2A validation helpers - adapted from https://github.com/a2aproject/a2a-inspector

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and fields of an agent card."""
    errors: list[str] = []

    required_fields = frozenset([
        'name', 'description', 'url', 'version',
        'capabilities', 'defaultInputModes', 'defaultOutputModes', 'skills',
    ])

    # Check for presence of all required fields
    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    # Check if 'url' is an absolute URL
    if 'url' in card_data and not (
        card_data['url'].startswith('http://') or card_data['url'].startswith('https://')
    ):
        errors.append("Field 'url' must be an absolute URL starting with http:// or https://.")

    # Check if capabilities is a dictionary
    if 'capabilities' in card_data and not isinstance(card_data['capabilities'], dict):
        errors.append("Field 'capabilities' must be an object.")

    # Check defaultInputModes and defaultOutputModes are arrays of strings
    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    # Check skills array
    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append("Field 'skills' must be an array of AgentSkill objects.")
        elif not card_data['skills']:
            errors.append("Field 'skills' array is empty. Agent must have at least one skill.")

    return errors


def _validate_task(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'id' not in data:
        errors.append("Task object missing required field: 'id'.")
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("Task object missing required field: 'status.state'.")
    return errors


def _validate_status_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("StatusUpdate object missing required field: 'status.state'.")
    return errors


def _validate_artifact_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'artifact' not in data:
        errors.append("ArtifactUpdate object missing required field: 'artifact'.")
    elif (
        'parts' not in data.get('artifact', {})
        or not isinstance(data.get('artifact', {}).get('parts'), list)
        or not data.get('artifact', {}).get('parts')
    ):
        errors.append("Artifact object must have a non-empty 'parts' array.")
    return errors


def _validate_message(data: dict[str, Any]) -> list[str]:
    errors = []
    if (
        'parts' not in data
        or not isinstance(data.get('parts'), list)
        or not data.get('parts')
    ):
        errors.append("Message object must have a non-empty 'parts' array.")
    if 'role' not in data or data.get('role') != 'agent':
        errors.append("Message from agent must have 'role' set to 'agent'.")
    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    """Validate an incoming event from the agent based on its kind."""
    if 'kind' not in data:
        return ["Response from agent is missing required 'kind' field."]

    kind = data.get('kind')
    validators = {
        'task': _validate_task,
        'status-update': _validate_status_update,
        'artifact-update': _validate_artifact_update,
        'message': _validate_message,
    }

    validator = validators.get(str(kind))
    if validator:
        return validator(data)

    return [f"Unknown message kind received: '{kind}'."]


# A2A messaging helpers

async def send_text_message(
    text: str, url: str, context_id: str | None = None, streaming: bool = False
):
    """Send a text message to an A2A agent and collect events."""
    async with httpx.AsyncClient(timeout=10) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]

    return events


# A2A conformance tests

def test_agent_card(agent):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)
    
    # Werewolf-specific validations
    assert card_data['name'] == "Werewolf Game Orchestrator"
    assert 'werewolf' in card_data['description'].lower() or 'game' in card_data['description'].lower()
    assert len(card_data['skills']) >= 1, "Agent must have at least one skill"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent, streaming):
    """Test that agent returns valid A2A message format."""
    # Simple test message (will likely be rejected as invalid game request, but structure should be valid)
    events = await send_text_message("test", agent, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)


# Werewolf-specific tests

@pytest.mark.asyncio
async def test_invalid_request_rejected(agent):
    """Test that invalid game requests are properly rejected."""
    invalid_request = '{"invalid": "request"}'
    events = await send_text_message(invalid_request, agent)
    
    # Should receive rejection
    assert events, "Agent should respond to invalid requests"
    
    # Check for rejection or error state
    has_rejection = False
    for event in events:
        match event:
            case (task, update):
                if task.status.state.value in ['rejected', 'failed']:
                    has_rejection = True
            case _:
                pass
    
    assert has_rejection, "Invalid request should be rejected or failed"



