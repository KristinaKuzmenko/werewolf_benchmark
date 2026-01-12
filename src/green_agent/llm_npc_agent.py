"""
LLM NPC Agent for Werewolf Benchmark

An NPC that uses LLM for:
- Reading and understanding other players' speeches
- Generating contextual responses
- Making informed decisions based on conversation

Designed to work with Groq free tier (30 req/min).
Includes rate limiting, retries, and fallback to rule-based.
"""

import asyncio
import json
import logging
import os
import random
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RoleType(str, Enum):
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    GUARD = "guard"
    VILLAGER = "villager"


@dataclass
class NPCState:
    """State for LLM NPC."""
    player_id: int
    role: RoleType
    camp: str
    game_id: str
    
    # Game state
    alive_players: List[int] = field(default_factory=list)
    round_number: int = 0
    
    # Knowledge
    wolf_team: List[int] = field(default_factory=list)
    seer_results: Dict[int, bool] = field(default_factory=dict)
    witch_heal_used: bool = False
    witch_poison_used: bool = False
    last_protected: Optional[int] = None
    
    # Conversation history (for context)
    conversation_history: List[Dict] = field(default_factory=list)
    my_previous_statements: List[str] = field(default_factory=list)


class LLMClient:
    """Async LLM client with rate limiting and retries."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        requests_per_minute: int = 25,  # Stay under 30 limit
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or self._get_api_key()
        self.requests_per_minute = requests_per_minute
        
        # Rate limiting
        self._request_times: List[float] = []
        self._lock = asyncio.Lock()
        
        # HTTP client
        self._client = None
    
    def _get_api_key(self) -> str:
        """Get API key from environment."""
        if self.provider == "groq":
            return os.getenv("GROQ_API_KEY", "")
        elif self.provider == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        return ""
    
    async def _ensure_client(self):
        """Ensure HTTP client exists."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=30.0)
    
    async def _wait_for_rate_limit(self):
        """Wait if we're at rate limit."""
        async with self._lock:
            import time
            now = time.time()
            
            # Remove old requests (older than 60 seconds)
            self._request_times = [t for t in self._request_times if now - t < 60]
            
            # If at limit, wait
            if len(self._request_times) >= self.requests_per_minute:
                wait_time = 60 - (now - self._request_times[0]) + 0.5
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
            
            self._request_times.append(time.time())
    
    async def complete(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 600,
        temperature: float = 0.7,
        retries: int = 3
    ) -> str:
        """Get completion from LLM."""
        # Check if event loop is closed before attempting async operations
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                logger.error("Event loop is closed, cannot make LLM request")
                raise RuntimeError("Event loop is closed")
        except RuntimeError as e:
            if "no running event loop" in str(e).lower() or "no current event loop" in str(e).lower():
                logger.error("No event loop available for LLM request")
                raise RuntimeError("No event loop available")
            raise
        
        await self._ensure_client()
        
        for attempt in range(retries):
            try:
                await self._wait_for_rate_limit()
                
                if self.provider == "groq":
                    return await self._groq_complete(prompt, system, max_tokens, temperature)
                elif self.provider == "openai":
                    return await self._openai_complete(prompt, system, max_tokens, temperature)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
                    
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    logger.debug("Event loop closed during LLM request")
                    return ""
                raise
            except Exception as e:
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        
        return ""
    
    async def _groq_complete(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Groq API call."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self._client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def _openai_complete(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """OpenAI API call."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self._client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()


class LLMNPCAgent:
    """
    LLM-powered NPC for social deduction.
    
    Features:
    - Reads and analyzes other players' speeches
    - Maintains conversation context
    - Generates contextual responses
    - Falls back to rule-based on LLM failure
    """
    
    SYSTEM_PROMPT = """You are playing Werewolf (Mafia) as Player {player_id}.
Your role is: {role}
Your goal: {goal}

Game rules:
- Werewolves kill one villager each night
- Villagers vote to eliminate one player each day
- Special roles: Seer (check identities), Witch (heal/poison), Guard (protect), Hunter (shoot when dying)
- Werewolves win when they equal or outnumber villagers
- Villagers win when all werewolves are eliminated

Your personality: {personality}

IMPORTANT:
- Stay in character
- Be concise (1-3 sentences)
- Reference specific players by number (e.g., "Player 3")
- React to what others have said
- {role_instruction}"""

    PERSONALITIES = [
        "analytical and logical, you focus on voting patterns",
        "suspicious and cautious, you question everyone",
        "friendly but observant, you build alliances",
        "quiet but perceptive, you speak when you have evidence",
        "aggressive and direct, you push for quick decisions",
    ]
    
    ROLE_INSTRUCTIONS = {
        "werewolf": "Deflect suspicion, accuse villagers subtly, don't defend your wolf teammates too obviously",
        "seer": "Share information strategically, don't reveal yourself too early",
        "witch": "Use your powers wisely, don't reveal you're the witch",
        "guard": "Protect valuable players, stay hidden",
        "hunter": "Play as villager, remember you can shoot when eliminated",
        "villager": "Find wolves through discussion and voting patterns",
    }
    
    ROLE_GOALS = {
        "werewolf": "Eliminate villagers without being detected",
        "seer": "Find werewolves and help village eliminate them",
        "witch": "Use heal to save villagers, poison to kill wolves",
        "guard": "Protect important villagers from wolf attacks",
        "hunter": "Help village, shoot a wolf if you're eliminated",
        "villager": "Find and vote out all werewolves",
    }
    
    def __init__(
        self,
        agent_id: str = "llm-npc",
        provider: str = "groq",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        personality: str = None
    ):
        self.agent_id = agent_id
        self.llm = LLMClient(provider=provider, model=model, api_key=api_key)
        self.personality = personality or random.choice(self.PERSONALITIES)
        self.state: Optional[NPCState] = None
        
        # Fallback agent for when LLM fails
        self._fallback_agent = None
    
    def _get_fallback(self):
        """Get fallback rule-based agent."""
        if self._fallback_agent is None:
            from .ncp_agent import SmartBaselineAgent
            self._fallback_agent = SmartBaselineAgent(agent_id=f"{self.agent_id}-fallback")
        return self._fallback_agent
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for current role."""
        role = self.state.role.value
        return self.SYSTEM_PROMPT.format(
            player_id=self.state.player_id,
            role=role.upper() if role == "werewolf" else role.capitalize(),
            goal=self.ROLE_GOALS.get(role, "Help your team win"),
            personality=self.personality,
            role_instruction=self.ROLE_INSTRUCTIONS.get(role, "Play to win")
        )
    
    def _build_context(self, max_events: int = 15) -> str:
        """Build conversation context from recent events."""
        if not self.state.conversation_history:
            return "No discussion yet."
        
        recent = self.state.conversation_history[-max_events:]
        context_lines = []
        
        for event in recent:
            event_type = event.get("type")
            
            if event_type == "speech":
                pid = event.get("player_id")
                text = event.get("speech", event.get("text", ""))
                if pid == self.state.player_id:
                    context_lines.append(f"You said: \"{text}\"")
                else:
                    context_lines.append(f"Player {pid}: \"{text}\"")
            
            elif event_type == "vote":
                voter = event.get("voter_id")
                target = event.get("target_id")
                if voter == self.state.player_id:
                    context_lines.append(f"You voted for Player {target}")
                else:
                    context_lines.append(f"Player {voter} voted for Player {target}")
            
            elif event_type == "elimination":
                pid = event.get("player_id")
                role = event.get("role", "unknown")
                phase = event.get("phase", "day")
                if phase == "night":
                    context_lines.append(f"Player {pid} was killed by wolves (was {role})")
                else:
                    context_lines.append(f"Player {pid} was eliminated by vote (was {role})")
        
        return "\n".join(context_lines) if context_lines else "No discussion yet."
    
    def _add_role_knowledge(self) -> str:
        """Add role-specific knowledge to prompt."""
        lines = []
        
        if self.state.role == RoleType.WEREWOLF:
            teammates = [p for p in self.state.wolf_team if p != self.state.player_id]
            if teammates:
                lines.append(f"Your werewolf teammates: {teammates}")
        
        elif self.state.role == RoleType.SEER:
            if self.state.seer_results:
                for pid, is_wolf in self.state.seer_results.items():
                    result = "WEREWOLF" if is_wolf else "NOT a werewolf"
                    lines.append(f"You checked Player {pid}: {result}")
        
        elif self.state.role == RoleType.WITCH:
            if self.state.witch_heal_used:
                lines.append("You have already used your heal potion")
            if self.state.witch_poison_used:
                lines.append("You have already used your poison potion")
        
        return "\n".join(lines) if lines else ""
    
    # =========================================================================
    # MESSAGE HANDLERS
    # =========================================================================
    
    def handle_game_start(self, message: Dict) -> Dict:
        """Initialize NPC state."""
        self.state = NPCState(
            player_id=message.get("player_id"),
            role=RoleType(message.get("role", "villager")),
            camp=message.get("camp", "good"),
            game_id=message.get("game_id", ""),
            alive_players=message.get("alive_players", []),
            wolf_team=message.get("werewolf_team", [])
        )
        
        # Initialize fallback agent too
        fallback = self._get_fallback()
        fallback.handle_game_start(message)
        
        logger.info(f"LLM NPC {self.state.player_id} initialized as {self.state.role.value}")
        return {"status": "initialized"}
    
    async def handle_speak_async(self, message: Dict) -> Dict:
        """Generate speech using LLM."""
        observation = message.get("observation", {})
        self._update_state(observation)
        
        # Build prompt
        system = self._build_system_prompt()
        context = self._build_context()
        role_knowledge = self._add_role_knowledge()
        
        prompt = f"""Current game state:
- Round: {self.state.round_number}
- Alive players: {self.state.alive_players}
{role_knowledge}

Recent discussion:
{context}

It's your turn to speak. What do you say? Be concise (1-3 sentences).
React to what others have said and express your opinion about who might be a werewolf."""
        
        try:
            response = await self.llm.complete(prompt, system=system, max_tokens=150)
            logger.debug(f"Player {self.agent_id} raw LLM response: '{response}'")
            
            speech = self._clean_speech(response)
            logger.debug(f"Player {self.agent_id} cleaned speech: '{speech}'")
            
            # Check if speech is empty
            if not speech or not speech.strip():
                logger.warning(f"Player {self.agent_id} (LLM NPC) generated empty speech. Raw response: '{response}'. Using fallback.")
                return self._get_fallback().handle_speak(message)
            
            # Track our statements
            self.state.my_previous_statements.append(speech)
            
            logger.info(f"Player {self.agent_id} (LLM NPC) speech: {speech}")
            
            return {"speech": speech}
            
        except Exception as e:
            logger.error(f"Player {self.agent_id} LLM speech failed: {e}", exc_info=True)
            return self._get_fallback().handle_speak(message)
    
    def handle_speak(self, message: Dict) -> Dict:
        """Sync wrapper for speak."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in async context
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.handle_speak_async(message))
                    return future.result(timeout=30)
            else:
                return asyncio.run(self.handle_speak_async(message))
        except Exception as e:
            logger.warning(f"Async speak failed: {e}")
            return self._get_fallback().handle_speak(message)
    
    async def handle_vote_async(self, message: Dict) -> Dict:
        """Choose vote target using LLM."""
        observation = message.get("observation", {})
        candidates = message.get("candidates", [])
        current_votes = message.get("current_votes", {})
        
        self._update_state(observation)
        
        if not candidates:
            return {"vote": self.state.player_id}
        
        # Build prompt
        system = self._build_system_prompt()
        context = self._build_context()
        role_knowledge = self._add_role_knowledge()
        
        # Format current votes
        vote_info = ""
        if current_votes:
            vote_lines = [f"Player {v} voted for Player {t}" for v, t in current_votes.items()]
            vote_info = "Current votes:\n" + "\n".join(vote_lines)
        
        prompt = f"""Current game state:
- Round: {self.state.round_number}
- Alive players: {self.state.alive_players}
- You can vote for: {candidates}
{role_knowledge}

Recent discussion:
{context}

{vote_info}

Based on the discussion and your role, who do you vote to eliminate?
Reply with ONLY the player number (e.g., "3" or "Player 3")."""
        
        try:
            response = await self.llm.complete(prompt, system=system, max_tokens=20)
            vote = self._parse_player_id(response, candidates)
            logger.info(f"Player {self.agent_id} (LLM NPC) vote: Player {vote}")
            return {"vote": vote}
            
        except Exception as e:
            logger.warning(f"LLM vote failed, using fallback: {e}")
            return self._get_fallback().handle_vote(message)
    
    def handle_vote(self, message: Dict) -> Dict:
        """Sync wrapper for vote."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.handle_vote_async(message))
                    return future.result(timeout=30)
            else:
                return asyncio.run(self.handle_vote_async(message))
        except Exception as e:
            logger.warning(f"Async vote failed: {e}")
            return self._get_fallback().handle_vote(message)
    
    async def handle_night_action_async(self, message: Dict) -> Dict:
        """Choose night action using LLM."""
        observation = message.get("observation", {})
        role_info = observation.get("role_specific_info", {})
        
        self._update_state(observation)
        
        # Process role-specific info
        if self.state.role == RoleType.SEER and "check_result" in role_info:
            checked = role_info.get("checked_player")
            is_wolf = role_info.get("check_result")
            if checked is not None:
                self.state.seer_results[checked] = is_wolf
        
        # Build prompt based on role
        action = await self._get_night_action_llm(role_info)
        
        # Log action
        if action:
            logger.info(f"Player {self.agent_id} (LLM NPC) night action: {action.get('action_type')} → target={action.get('target_id')}")
        
        # Update fallback too
        self._get_fallback().handle_night_action(message)
        
        return {"action": action}
    
    async def _get_night_action_llm(self, role_info: Dict) -> Optional[Dict]:
        """Get night action from LLM."""
        role = self.state.role
        
        if role == RoleType.WEREWOLF:
            return await self._wolf_night_action()
        elif role == RoleType.SEER:
            return await self._seer_night_action()
        elif role == RoleType.GUARD:
            return await self._guard_night_action()
        elif role == RoleType.WITCH:
            return await self._witch_night_action(role_info)
        
        return None
    
    async def _wolf_night_action(self) -> Dict:
        """Wolf chooses kill target."""
        candidates = [p for p in self.state.alive_players 
                     if p not in self.state.wolf_team]
        
        if not candidates:
            return None
        
        system = self._build_system_prompt()
        context = self._build_context()
        
        prompt = f"""Night phase - you must choose someone to kill.

Recent discussion:
{context}

Alive non-wolf players: {candidates}

Who is the biggest threat to the werewolves? Consider:
- Who suspects the wolves?
- Who might be the Seer?
- Who is leading the village?

Reply with ONLY the player number to kill."""
        
        try:
            response = await self.llm.complete(prompt, system=system, max_tokens=20)
            target = self._parse_player_id(response, candidates)
            return {
                "action_type": "kill",
                "player_id": self.state.player_id,
                "target_id": target
            }
        except Exception:
            target = random.choice(candidates)
            return {
                "action_type": "kill",
                "player_id": self.state.player_id,
                "target_id": target
            }
    
    async def _seer_night_action(self) -> Dict:
        """Seer chooses check target."""
        candidates = [p for p in self.state.alive_players 
                     if p != self.state.player_id and p not in self.state.seer_results]
        
        if not candidates:
            return None
        
        system = self._build_system_prompt()
        context = self._build_context()
        
        checked_info = ""
        if self.state.seer_results:
            lines = [f"Player {p}: {'WOLF' if w else 'not wolf'}" 
                    for p, w in self.state.seer_results.items()]
            checked_info = "Already checked:\n" + "\n".join(lines)
        
        prompt = f"""Night phase - you can check one player's identity.

Recent discussion:
{context}

{checked_info}

Unchecked players: {candidates}

Who is most suspicious and worth checking?
Reply with ONLY the player number."""
        
        try:
            response = await self.llm.complete(prompt, system=system, max_tokens=20)
            target = self._parse_player_id(response, candidates)
            return {
                "action_type": "check",
                "player_id": self.state.player_id,
                "target_id": target
            }
        except Exception:
            target = random.choice(candidates)
            return {
                "action_type": "check",
                "player_id": self.state.player_id,
                "target_id": target
            }
    
    async def _guard_night_action(self) -> Dict:
        """Guard chooses protect target."""
        candidates = [p for p in self.state.alive_players 
                     if p != self.state.player_id and p != self.state.last_protected]
        
        if not candidates:
            return None
        
        target = random.choice(candidates)  # Guard uses simple logic
        self.state.last_protected = target
        
        return {
            "action_type": "protect",
            "player_id": self.state.player_id,
            "target_id": target
        }
    
    async def _witch_night_action(self, role_info: Dict) -> Optional[Dict]:
        """Witch chooses heal or poison."""
        victim = role_info.get("werewolf_victim")
        heal_available = role_info.get("heal_available", not self.state.witch_heal_used)
        poison_available = role_info.get("poison_available", not self.state.witch_poison_used)
        
        # Simple logic: heal if victim seems good, poison most suspicious
        if heal_available and victim and victim != self.state.player_id:
            self.state.witch_heal_used = True
            return {
                "action_type": "heal",
                "player_id": self.state.player_id,
                "target_id": victim
            }
        
        return None
    
    def handle_night_action(self, message: Dict) -> Dict:
        """Sync wrapper for night action."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.handle_night_action_async(message))
                    return future.result(timeout=30)
            else:
                return asyncio.run(self.handle_night_action_async(message))
        except Exception as e:
            logger.warning(f"Async night action failed: {e}")
            return self._get_fallback().handle_night_action(message)
    
    def handle_hunter_shoot(self, message: Dict) -> Dict:
        """Hunter shoots - use fallback logic."""
        return self._get_fallback().handle_hunter_shoot(message)
    
    def handle_sheriff_election(self, message: Dict) -> Dict:
        """Sheriff vote - use fallback logic."""
        return self._get_fallback().handle_sheriff_election(message)
    
    def handle_sheriff_summary(self, message: Dict) -> Dict:
        """Sheriff makes recommendation - use fallback logic."""
        return self._get_fallback().handle_sheriff_summary(message)
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _update_state(self, observation: Dict):
        """Update state from observation."""
        if "alive_players" in observation:
            self.state.alive_players = observation["alive_players"]
        if "round_number" in observation:
            self.state.round_number = observation["round_number"]
        
        # Add recent events to conversation history
        for event in observation.get("recent_events", []):
            if event not in self.state.conversation_history:
                self.state.conversation_history.append(event)
    
    def _clean_speech(self, response: str) -> str:
        """Clean LLM response for speech."""
        # Remove quotes
        response = response.strip().strip('"\'')
        
        # Remove "I say:" or similar prefixes
        prefixes = ["i say:", "i would say:", "my response:", "speech:"]
        for prefix in prefixes:
            if response.lower().startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Limit length
        if len(response) > 600:
            response = response[:597] + "..."
        
        return response
    
    def _parse_player_id(self, response: str, candidates: List[int]) -> int:
        """Parse player ID from LLM response."""
        # Find numbers in response
        numbers = re.findall(r'\d+', response)
        
        for num_str in numbers:
            num = int(num_str)
            if num in candidates:
                return num
        
        # Fallback to random
        return random.choice(candidates) if candidates else 0
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.llm.close()