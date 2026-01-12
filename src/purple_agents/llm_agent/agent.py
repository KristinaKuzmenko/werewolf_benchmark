"""
LLM Werewolf Agent - Game logic implementation.
LLM-powered strategy with enhanced reasoning and bidding.
"""
import json
import logging
import os
import random
import httpx
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import Message
from a2a.utils import get_message_text

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class LLMWerewolfAgent:
    """
    LLM-powered purple agent with enhanced reasoning and bidding.
    """
    
    def __init__(
        self,
        agent_id: str = "llm-agent",
        provider: str = "openai",
        model: str = "gpt-4o-mini"
    ):
        self.agent_id = agent_id
        self.provider = provider
        self.model = model
        
        # Agent state
        self.player_id: Optional[int] = None
        self.role: Optional[str] = None
        self.camp: Optional[str] = None
        self.game_id: Optional[str] = None
        self.personality: Optional[str] = None
        
        # Enhanced state
        self.suspicions: Dict[int, float] = {}
        self.werewolf_team: List[int] = []
        self.alive_players: List[int] = []
        self.conversation_history: List[Dict] = []
        
        logger.info(
            f"LLM Agent '{agent_id}' initialized (provider={provider}, model={model})"
        )
    
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Process incoming A2A message and respond.
        
        Args:
            message: A2A Message with JSON-encoded game message
            updater: TaskUpdater for sending response
        """
        input_text = get_message_text(message)
        
        try:
            game_message = json.loads(input_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
            response = {"status": "error", "message": f"Invalid JSON: {e}"}
            await updater.add_artifact(
                [{"type": "text", "text": json.dumps(response)}]
            )
            return
        
        msg_type = game_message.get("type")
        logger.info(f"Agent {self.agent_id} received: {msg_type}")
        
        # Route to appropriate handler
        handlers = {
            "game_start": self._handle_game_start,
            "bid_request": self._handle_bid_request,
            "night_action": self._handle_night_action,
            "speak": self._handle_speak,
            "vote": self._handle_vote,
            "sheriff_election": self._handle_sheriff_election,
            "sheriff_summary": self._handle_sheriff_summary,
            "hunter_shoot": self._handle_hunter_shoot,
        }
        
        handler = handlers.get(msg_type)
        if handler:
            response = await handler(game_message)
        else:
            response = {"status": "acknowledged"}
        
        # Send response as JSON text
        response_text = json.dumps(response)
        await updater.add_artifact(
            [{"type": "text", "text": response_text}]
        )
    
    def _safe_parse_json(self, response: str, fallback: Dict = None) -> Dict:
        """Safely parse JSON from LLM response"""
        if fallback is None:
            fallback = {}
        
        cleaned = response.replace("```json", "").replace("```", "").strip()
        
        if not cleaned:
            logger.warning("Empty response from LLM")
            return fallback
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            return fallback
    
    async def _handle_game_start(self, message: Dict) -> Dict:
        """Initialize when game starts"""
        self.game_id = message.get("game_id")
        self.player_id = message.get("player_id")
        self.role = message.get("role")
        self.camp = message.get("camp")
        self.alive_players = message.get("alive_players", [])
        
        # Reset state
        self.suspicions = {}
        self.conversation_history = []
        
        if "werewolf_team" in message:
            self.werewolf_team = message["werewolf_team"]
        else:
            self.werewolf_team = []
        
        # Assign personality once per game
        personalities = ["cautious", "aggressive", "chaotic", "logical"]
        self.personality = random.choice(personalities)
        
        logger.info(
            f"Agent {self.agent_id} (Player {self.player_id}): "
            f"Role={self.role}, Camp={self.camp}, Personality={self.personality}"
        )
        
        return {"status": "initialized"}
    
    async def _handle_bid_request(self, message: Dict) -> Dict:
        """Handle bidding for speaking turn"""
        game_state = message.get("game_state", {})
        context = message.get("context", "")
        round_num = message.get("round", 0)
        
        personality = self.personality or "neutral"
        
        # Calculate urgency
        urgency_factors = []
        
        if self.role == "seer" and round_num > 1:
            urgency_factors.append("Have seer information to share")
        
        high_suspicions = [pid for pid, sus in self.suspicions.items() if sus > 0.7]
        if high_suspicions:
            urgency_factors.append(f"High suspicion on players: {high_suspicions}")
        
        # Personality-based base bid
        base_bids = {
            "cautious": 30,
            "logical": 40,
            "aggressive": 60,
            "chaotic": 50,
        }
        base_bid = base_bids.get(personality, 40)
        
        prompt = f"""You are deciding how urgently you want to speak in Werewolf game.

Your role: {self.role}
Your camp: {self.camp}
Personality: {personality}
Round: {round_num}
Context: {context}

Urgency factors:
{urgency_factors if urgency_factors else ['No urgent information']}

Bid guidelines (0-100):
- 80-100: Critical information (seer with wolf ID, hunter dying)
- 60-80: Strong suspicions or need to defend
- 40-60: Want to contribute to discussion
- 20-40: Willing to listen first
- 0-20: No urgent need to speak

RESPONSE FORMAT (JSON only):
{{"bid_value": 65, "bid_reasoning": "I have important observations"}}

Base suggestion: {base_bid}"""
        
        temp_map = {"cautious": 0.4, "logical": 0.5, "aggressive": 0.7, "chaotic": 0.9}
        temperature = temp_map.get(personality, 0.6)
        
        try:
            llm_response = await self._call_llm(prompt, temperature=temperature)
            parsed = self._safe_parse_json(llm_response, {"bid_value": base_bid})
            
            bid_value = max(0, min(100, parsed.get("bid_value", base_bid)))
            bid_reasoning = parsed.get("bid_reasoning", "Standard bid")
            
            if personality == "chaotic" and random.random() < 0.3:
                bid_value = random.randint(20, 90)
                bid_reasoning += " (chaotic impulse)"
            
            logger.info(f"Player {self.player_id} bid: {bid_value} - {bid_reasoning}")
            
            return {"bid_value": bid_value, "bid_reasoning": bid_reasoning}
        
        except Exception as e:
            logger.error(f"Error in bidding: {e}")
            return {"bid_value": base_bid, "bid_reasoning": "Default bid"}
    
    async def _handle_speak(self, message: Dict) -> Dict:
        """Generate speech with private reasoning"""
        observation = message.get("observation", {})
        alive = observation.get("alive_players", self.alive_players)
        self.alive_players = alive
        
        targets = [p for p in alive if p != self.player_id]
        target = targets[0] if targets else (alive[0] if alive else None)
        personality = self.personality or "neutral"
        
        prompt = f"""You are playing Werewolf and need to speak publicly.

Your role: {self.role}
Your camp: {self.camp}
Personality: {personality}
Alive players: {alive}
Your current suspicions: {self.suspicions}

CRITICAL: You must provide BOTH private reasoning and public speech.

PRIVATE REASONING (what you think internally):
- What facts do you KNOW?
- Who do you SUSPECT and why?
- What STRATEGY should you use?
- How CONFIDENT are you (0.0-1.0)?

PUBLIC SPEECH (what you say to others):
- MUST accuse a specific player by ID number
- Keep it 1-3 sentences
- Be strategic based on your camp and personality

RESPONSE FORMAT (JSON only):
{{
    "private_thoughts": "My analysis...",
    "suspicions": {{"5": 0.95, "3": 0.40}},
    "strategy": "observe",
    "confidence": 0.85,
    "speech": "I suspect Player X is a werewolf because..."
}}

Target suggestions: {targets[:3] if len(targets) > 3 else targets}"""
        
        temp_map = {"cautious": 0.6, "logical": 0.7, "aggressive": 0.9, "chaotic": 1.0}
        temperature = temp_map.get(personality, 0.8)
        
        try:
            llm_response = await self._call_llm(prompt, temperature=temperature)
            default_target = target if target is not None else (alive[0] if alive else 1)
            parsed = self._safe_parse_json(llm_response, {
                "speech": f"I suspect Player {default_target} is a werewolf.",
                "private_thoughts": "Analyzing the situation...",
                "suspicions": {},
                "strategy": "observe",
                "confidence": 0.5
            })
            
            speech = parsed.get("speech", f"I suspect Player {default_target} is a werewolf.")
            private_thoughts = parsed.get("private_thoughts", "Analyzing...")
            suspicions = parsed.get("suspicions", {})
            strategy = parsed.get("strategy", "observe")
            confidence = parsed.get("confidence", 0.5)
            
            # Update internal suspicions
            if suspicions:
                self.suspicions.update({int(k): float(v) for k, v in suspicions.items()})
            
            # Ensure speech mentions a player
            if targets:
                if not any(f"player {p}" in speech.lower() for p in targets):
                    speech = f"I suspect Player {default_target} is a werewolf. " + speech
            
            logger.info(f"Player {self.player_id} speech: {speech}")
            
            return {
                "speech": speech,
                "private_thoughts": private_thoughts,
                "suspicions": suspicions,
                "strategy": strategy,
                "confidence": confidence
            }
        
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            fallback = target if target is not None else (alive[0] if alive else 0)
            return {
                "speech": f"I suspect Player {fallback} is a werewolf.",
                "private_thoughts": "Error in reasoning process",
                "suspicions": {},
                "strategy": "observe",
                "confidence": 0.0
            }
    
    async def _handle_night_action(self, message: Dict) -> Dict:
        """Handle night action using LLM"""
        observation = message.get("observation", {})
        personality = self.personality or "neutral"
        alive_players = observation.get("alive_players", self.alive_players)
        
        prompt = f"""You are playing Werewolf game.

Your role: {self.role}
Your camp: {self.camp}
Player ID: {self.player_id}
Personality: {personality}

Current game state:
- Round: {observation.get('round_number', 0)}
- Phase: {observation.get('phase', 'night')}
- Alive players: {alive_players}

ROLE-SPECIFIC ACTIONS:
- werewolf: Use "kill" to eliminate a player
- seer: Use "check" to investigate a player's role
- guard: Use "protect" to protect a player
- witch: Use "heal" to save victim, or "poison" to eliminate

RESPONSE FORMAT (JSON only):
{{"action_type": "kill", "target_id": 3}}

Valid action_types: kill, check, protect, heal, poison
If you don't want to act: {{"skip": true}}"""
        
        temp_map = {"cautious": 0.4, "logical": 0.5, "aggressive": 0.7, "chaotic": 0.9}
        temperature = temp_map.get(personality, 0.6)
        
        try:
            llm_response = await self._call_llm(prompt, temperature=temperature)
            parsed = self._safe_parse_json(llm_response, {})
            
            if parsed.get("skip") or not parsed.get("action_type"):
                return {"action": None}
            
            action_type = parsed.get("action_type")
            target_id = parsed.get("target_id")
            
            # Chaotic personality: sometimes change target
            if (personality == "chaotic" and target_id in alive_players 
                and len(alive_players) > 1 and random.random() < 0.25):
                alt_targets = [p for p in alive_players if p != target_id]
                if alt_targets:
                    target_id = random.choice(alt_targets)
                    logger.info(f"[chaotic] changed night target to {target_id}")
            
            action = {
                "action_type": action_type,
                "player_id": self.player_id,
                "target_id": target_id,
            }
            
            logger.info(f"Night action: {action}")
            return {"action": action}
        
        except Exception as e:
            logger.error(f"Error in night action: {e}")
            return {"action": None}
    
    async def _handle_vote(self, message: Dict) -> Dict:
        """Vote using LLM"""
        candidates = message.get("candidates", [])
        observation = message.get("observation", {})
        personality = self.personality or "neutral"
        
        if not candidates:
            return {"vote": self.player_id or 0}
        
        prompt = f"""You are voting to exile someone in Werewolf game.

Your role: {self.role}
Your camp: {self.camp}
Personality: {personality}
Candidates: {candidates}
Your suspicions: {self.suspicions}

Game state:
- Round: {observation.get('round_number', 0)}
- Alive: {observation.get('alive_players', [])}

Strategy:
- If good camp: Vote for suspected werewolves
- If werewolf: Vote for good players, avoid suspicion

RESPONSE FORMAT (JSON only):
{{"vote": 3}}

Choose from candidates: {candidates}"""
        
        temp_map = {"cautious": 0.4, "logical": 0.5, "aggressive": 0.6, "chaotic": 0.8}
        temperature = temp_map.get(personality, 0.5)
        
        try:
            llm_response = await self._call_llm(prompt, temperature=temperature)
            parsed = self._safe_parse_json(llm_response, {"vote": candidates[0]})
            
            vote = parsed.get("vote", candidates[0])
            if vote not in candidates:
                logger.warning(f"Invalid vote {vote}, using {candidates[0]}")
                vote = candidates[0]
            
            # Random vote change for chaotic
            p_random = 0.3 if personality == "chaotic" else 0.1
            if len(candidates) > 1 and random.random() < p_random:
                alt = [c for c in candidates if c != vote]
                if alt:
                    vote = random.choice(alt)
                    logger.info(f"[{personality}] randomized vote to {vote}")
            
            logger.info(f"Vote: {vote}")
            return {"vote": vote}
        
        except Exception as e:
            logger.error(f"Error voting: {e}")
            return {"vote": candidates[0]}
    
    async def _handle_sheriff_election(self, message: Dict) -> Dict:
        """Vote for sheriff"""
        candidates = message.get("candidates", [])
        if not candidates:
            return {"vote": self.player_id or 0}
        
        personality = self.personality or "neutral"
        
        prompt = f"""Vote for sheriff in Werewolf game.

Your role: {self.role}
Your camp: {self.camp}
Personality: {personality}
Candidates: {candidates}

Sheriff gets extra power in voting.

RESPONSE FORMAT (JSON only):
{{"vote": 3}}"""
        
        try:
            llm_response = await self._call_llm(prompt, temperature=0.5)
            parsed = self._safe_parse_json(llm_response, {"vote": candidates[0]})
            
            vote = parsed.get("vote", candidates[0])
            if vote not in candidates:
                vote = candidates[0]
            
            return {"vote": vote}
        
        except Exception as e:
            logger.error(f"Error in sheriff vote: {e}")
            return {"vote": candidates[0]}
    
    async def _handle_sheriff_summary(self, message: Dict) -> Dict:
        """Sheriff makes recommendation"""
        votes = message.get("votes", {}) or {}
        
        vote_counts: Dict[int, int] = {}
        for _, target in votes.items():
            vote_counts[target] = vote_counts.get(target, 0) + 1
        
        most_voted = None
        if vote_counts:
            most_voted = max(vote_counts, key=lambda k: vote_counts[k])
        elif votes:
            most_voted = list(votes.values())[0]
        else:
            most_voted = self.player_id or 0
        
        personality = self.personality or "neutral"
        
        prompt = f"""You are the Sheriff in Werewolf game.

Personality: {personality}
Voting results: {vote_counts}
Most voted player: {most_voted}

Make a recommendation for who to exile.

RESPONSE FORMAT (JSON only):
{{"recommendation": {most_voted}}}"""
        
        temperature = 0.6 if personality == "chaotic" else 0.4
        
        try:
            llm_response = await self._call_llm(prompt, temperature=temperature)
            parsed = self._safe_parse_json(llm_response, {"recommendation": most_voted})
            
            return {"recommendation": parsed.get("recommendation", most_voted)}
        
        except Exception as e:
            logger.error(f"Error in sheriff summary: {e}")
            return {"recommendation": most_voted}
    
    async def _handle_hunter_shoot(self, message: Dict) -> Dict:
        """Hunter shoots after death"""
        targets = message.get("targets", [])
        if not targets:
            return {"target": self.player_id or 0}
        
        personality = self.personality or "neutral"
        
        prompt = f"""You are the Hunter in Werewolf and you're dying.
You can shoot ONE player.

Personality: {personality}
Available targets: {targets}
Your suspicions: {self.suspicions}

RESPONSE FORMAT (JSON only):
{{"target": 3}}"""
        
        temp_map = {"cautious": 0.4, "logical": 0.5, "aggressive": 0.6, "chaotic": 0.8}
        temperature = temp_map.get(personality, 0.5)
        
        try:
            llm_response = await self._call_llm(prompt, temperature=temperature)
            parsed = self._safe_parse_json(llm_response, {"target": targets[0]})
            
            target = parsed.get("target", targets[0])
            if target not in targets:
                target = targets[0]
            
            # Chaotic randomization
            if personality == "chaotic" and len(targets) > 1 and random.random() < 0.3:
                alt = [t for t in targets if t != target]
                if alt:
                    target = random.choice(alt)
                    logger.info(f"[chaotic] randomized hunter shot to {target}")
            
            return {"target": target}
        
        except Exception as e:
            logger.error(f"Error in hunter shoot: {e}")
            return {"target": targets[0]}
    
    async def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call LLM API"""
        if self.provider == "openai":
            return await self._call_openai(prompt, temperature)
        elif self.provider == "anthropic":
            return await self._call_anthropic(prompt, temperature)
        elif self.provider == "groq":
            return await self._call_groq(prompt, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _call_openai(self, prompt: str, temperature: float) -> str:
        """Call OpenAI API"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        logger.debug(f"Calling OpenAI API (model: {self.model}, temp: {temperature})")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self.model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an intelligent Werewolf game agent. Respond with ONLY pure JSON."
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": 300,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    
    async def _call_anthropic(self, prompt: str, temperature: float) -> str:
        """Call Anthropic API"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 300,
                    "messages": [
                        {
                            "role": "user",
                            "content": "You are a Werewolf game agent. Respond with ONLY valid JSON.\n\n" + prompt
                        }
                    ],
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            return response.json()["content"][0]["text"]
    
    async def _call_groq(self, prompt: str, temperature: float) -> str:
        """Call Groq API (OpenAI-compatible)"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        
        logger.debug(f"Calling Groq API (model: {self.model}, temp: {temperature})")
        
        # Adjust tokens for different models
        if self.model == "openai/gpt-oss-20b":
            max_tokens = 1500
        elif self.model.startswith("openai/gpt-oss"):
            max_tokens = 1200
        else:
            max_tokens = 1000
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self.model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an intelligent Werewolf game agent. Respond with ONLY pure JSON."
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            
            if response.status_code != 200:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
            
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
