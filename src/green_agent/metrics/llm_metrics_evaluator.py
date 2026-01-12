""" 
LLM Metrics Evaluator

Low-level LLM integration for metrics: API calls, retries, and JSON cleanup.

Used by llm_metrics_calculator.py and the unified MetricsCalculator.
"""

import os
import json
import logging
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)


class MetricsLLMEvaluator:
    """LLM-based evaluator for sophisticated metrics"""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model

        if provider == "groq":
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            if not self.api_key:
                logger.warning(
                    "⚠️  No GROQ_API_KEY found. LLM-based metrics (IRS/MSS) will be disabled.\n"
                    "   To enable: export GROQ_API_KEY='gsk_...'"
                )
        else:  # openai
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning(
                    "⚠️  No OPENAI_API_KEY found. LLM-based metrics (IRS/MSS) will be disabled.\n"
                    "   To enable: export OPENAI_API_KEY='sk-...'"
                )
            elif not self.api_key.startswith("sk-"):
                logger.warning(
                    "⚠️  Invalid OPENAI_API_KEY format (should start with 'sk-'). "
                    "LLM metrics will likely fail."
                )

    async def evaluate_all_players_batch(
        self,
        game_history: List[Dict[str, Any]],
        all_players: Dict[int, Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """Evaluate all players' IRS + MSS in one API call."""
        with open("/tmp/metrics_debug.log", "a") as f:
            f.write(f"🔥 evaluate_all_players_batch CALLED for {len(all_players)} players\n")

        if not self.api_key:
            with open("/tmp/metrics_debug.log", "a") as f:
                f.write("❌ No API key, returning empty\n")
            return {}

        logger.info(f"[METRICS] 🚀 BATCH evaluation for all {len(all_players)} players in ONE API call")

        player_speeches: Dict[int, List[Dict[str, Any]]] = {}
        for event in game_history:
            if event.get("type") == "speech":
                pid = event.get("player_id")
                if pid not in player_speeches:
                    player_speeches[pid] = []
                player_speeches[pid].append(event)

        players_data = []
        for player_id in all_players.keys():
            speeches = player_speeches.get(player_id, [])
            speech_text = " | ".join(
                [
                    f"R{s.get('round', 0)}: '{s.get('text', '')[:300]}...'"
                    for s in speeches[:5]
                ]
            )
            players_data.append({"id": player_id, "speeches": speech_text if speech_text else "No speeches"})

        max_round = max(
            (s.get("round", 1) for s in game_history if s.get("type") == "speech"),
            default=1,
        )
        eliminations = [e for e in game_history if e.get("type") == "elimination"]
        eliminated_ids = [e.get("player_id") for e in eliminations]
        recent_eliminations = eliminations[-3:] if len(eliminations) > 3 else eliminations

        strict_rules = """
    STRICT JSON OUTPUT RULES (CRITICAL):
    - Return ONE valid JSON object only (no prose, no markdown fences).
    - Use double quotes for every key and string value.
    - Escape internal quotes or newlines using standard JSON escaping (e.g., \" or \n).
    - Do not add trailing commas.
    - Do not truncate strings; ensure every opening quote has a closing quote.
    - The output MUST be parseable by Python json.loads without modification.
    """

        base_prompt = f"""Analyze ALL players in this Werewolf game simultaneously.

Game context:
- Total players: {len(all_players)}
- Current round: {max_round}
- Eliminated players: {eliminated_ids}

Players data:
{json.dumps(players_data, indent=2)}

Recent eliminations context:
{json.dumps(recent_eliminations, indent=2)}

For EACH player, evaluate:

1. IRS (Identity Recognition): Based on their speeches, who do they believe are wolves vs good?
   - Analyze accusations, defenses, voting patterns mentioned
   - For each OTHER player, predict: "wolf", "good", or "unknown"
    - IMPORTANT: You MUST output a prediction for every other player.
      Use "unknown" only if there is absolutely no signal about that player.
      If weak signals exist, make your best guess (prefer "good" if truly ambiguous).
   - Consider consistency of claims across rounds

2. MSS (Message Quality): How human-like and strategically sound are their messages?
   Scale:
   - 0.0-0.3: bot-like, generic, no personality
   - 0.4-0.5: mechanical, repetitive patterns
   - 0.5-0.7: decent but somewhat formulaic
   - 0.7-0.85: natural, shows strategy and emotion
   - 0.85-1.0: excellent, nuanced, highly believable

Evaluation criteria for MSS:
- Natural language flow (not overly formal)
- Strategic reasoning visible
- Emotional authenticity (suspicion, confidence, fear)
- Appropriate detail level
- Personality consistency

Respond with ONLY valid JSON:
{{
  "1": {{
    "irs": {{"2": "wolf", "3": "good", "4": "unknown", ...}},
    "mss": 0.75,
    "mss_reasoning": "Brief explanation"
  }},
  ...
}}

Include ALL {len(all_players)} players. Be precise with MSS scores - avoid defaulting to 0.5.

{strict_rules}
"""

        try:
            with open("/tmp/metrics_debug.log", "a") as f:
                f.write(f"📤 About to call _call_llm with {len(base_prompt)} chars\n")

            logger.info(f"[METRICS] 📤 Sending batch request (prompt: {len(base_prompt)} chars)")

            parsed_result: Optional[Dict[str, Any]] = None
            parse_errors: List[str] = []
            max_attempts = 3

            for attempt in range(max_attempts):
                attempt_prompt = base_prompt
                if attempt > 0:
                    last_error = parse_errors[-1] if parse_errors else "unknown format error"
                    attempt_prompt += (
                        f"\nREMINDER #{attempt}: Previous output was invalid JSON ({last_error}). "
                        "Respond with STRICT JSON that can be parsed without edits."
                    )

                response = await self._call_llm(attempt_prompt, max_tokens=6000)

                with open("/tmp/metrics_debug.log", "a") as f:
                    f.write(f"✅ Got response from _call_llm (attempt {attempt+1}): {len(response)} chars\n")
                    f.write(f"Raw response: {response[:500]}...\n")
                    if len(response) > 5500:
                        f.write(f"⚠️ WARNING: Response length {len(response)} is close to max_tokens limit!\n")

                cleaned = self._normalize_json_response(response)
                parsed_result, parse_error = self._attempt_json_parse(cleaned)

                if parsed_result is not None:
                    break

                error_msg = str(parse_error)
                parse_errors.append(error_msg)
                logger.error(f"[METRICS] JSON parse attempt {attempt+1} failed: {error_msg}")
                logger.error(f"Problematic JSON (first 1000 chars): {cleaned[:1000]}")

            if parsed_result is None:
                raise ValueError(
                    f"LLM returned invalid JSON after {max_attempts} attempts: "
                    f"{parse_errors[-1] if parse_errors else 'unknown error'}"
                )

            result = parsed_result

            batch_results: Dict[int, Dict[str, Any]] = {}
            for k, v in result.items():
                try:
                    player_id = int(k)
                    irs_predictions: Dict[int, str] = {}
                    for target_k, target_v in v.get("irs", {}).items():
                        try:
                            irs_predictions[int(target_k)] = target_v
                        except Exception:
                            continue

                    batch_results[player_id] = {
                        "irs": irs_predictions,
                        "mss": float(v.get("mss", 0.5)),
                    }
                except Exception as e:
                    logger.warning(f"Error parsing player {k}: {e}")
                    continue

            with open("/tmp/metrics_debug.log", "a") as f:
                f.write(f"✅ Batch evaluation complete for {len(batch_results)} players\n")

            logger.info(f"[METRICS] ✅ Batch evaluation complete for {len(batch_results)} players")
            return batch_results

        except Exception as e:
            with open("/tmp/metrics_debug.log", "a") as f:
                f.write(f"❌ EXCEPTION in batch evaluation: {str(e)[:200]}\n")
            logger.error(f"[METRICS] ❌ Error in batch evaluation: {e}", exc_info=True)
            return {}

    async def evaluate_identity_recognition(
        self,
        speeches: List[Dict[str, Any]],
        player_id: int,
        all_players: Dict[int, Dict[str, Any]],
    ) -> Dict[int, str]:
        """Legacy per-player method (prefer batch)."""
        with open("/tmp/metrics_debug.log", "a") as f:
            f.write(
                f"⚠️ OLD METHOD evaluate_identity_recognition called for player {player_id} (should use batch!)\n"
            )
        if not self.api_key or not speeches:
            return {}

        speech_text = "\n".join([f"Round {s.get('round', 0)}: \"{s.get('text', '')}\"" for s in speeches])
        other_players = [pid for pid in all_players.keys() if pid != player_id]

        prompt = f"""You are analyzing a Werewolf game player's speeches to determine who they think are werewolves vs good players.

Player {player_id}'s speeches:
{speech_text}

Other players in game: {other_players}

Based on these speeches, for EACH other player, determine if Player {player_id} thinks they are:
- "wolf" (werewolf / suspicious)
- "good" (villager / innocent)
- "unknown" (no clear indication)

IMPORTANT:
- You MUST output a prediction for EVERY other player.
- Use "unknown" only if there is absolutely no signal about that player.
- If evidence is weak, make your best guess (prefer "good" if truly ambiguous).

Respond with ONLY valid JSON (no markdown):
{{
  "predictions": {{
    "2": "wolf",
    "3": "good",
    "4": "unknown",
    ...
  }}
}}

Include ALL players from the list {other_players}."""

        try:
            response = await self._call_llm(prompt)
            cleaned = response.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned)

            predictions: Dict[int, str] = {}
            for k, v in result.get("predictions", {}).items():
                try:
                    predictions[int(k)] = v
                except (ValueError, TypeError):
                    continue
            return predictions
        except Exception as e:
            logger.error(f"[METRICS] ❌ Error in IRS evaluation: {e}", exc_info=True)
            return {}

    async def evaluate_message_quality(self, speeches: List[Dict[str, Any]], player_id: int) -> float:
        """Legacy per-player method (prefer batch)."""
        with open("/tmp/metrics_debug.log", "a") as f:
            f.write(
                f"⚠️ OLD METHOD evaluate_message_quality called for player {player_id} (should use batch!)\n"
            )

        if not self.api_key or not speeches:
            return 0.5

        speech_text = "\n".join([f"Round {s.get('round', 0)}: \"{s.get('text', '')}\"" for s in speeches])

        prompt = f"""You are evaluating the realism and human-likeness of messages in a Werewolf social deduction game.

Player {player_id}'s messages:
{speech_text}

Evaluate these messages on a scale of 0.0 to 1.0 based on:
1. Natural language (not overly formal or robotic)
2. Strategic reasoning (shows game understanding)
3. Social dynamics (engages with other players)
4. Appropriate detail (not too generic, not overly verbose)
5. Emotional authenticity (shows suspicion, confidence, uncertainty)

Score criteria:
- 0.0-0.3: Clearly bot-like, generic, repetitive
- 0.3-0.5: Somewhat mechanical, lacks personality
- 0.5-0.7: Decent, could be human but somewhat bland
- 0.7-0.9: Natural, engaging, human-like
- 0.9-1.0: Excellent, highly realistic and strategic

Respond with ONLY valid JSON (no markdown):
{{
  "score": 0.75,
  "reasoning": "Brief explanation"
}}"""

        try:
            response = await self._call_llm(prompt)
            cleaned = response.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned)
            score = result.get("score", 0.5)
            score = max(0.0, min(1.0, float(score)))
            return score
        except Exception as e:
            logger.error(f"[METRICS] ❌ Error in MSS evaluation: {e}")
            return 0.5

    async def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        if self.provider == "groq":
            return await self._call_groq(prompt, max_tokens)
        if self.provider == "openai":
            return await self._call_openai(prompt, max_tokens)
        raise ValueError(f"Provider {self.provider} not supported")

    @staticmethod
    def _parse_retry_after_seconds(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _compute_backoff_seconds(attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return max(0.5, min(60.0, retry_after))
        return min(30.0, 2.0**attempt)

    async def _call_groq(self, prompt: str, max_tokens: int) -> str:
        logger.debug(f"[METRICS] 📡 Calling Groq API (model: {self.model})...")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a Werewolf game analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }

        max_attempts = 5
        last_error: Optional[Exception] = None
        
        # Select API URL based on provider
        if self.provider == "groq":
            api_url = "https://api.groq.com/openai/v1/chat/completions"
        else:  # openai
            api_url = "https://api.openai.com/v1/chat/completions"

        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(max_attempts):
                try:
                    response = await client.post(
                        api_url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                    )
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                    logger.debug(f"[METRICS] ✓ {self.provider.upper()} API call complete ({len(content)} chars)")
                    return content
                except httpx.HTTPStatusError as e:
                    last_error = e
                    status_code = e.response.status_code
                    retry_after = self._parse_retry_after_seconds(e.response.headers.get("retry-after"))

                    is_retryable = status_code == 429 or 500 <= status_code < 600
                    if is_retryable and attempt < max_attempts - 1:
                        delay = self._compute_backoff_seconds(attempt, retry_after)
                        logger.warning(
                            f"[METRICS] {self.provider.upper()} HTTP {status_code} (attempt {attempt+1}/{max_attempts}); retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.error(f"[METRICS] Groq API call failed (HTTP {status_code}): {e}")
                    raise
                except (httpx.TimeoutException, httpx.RequestError) as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        delay = self._compute_backoff_seconds(attempt, None)
                        logger.warning(
                            f"[METRICS] Groq request error (attempt {attempt+1}/{max_attempts}): {e}; retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.error(f"[METRICS] Groq API call failed: {e}")
                    raise

        assert last_error is not None
        raise last_error

    async def _call_openai(self, prompt: str, max_tokens: int) -> str:
        logger.debug(f"[METRICS] 📡 Calling OpenAI API (model: {self.model})...")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a Werewolf game analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }

        max_attempts = 5
        last_error: Optional[Exception] = None

        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(max_attempts):
                try:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                    )
                    response.raise_for_status()
                    response_data = response.json()
                    return response_data["choices"][0]["message"]["content"]

                except httpx.TimeoutException as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        delay = self._compute_backoff_seconds(attempt, None)
                        logger.warning(
                            f"[METRICS] OpenAI timeout (attempt {attempt+1}/{max_attempts}); retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.error(f"[METRICS] ⏱️ OpenAI API timeout after 120s: {e}")
                    raise

                except httpx.HTTPStatusError as e:
                    last_error = e
                    status_code = e.response.status_code
                    retry_after = self._parse_retry_after_seconds(e.response.headers.get("retry-after"))
                    is_retryable = status_code == 429 or 500 <= status_code < 600

                    if is_retryable and attempt < max_attempts - 1:
                        delay = self._compute_backoff_seconds(attempt, retry_after)
                        logger.warning(f"[METRICS] Retrying OpenAI in {delay:.1f}s")
                        await asyncio.sleep(delay)
                        continue

                    logger.error(f"[METRICS] ❌ OpenAI API HTTP error: {status_code} - {e.response.text[:200]}")
                    raise

                except httpx.RequestError as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        delay = self._compute_backoff_seconds(attempt, None)
                        logger.warning(
                            f"[METRICS] OpenAI request error (attempt {attempt+1}/{max_attempts}): {e}; retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.error(f"[METRICS] OpenAI API call failed: {e}", exc_info=True)
                    raise

                except Exception as e:
                    last_error = e
                    logger.error(f"[METRICS] OpenAI API call failed: {e}", exc_info=True)
                    raise

        assert last_error is not None
        raise last_error

    def _normalize_json_response(self, raw_response: str) -> str:
        text = raw_response.strip()
        text = text.replace("\r", "")
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            text = text[start : end + 1]
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        diff_curly = text.count("{") - text.count("}")
        if diff_curly > 0:
            text += "}" * diff_curly
        diff_square = text.count("[") - text.count("]")
        if diff_square > 0:
            text += "]" * diff_square
        return text

    def _attempt_json_parse(self, candidate: str) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            repaired = candidate
            repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
            repaired = repaired.replace("\\'", "'")
            repaired = repaired.replace("\t", " ")
            try:
                return json.loads(repaired), None
            except Exception as secondary_error:
                return None, secondary_error
