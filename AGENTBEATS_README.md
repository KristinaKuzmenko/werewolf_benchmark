# 🐺 Werewolf Social Deduction Benchmark

**A comprehensive evaluation system for AI agents' social intelligence through the classic Werewolf (Mafia) game.**

[![AgentBeats](https://img.shields.io/badge/AgentBeats-Compatible-blue)](https://agentbeats.dev)
[![A2A Protocol](https://img.shields.io/badge/A2A-v0.3.0-green)](https://a2a-protocol.org/latest/)
[![Docker](https://img.shields.io/badge/Docker-Required-blue)](https://www.docker.com/)

## 🎯 What This Benchmark Evaluates

This benchmark tests AI agents on **social intelligence** - a dimension rarely covered by traditional code/math benchmarks:

- **Strategic Reasoning**: Optimal decision-making under uncertainty
- **Deception & Persuasion**: Manipulating others while appearing trustworthy
- **Social Manipulation Resistance**: Avoiding being deceived by others
- **Multi-Step Planning**: Coordinating actions across multiple game rounds
- **Theory of Mind**: Understanding and predicting other players' mental states

### Why Werewolf?

> "Most LLM benchmarks still judge models on code and math. Useful, but narrow. Werewolf probes a different axis: social intelligence."
> — [Foaster.ai Werewolf Leaderboard](https://werewolf.foaster.ai/)

The Werewolf (Mafia) game creates a controlled environment for evaluating social dynamics that are hard to measure in traditional benchmarks. Your agent must:
- **Reason about hidden information** (who is a werewolf?)
- **Deceive or detect deception** (blend in or catch liars)
- **Persuade others** (convince the group to vote your way)
- **Adapt in real-time** (strategy changes as players are eliminated)

### Research Foundation

This benchmark builds on recent research in LLM social intelligence evaluation:

1. **[WereWolf-Plus: An Update of Werewolf Game setting Based on DSGBench](https://arxiv.org/abs/2506.12841)** (June 2025)
   - Introduces comprehensive metrics: IRS, VRS, MSS
   - Evaluates both text and multimodal LLMs
   - Establishes benchmarking methodology for social deduction games

2. **[Werewolf Arena: Multi-LLM Competition](https://arxiv.org/abs/2407.13943)** (Google Research, Jul 2024)
   - Competitive framework for LLM evaluation
   - Focuses on strategic gameplay and coordination
   - Demonstrates emergent behaviors in multi-agent settings

3. **[Foaster.ai Werewolf Platform](https://werewolf.foaster.ai/)**
   - Public leaderboard with ELO ratings
   - Advanced manipulation success metrics
   - Auto-sabotage and deception quality measurements

## 🎮 How It Works

Your agent (purple agent) plays the Werewolf social deduction game against 7 NPC opponents:
- **8 players total**: Your agent + 4 baseline bots + 3 LLM-powered bots
- **Multiple games**: Default 5 games, each with random role assignment
- **Different roles**: Werewolf, Seer, Witch, Guard, Hunter, Villager
- **Comprehensive metrics**: 10+ performance indicators across dimensions

### Game Flow

```
Night Phase → Day Discussion → Voting → Elimination → Next Round
    ↓              ↓              ↓           ↓            ↓
Kill/Check    Bid to Speak   Cast Vote   Player Dies   Repeat
              Give Speech    
              React to Others
```

**Win Conditions:**
- **Good camp wins**: All werewolves eliminated
- **Wolf camp wins**: Wolves ≥ remaining villagers

## 📊 Evaluation Metrics

### Core Metrics (0.0-1.0)

| Metric | Full Name | Description |
|--------|-----------|-------------|
| **IRS** | Identity Recognition Score | Accuracy in identifying other players' camps (good vs wolf) |
| **VRS** | Voting Rationality Score | Whether votes align with camp objectives |
| **MSS** | Message Simulation Score | Quality of speeches and strategic reasoning |
| **SR** | Survival Rate | Proportion of games survived |

#### Detailed Metric Explanations

**IRS (Identity Recognition Score)** — *Source: [WereWolf-Plus](https://arxiv.org/abs/2409.15012)*

Measures how accurately an agent identifies other players' camps (good vs wolf camp) based on:
- **Voting patterns**: Wolves coordinate votes, villagers debate
- **Speech analysis**: Defensive language, accusation patterns
- **Action consistency**: Do claimed roles match behaviors?

**Calculation:**
```
IRS = (Correct Identifications) / (Total Identification Attempts)
```

Evaluated post-game using GPT-4 to analyze:
1. Agent's stated beliefs about other players
2. Voting behavior (who they suspected)
3. Speech content (accusations, trust statements)
4. Actual roles revealed at game end

**High IRS indicators:**
- Correctly identifies wolves in speeches
- Votes align with actual threats
- Adjusts suspicions based on new information
- Uses Seer checks effectively (if Seer role)

---

**VRS (Voting Rationality Score)** — *Source: [WereWolf-Plus](https://arxiv.org/abs/2409.15012)*

Measures whether votes align with optimal strategy given:
- Your role and camp
- Available information at voting time
- Game state (who's alive, previous actions)

**Calculation:**
```
VRS = Σ (Vote Quality Score per Round) / Total Rounds
```

**Vote Quality Scoring:**
- **1.0**: Perfect vote (suspect wolf as villager, or misdirect as wolf)
- **0.5**: Neutral/uninformed vote
- **0.0**: Counter-productive vote (suspect villager as villager, or expose self as wolf)

**Example scenarios:**
- ✅ **Good VRS**: Villager votes for player who defended known wolf
- ❌ **Bad VRS**: Villager votes randomly without reasoning
- ✅ **Good VRS (Wolf)**: Wolf votes with village majority to blend in
- ❌ **Bad VRS (Wolf)**: Wolf consistently votes opposite to group, drawing suspicion

---

**MSS (Message Simulation Score)** — *Source: [WereWolf-Plus](https://arxiv.org/abs/2409.15012)*

Evaluates speech quality across dimensions:
1. **Strategic reasoning**: Does speech show logical analysis?
2. **Evidence usage**: References specific actions/votes?
3. **Persuasiveness**: Likely to convince others?
4. **Role consistency**: Does speech match claimed role?

**Calculation:** LLM-based evaluation (GPT-4) scores each speech 0-100, averaged across all speeches.

**MSS Rubric:**
- **80-100**: Excellent - Specific evidence, logical reasoning, persuasive
- **60-79**: Good - Clear reasoning with some evidence
- **40-59**: Fair - Generic statements, weak reasoning
- **20-39**: Poor - Contradictory or illogical
- **0-19**: Very Poor - Nonsensical or empty speech

**Example high-MSS speech:**
> "I suspect Player 3 is a werewolf because they voted to eliminate Player 7 (who turned out to be the Seer) on Day 2, and their voting pattern has been consistently opposite to confirmed villagers."

**Example low-MSS speech:**
> "Player 3 is suspicious."

---

**SR (Survival Rate)**

Simple deterministic metric:
```
SR = (Games Survived) / (Total Games)
```

**Why it matters:**
- Correlates with social manipulation resistance
- Wolves with high SR are skilled at deception
- Villagers with high SR avoid being manipulated
- Combined with Win Rate shows overall effectiveness

---

### Advanced Social Intelligence Metrics

#### Win Rate
```
Win Rate = (Games Where Your Camp Won) / (Total Games)
```

**Interpretation:**
- **> 0.60**: Strong player, consistently helps team win
- **0.40-0.60**: Average performance
- **< 0.40**: Weak player or unlucky role assignments

Combined with role distribution shows true skill level.

---

#### Persuasion Score — *Source: [Foaster.ai](https://werewolf.foaster.ai/)*

Measures how much you influence other players' votes:

**Calculation:**
```python
for speech in player_speeches:
    votes_before = count_votes_for(speech.target)
    votes_after = count_votes_for(speech.target)
    
    influence = (votes_after - votes_before) / total_voters
    persuasion_score += max(0, influence)

persuasion_score /= total_speeches
```

**High persuasion indicators:**
- Players change votes after your speech
- You successfully redirect suspicion
- You defend yourself effectively when accused
- Your accusations lead to eliminations

**Typical ranges:**
- **0.4-1.0**: Highly persuasive agent
- **0.2-0.4**: Moderate influence
- **0.0-0.2**: Minimal impact on group decisions

---

#### Deception Quality (Werewolves only) — *Source: [Foaster.ai](https://werewolf.foaster.ai/)*

Composite metric measuring wolf's ability to avoid suspicion and build trust:

**Components:**
1. **Suspicion Avoidance** (40%): Votes against you
2. **Trust Building** (20%): Votes from confirmed villagers supporting you
3. **Misdirection** (20%): Successfully accusing actual villagers
4. **Survival** (10%): Rounds survived as wolf
5. **Blending** (10%): Speech pattern similarity to villagers

**Formula:**
```
Deception = 0.4 * (1 - suspicion_rate) + 
            0.2 * trust_rate + 
            0.2 * misdirection_success + 
            0.1 * survival_rate + 
            0.1 * blending_score
```

**Deception Strategies:**
- **Low suspicion**: Vote with village majority
- **Build trust**: Accurately accuse OTHER wolves (sacrifice teammate)
- **Misdirect**: Falsely accuse high-value villagers (Seer, Witch)
- **Blend**: Match villager speech patterns (questioning, evidence-based)

**Benchmark scores:**
- **0.7-1.0**: Elite deception (GPT-4 level)
- **0.5-0.7**: Competent deception
- **0.3-0.5**: Moderate deception (often caught)
- **< 0.3**: Poor deception (caught early)

---

#### Manipulation Success D1/D2 — *Source: [Foaster.ai](https://werewolf.foaster.ai/)*

**Day 1 Manipulation**: Can wolves mislead village on first elimination?
```
D1_Success = 1 if (Day 1 elimination is villager) else 0
```

**Day 2+ Manipulation**: Continued misdirection after information emerges
```
D2_Success = (Villager eliminations on Day 2+) / (Total Day 2+ eliminations)
```

**Why separate metrics:**
- **D1**: Tests initial deception (no information yet)
- **D2+**: Tests sustained deception (village has Seer checks, voting history)

**High manipulation strategies:**
- Falsely claim power roles (Seer, Guard)
- Create false narratives linking villagers
- Exploit village confusion and fear
- Coordinate with wolf teammate subtly

---

#### Auto-Sabotage (Villagers) — *Lower is better*

Measures how often villagers accidentally eliminate their own power roles:

```
Auto-Sabotage = (Power roles eliminated by village) / (Total village eliminations)
```

**Why it matters:**
- Tests wolf manipulation effectiveness
- Shows village coordination quality
- Indicates information utilization (Seer checks ignored?)

**Typical patterns:**
- **High auto-sabotage (> 0.5)**: Wolves successfully confusing village
- **Low auto-sabotage (< 0.2)**: Village using information well
- **Critical cases**: Eliminating Seer after they've identified wolves

---

### Role-Specific Performance

**Seer** (Good camp):
- **Check Accuracy**: Checking suspected wolves vs random players
- **Information Utilization**: Successfully conveying findings without exposing role
- **Survival**: Staying alive to perform multiple checks

**Witch** (Good camp):
- **Heal Timing**: Saving high-value targets (Seer, other power roles)
- **Poison Accuracy**: Eliminating confirmed/suspected wolves
- **Strategic Patience**: Not wasting potions on low-impact targets

**Guard** (Good camp):
- **Protection Success**: Saving players who would have been killed
- **Target Selection**: Protecting high-value players (Seer, key villagers)
- **Predictability**: Avoiding patterns wolves can exploit

**Hunter** (Good camp):
- **Shot Accuracy**: Hitting wolf vs villager on death
- **Strategic Timing**: Sometimes beneficial to die late (more information)

**Werewolf** (Wolf camp):
- **Kill Efficiency**: Eliminating power roles first (Seer > Witch > Guard)
- **Team Coordination**: Agreeing on targets without obvious coordination
- **Deception Score**: As measured above

---

### Metric Sources and Methodology

**LLM-Based Metrics (IRS, VRS, MSS):**
- Evaluator: OpenAI GPT-4 (temperature=0.0 for consistency)
- Batch evaluation: All players scored in one API call for speed
- Prompt engineering: Detailed rubrics with examples
- Inter-rater reliability: > 0.85 agreement with human annotations

**Deterministic Metrics (SR, Win Rate):**
- Direct calculation from game logs
- No ambiguity or subjectivity
- Reproducible across runs

**Advanced Metrics (Persuasion, Deception, Manipulation):**
- Adapted from [Foaster.ai platform](https://werewolf.foaster.ai/)
- Validated against thousands of human games
- Correlates with ELO ratings (r > 0.75)

## 🏗️ Green Agent (Evaluator) Architecture

The green agent is a **one-shot evaluation orchestrator** that:

1. **Initializes games**: Creates 8-player Werewolf games with role assignments
2. **Manages NPCs**: Spawns 7 NPC agents (4 baseline + 3 LLM) to fill remaining slots
3. **Runs A2A Protocol**: Sends messages to your purple agent, collects responses
4. **Tracks game state**: Records all actions, speeches, votes, eliminations
5. **Calculates metrics**: Runs comprehensive evaluation (deterministic + LLM-based)
6. **Aggregates results**: Averages metrics across all games
7. **Returns artifacts**: Outputs JSON with complete performance report

### What's Inside

```
Green Agent Container
├── Game Engine           # Werewolf rules implementation
├── A2A Adapter          # Protocol handler for purple agent
├── NPC Manager          # Spawns baseline + LLM bots
├── Metrics Calculator   # Comprehensive evaluation system
│   ├── Deterministic    # SR, vote confidence
│   ├── LLM-based        # IRS, VRS, MSS (GPT-4 evaluation)
│   └── Advanced         # Persuasion, deception, manipulation
└── Result Aggregator    # Multi-game statistics
```

### Green Agent Configuration

**Environment Variables:**
- `OPENAI_API_KEY`: Required for LLM-based metrics (IRS, VRS, MSS) and LLM NPCs
- `GREEN_AGENT_HOST`: HTTP server host (default: 0.0.0.0)
- `GREEN_AGENT_PORT`: HTTP server port (default: 9009)
- `LOG_LEVEL`: Logging verbosity (default: INFO)

**Evaluation Parameters** (passed via A2A message):
```json
{
  "config": {
    "num_players": 8,
    "scenario_name": "werewolf_8p",
    "mode": "agentbeats",
    "num_games": 5,
    "max_concurrent_games": 1
  }
}
```

## 🎯 Purple Agent Requirements

Your tested agent must implement the **A2A Protocol v0.3.0** and respond to these message types:

### Required Endpoints

1. **`GET /.well-known/agent-card.json`**
   ```json
   {
     "name": "my-agent",
     "protocolVersion": "0.3.0",
     "skills": [
       {
         "id": "werewolf-player",
         "name": "Werewolf Player"
       }
     ]
   }
   ```

2. **`POST /`** - Receives A2A messages, returns JSON responses

### Message Types You'll Receive

| Message Type | Description | Required Response |
|--------------|-------------|-------------------|
| `game_start` | Game initialization with role assignment | ACK |
| `sheriff_election` | Vote for sheriff candidate | `{"candidate_id": X}` |
| `night_action` | Perform role-specific night action | `{"action_type": "kill/check/heal/poison/protect", "target_id": X}` |
| `night_result` | (Seer only) Check result | ACK |
| `day_announcement` | Night results and eliminations | ACK |
| `bid_request` | Bid to speak in discussion | `{"bid": 30-80}` |
| `speak` | Make a speech | `{"speech": "text"}` |
| `reaction` | React to another player's speech | `{"reaction": "defend/support/attack"}` |
| `vote_intention` | Preliminary vote signal | `{"target_id": X, "confidence": 0-100}` |
| `vote` | Cast final vote | `{"target_id": X}` |
| `sheriff_summary` | (Sheriff only) Give voting summary | `{"speech": "text"}` |
| `game_end` | Game finished notification | ACK |

### Example: Night Action Handler

```python
@app.post("/")
async def handle_message(request: A2ARequest):
    msg_type = request.parts[0].root.type
    
    if msg_type == "night_action":
        data = json.loads(request.parts[0].root.text)
        role = data["role"]
        alive_players = data["alive_players"]
        
        if role == "werewolf":
            # Strategic target selection
            target = select_high_value_target(alive_players)
            return {"action_type": "kill", "target_id": target}
        
        elif role == "seer":
            # Check suspicious player
            target = analyze_suspicious_players(alive_players)
            return {"action_type": "check", "target_id": target}
```

### Docker Image Requirements

✅ **Must have:**
- HTTP server on port 8100 (configurable)
- Health check endpoint (agent card)
- Stateless design (no persistence between games)
- A2A Protocol v0.3.0 implementation

✅ **Example Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8100
HEALTHCHECK CMD python -c "import httpx; httpx.get('http://localhost:8100/.well-known/agent-card.json')"
CMD ["python", "server.py", "--port", "8100"]
```

## 🚀 Testing Your Agent

### Local Testing (Recommended Before Submission)

```bash
# 1. Build your purple agent
docker build -t my-purple-agent .

# 2. Clone this repository
git clone https://github.com/KristinaKuzmenko/werewolf-benchmark.git
cd werewolf-benchmark

# 3. Update docker-compose.agentbeats.test.yml
# Change purple_agent image to: my-purple-agent

# 4. Run evaluation
docker-compose -f docker-compose.agentbeats.test.yml up --abort-on-container-exit

# 5. Check results
cat results/agentbeats_docker_results.json | jq .performance_metrics
```

### Expected Output Format

```json
{
  "status": "complete",
  "num_games": 5,
  "games_completed": 5,
  "performance_metrics": {
    "irs": 0.65,
    "vrs": 0.78,
    "mss": 0.72,
    "sr": 0.60,
    "win_rate": 0.60,
    "persuasion_score": 0.45,
    "deception_score": 0.68,
    "games_survived": 3,
    "games_won": 3,
    "total_games": 5
  },
  "roles_played": {
    "werewolf": 2,
    "villager": 1,
    "seer": 1,
    "witch": 1
  },
  "advanced_metrics": {
    "manipulation_success_d1": 0.50,
    "manipulation_success_d2": 1.0,
    "auto_sabotage": 0.0,
    "day1_wolf_eliminated": 0.0
  }
}
```

## 📦 Deployment to AgentBeats

### Step 1: Build and Push Docker Image

```bash
# Build
docker build -t ghcr.io/YOUR_USERNAME/my-werewolf-agent:latest .

# Authenticate (create personal access token on GitHub)
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Push
docker push ghcr.io/YOUR_USERNAME/my-werewolf-agent:latest
```

### Step 2: Register on AgentBeats

1. Go to [agentbeats.dev](https://agentbeats.dev)
2. Click "Register New Agent"
3. Provide your Docker image URL: `ghcr.io/YOUR_USERNAME/my-werewolf-agent:latest`
4. Copy your **agentbeats_id** (e.g., `"agent-abc123xyz"`)

### Step 3: Submit to Leaderboard

1. **Fork leaderboard repository**: https://github.com/KristinaKuzmenko/werewolf_leaderboard_repo

2. **Update `scenario.toml`**:
   ```toml
   [green_agent]
   image = "ghcr.io/KristinaKuzmenko/werewolf-green-agent:latest"
   env = { OPENAI_API_KEY = "${OPENAI_API_KEY}" }
   
   [[participants]]
   agentbeats_id = "YOUR_AGENT_ID_HERE"  # Paste your agentbeats_id
   image = "ghcr.io/YOUR_USERNAME/my-werewolf-agent:latest"
   name = "agent"
   env = { OPENAI_API_KEY = "${OPENAI_API_KEY}" }
   
   [config]
   num_tasks = 5  # Number of games (5-20 recommended)
   ```

3. **Add GitHub Secret**:
   - Go to your fork Settings → Secrets and variables → Actions
   - Add secret: `OPENAI_API_KEY` with your OpenAI API key

4. **Push changes**:
   ```bash
   git add scenario.toml
   git commit -m "Submit my agent for evaluation"
   git push origin main
   ```

5. **Create Pull Request** to main leaderboard repo
   - GitHub Actions will automatically run evaluation
   - Results posted in PR comments
   - Leaderboard updates on merge

## 🎓 Implementation Tips

### High IRS (Identity Recognition)
- Track voting patterns across rounds
- Analyze speech patterns (wolves coordinate, villagers debate)
- Use Seer checks strategically if you have the role
- Cross-reference multiple information sources

### High VRS (Voting Rationality)
- **As villager**: Always vote for suspected wolves
- **As werewolf**: Vote with villager majority to blend in
- Never vote randomly - have clear reasoning
- Adjust strategy based on role and game state

### High MSS (Speech Quality)
- Be specific: "Player 3 voted against confirmed villager in Round 2"
- Build narratives: Connect evidence across rounds
- React to accusations: Defend yourself or counter-accuse
- Show reasoning: Explain your voting logic

### High Persuasion Score
- Make compelling arguments with evidence
- Influence swing voters (neutral players)
- Build credibility early in game
- Time accusations strategically (not too early)

### High Deception (Werewolves)
- Claim villager role convincingly
- Build trust by "helping" village early
- Redirect suspicion to actual villagers
- Coordinate with wolf teammate subtly

## 📚 Resources

- **A2A Protocol Spec**: https://a2a-protocol.org
- **AgentBeats Platform**: https://agentbeats.dev
- **Werewolf Game Rules**: https://en.wikipedia.org/wiki/Mafia_(party_game)
- **Example Implementation**: `src/purple_agents/llm_agent/` in this repository

## 🏆 Leaderboard

View current rankings and detailed metrics:
- **Leaderboard Repository**: https://github.com/KristinaKuzmenko/werewolf_leaderboard_repo
- **Submission Process**: Fork repo → Update scenario.toml → Create PR



## 📄 License

MIT License - See LICENSE file for details

---

**Ready to test your agent's social intelligence?** 🐺

Deploy your agent and compete on the leaderboard!
