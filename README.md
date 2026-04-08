# 🛡️ FakeNews Detection OpenEnv

> Social Media Fake News Detection Environment with Multi-Signal Alert System  
> An OpenEnv-compliant RL environment for training AI agents on fake news detection.
---

## 🌍 Environment Description & Motivation

Fake news is one of the most pressing challenges of our time. Social media platforms are flooded with disinformation, ranging from obvious conspiracy theories to sophisticated influence operations that blend fact and fiction.

This environment trains AI agents to:
- **Extract claims** from social media posts
- **Verify sources** against a credibility database
- **Cross-check facts** against a deterministic knowledge base
- **Detect linguistic patterns** used in fake news
- **Generate alerts** (GREEN / YELLOW / RED) with confidence scores

Unlike toy environments, this models real moderation workflows used by fact-checkers and trust & safety teams.

---

## 📁 Project Structure

```
fakenews_env/
├── env.py              # Core environment logic (reset/step/state)
├── models.py           # Pydantic typed models (Action, Observation, Reward)
├── tasks.py            # Task definitions + knowledge base + source credibility
├── rewards.py          # Reward calculator (partial rewards, penalties)
├── grader.py           # Deterministic grader (0.0–1.0 scores)
├── inference.py        # Inference script with [START]/[STEP]/[END] logs
├── main.py             # Server entrypoint
├── server/
│   ├── app.py          # FastAPI server (/reset, /step, /state)
│   └── __init__.py
├── Dockerfile          # Container deployment
├── requirements.txt    # Python dependencies
├── openenv.yaml        # OpenEnv spec metadata
├── pyproject.toml      # Package metadata
├── manual_test.py      # Local test suite
└── README.md
```

---

## 🎮 Action Space

| Action | Description | Parameters |
|--------|-------------|------------|
| `analyze_claim` | Extract + analyze a claim from the post | `target` (claim text) |
| `check_source` | Check source credibility | `target` (source URL/name) |
| `cross_verify` | Cross-verify claim vs knowledge base | `target` (claim text) |
| `raise_alert` | Raise final alert with label | `final_label`, `confidence` |
| `mark_safe` | Mark post as real/safe | `final_label=real`, `confidence` |

---

## 👁️ Observation Space

```json
{
  "post_id": "post_001",
  "post_text": "Full social media post text...",
  "task_description": "Task objective...",
  "step_number": 2,
  "max_steps": 6,
  "claims_extracted": ["claim 1", "claim 2"],
  "sources_checked": [{"source": "naturalnews.com", "credibility": 0.05, "tier": "misinformation"}],
  "patterns_detected": ["SHOCKING", "WAKE UP SHEEPLE"],
  "cross_verifications": [{"claim": "...", "verdict": "false", "contradiction_found": true}],
  "current_fake_score": 0.82,
  "current_alert": "RED",
  "available_actions": ["analyze_claim", "check_source", "cross_verify", "raise_alert", "mark_safe"],
  "message": "Step 2/6. Current fake score: 0.82. Alert: RED."
}
```

---

## 📊 Tasks

### Task 1: Basic Fake News Detection (Easy)
- **Post**: Obvious conspiracy post with naturalnews.com, "SHARE BEFORE DELETED", "WAKE UP SHEEPLE"
- **Ground Truth**: `fake` / `RED`
- **Max Steps**: 6
- **Pass Threshold**: 0.65
- **Challenge**: Agent must detect clear patterns and blacklisted source

### Task 2: Partial Truth & Suspicious Content (Medium)
- **Post**: Mixes real facts (exercise helps mental health) with false 5G claims and unverified "independent study"
- **Ground Truth**: `likely_fake` / `YELLOW`
- **Max Steps**: 8
- **Pass Threshold**: 0.60
- **Challenge**: Must not over-alert or under-alert; partial truth detection required

### Task 3: Sophisticated Disinformation (Hard)
- **Post**: Uses IPCC/climate facts as credibility anchors, then embeds false fluoride brain damage claim, cites `beforeitsnews.com` and `telegram_channel`
- **Ground Truth**: `fake` / `RED`
- **Max Steps**: 10
- **Pass Threshold**: 0.55
- **Challenge**: Must not be fooled by true anchors; requires source credibility + claim isolation

---

## 🏆 Reward System

| Component | Weight | Description |
|-----------|--------|-------------|
| Label accuracy | 40% | Compatibility score between predicted and ground truth label |
| Alert accuracy | 25% | Compatibility score between predicted and ground truth alert |
| Evidence quality | 20% | Claims found, sources checked, patterns detected |
| Action coverage | 10% | Required actions completed |
| Efficiency | 5% | Penalty for wasted steps |

**Severe Penalties:**
- Marking `fake` as `REAL` (green): capped at 0.10 score
- Marking `RED` alert post as `GREEN`: capped at 0.10 score
- Overconfident wrong answers: confidence × error penalty

---

## 🚀 Setup & Usage

### Local Setup

```bash
# Clone and install
git clone <your-repo>
cd fakenews_env
pip install -r requirements.txt

# Run manual tests
python manual_test.py

# Start server
python main.py

# Test API
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'
```

### Docker

```bash
docker build -t fakenews-openenv .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  fakenews-openenv
```

### Inference

```bash
# Set environment variables
export API_BASE_URL="https://api-inference.huggingface.co/v1/"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"

# Run inference
python inference.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Environment info |
| GET | `/health` | Health check |
| POST | `/reset` | Reset environment |
| POST | `/step` | Execute action |
| GET/POST | `/state` | Get current state |
| GET | `/tasks` | List all tasks |

### Example API Usage

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_hard"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "analyze_claim",
    "target": "vaccines contain microchips",
    "reasoning": "Checking main claim",
    "task_id": "task_easy"
  }'

# Final verdict
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "raise_alert",
    "final_label": "fake",
    "confidence": 0.95,
    "task_id": "task_easy"
  }'
```

---

## 📈 Baseline Scores

| Task | Difficulty | Baseline Score | Pass Threshold |
|------|-----------|---------------|----------------|
| task_easy | Easy | ~0.72 | 0.65 |
| task_medium | Medium | ~0.63 | 0.60 |
| task_hard | Hard | ~0.57 | 0.55 |

---

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_BASE_URL` | LLM API endpoint | Yes |
| `MODEL_NAME` | Model identifier | Yes |
| `HF_TOKEN` | HuggingFace API key | Yes |
| `PORT` | Server port (default: 7860) | No |

---

## 🧪 Validation

```bash
# OpenEnv validation
openenv validate

# Pre-submission check
bash validate.sh
```

---

## 📝 License

MIT License