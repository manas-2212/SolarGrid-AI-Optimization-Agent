# ⚡ Solar Grid AI Optimization Agent — Milestone 2

Agentic AI system that takes solar generation forecasts and produces structured grid optimization recommendations using **LangGraph**, **FAISS RAG**, and **Mistral LLM**.

---

## Deployed Link - https://solargrid-ai-optimization-agent.streamlit.app/

---

## 🗂️ Project Structure

```
solar_grid_agent/
├── app.py                      
├── requirements.txt
├── agent/
│   ├── __init__.py
│   ├── graph.py                # langraph pipeline
│   └── evaluator.py            # Quality scoring + explainability + logging
├── rag/
│   ├── __init__.py
│   ├── knowledge_base.py       # FAISS vector store + document ingestion
│   └── faiss_index/            # Auto-created on first run
├── ui/
│   └── app.py                  # Streamlit interface ---> UI
├── data/
│   ├── demo_input.json         # Sample forecast input
│   └── expected_output.json    # Expected structured output
└── logs/
    └── agent_runs.jsonl        # Auto-created run log
```

---

## 🚀 Local Setup (Step-by-Step)

### Step 1 — Clone / Download the project
```bash
git clone <your-repo-url>
cd solar_grid_agent
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — (Optional) Set Mistral API key
Get a FREE key at https://console.mistral.ai/
```bash
export MISTRAL_API_KEY="your_key_here"   # Linux/Mac
set MISTRAL_API_KEY=your_key_here        # Windows
```
If no key is set, the agent uses the built-in rule-based fallback — still fully functional.

### Step 5 — Build the FAISS knowledge base
```bash
python rag/knowledge_base.py
```

### Step 6 — Run the Streamlit UI
```bash
streamlit run ui/app.py
```
Open http://localhost:8501 in your browser.

### Step 7 — (Optional) Test the agent directly
```bash
python agent/graph.py
```

---

## ☁️ Deploy to Hugging Face Spaces (FREE)

### Step 1 — Create a new Space
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. **SDK: Streamlit** | Hardware: CPU Basic (FREE)
4. Name it: `solar-grid-agent`

### Step 2 — Push your code
```bash
# In your project folder
git init
git add .
git commit -m "Solar Grid Agent - Milestone 2"

# Add HF remote (replace YOUR_USERNAME)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/solar-grid-agent
git push hf main
```

### Step 3 — Set API key as a Secret (optional)
In HF Space settings → Secrets → Add `MISTRAL_API_KEY`

### Step 4 — Your Space is live!
URL: `https://huggingface.co/spaces/YOUR_USERNAME/solar-grid-agent`

> **Note**: HF Spaces' `app.py` must be at the root. Our root `app.py` launches `ui/app.py`. 

---

## ☁️ Deploy to Streamlit Cloud (FREE)

### Step 1 — Push to GitHub
```bash
git init && git add . && git commit -m "Solar Grid Agent"
git remote add origin https://github.com/YOUR_USERNAME/solar-grid-agent.git
git push -u origin main
```

### Step 2 — Connect to Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. "New app" → select your repo
4. **Main file path**: `ui/app.py`
5. Deploy!

### Step 3 — Add Secrets
In app settings → Secrets:
```toml
MISTRAL_API_KEY = "your_key_here"
```

---

## 🧠 LangGraph Architecture

```
Input → [Node 1: input_processor] 
             ↓
       [Node 2: risk_analyzer]    ← computes VI, detects ramp events
             ↓
       [Node 3: rag_retriever]    ← FAISS similarity search
             ↓
       [Node 4: planner]          ← LLM (Mistral) or rule-based fallback
             ↓
       [Node 5: output_generator] → Structured JSON
```

---

## 📤 Output Schema

```json
{
  "forecast_summary":        "string",
  "risk_analysis":           "string",
  "grid_actions":            ["action1", "action2", "action3"],
  "optimization_strategies": ["strategy1", "strategy2", "strategy3"],
  "references":              ["doc1", "doc2"],
  "metadata": {
    "risk_level":         "LOW|MEDIUM|HIGH",
    "variability_index":  0.278,
    "risk_periods":       ["..."],
    "llm_used":           "mistral-7b|rule-based",
    "status":             "success|error"
  }
}
```

---

## 🆓 Free-Tier LLMs

| Provider | Model | Free Tier |
|----------|-------|-----------|
| **Mistral AI** | `open-mistral-7b` | ✅ Yes — get key at mistral.ai |
| HuggingFace Inference | `mistralai/Mistral-7B-Instruct-v0.2` | ✅ Yes — slow |
| Groq | `llama3-8b-8192` | ✅ Yes — very fast |
| **Rule-based** | Built-in fallback | ✅ No key needed |

To switch to Groq, change `MISTRAL_API_URL` in `agent/graph.py` to:
```
https://api.groq.com/openai/v1/chat/completions
```
And set `GROQ_API_KEY`.

---

## ✅ Evaluation & Quality

The `agent/evaluator.py` module:
- Checks all 5 required output fields are present
- Scores output 0–100 on completeness + content quality
- Generates a step-by-step explainability report
- Logs every run to `logs/agent_runs.jsonl`

---

## 🔮 Connect to Milestone 1

If your ML model (Scikit-learn) outputs forecasts as a list:
```python
predictions = model.predict(X_test)   # numpy array of kW values
forecast = [
    {"hour": f"{6+i:02d}:00", "generation_kw": float(p)}
    for i, p in enumerate(predictions)
]
from agent.graph import run_agent
result = run_agent(forecast, weather_summary="Clear sky, 30°C")
```
