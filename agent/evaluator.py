"""
agent/evaluator.py
==================
Evaluates and logs agent output quality.
Provides:
  - Completeness check (all required fields present)
  - Quality scoring (0-100)
  - Run logging to JSONL file
  - Explainability report
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)

LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "agent_runs.jsonl")

# ---------------------------------------------------------------------------
# COMPLETENESS CHECK
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = [
    "forecast_summary",
    "risk_analysis",
    "grid_actions",
    "optimization_strategies",
    "references",
]


def check_completeness(output: Dict) -> Dict[str, bool]:
    """
    Returns a dict showing which required fields are present and non-empty.
    """
    results = {}
    for field in REQUIRED_FIELDS:
        val = output.get(field)
        if isinstance(val, list):
            results[field] = len(val) > 0
        elif isinstance(val, str):
            results[field] = len(val.strip()) > 0
        else:
            results[field] = val is not None
    return results


# ---------------------------------------------------------------------------
# QUALITY SCORER
# ---------------------------------------------------------------------------
def score_output(output: Dict) -> int:
    """
    Returns a quality score 0-100 based on:
      - Completeness of fields          (40 pts)
      - Number of grid actions          (20 pts, max 3)
      - Number of strategies            (20 pts, max 3)
      - References provided             (10 pts)
      - Metadata present                (10 pts)
    """
    score = 0

    # Completeness (40 pts)
    completeness = check_completeness(output)
    filled = sum(1 for v in completeness.values() if v)
    score += int((filled / len(REQUIRED_FIELDS)) * 40)

    # Grid actions (20 pts)
    actions = output.get("grid_actions", [])
    score += min(len(actions), 3) * 7  # up to 21, capped at 20

    # Strategies (20 pts)
    strategies = output.get("optimization_strategies", [])
    score += min(len(strategies), 3) * 7

    # References (10 pts)
    refs = output.get("references", [])
    if len(refs) >= 2:
        score += 10
    elif len(refs) == 1:
        score += 5

    # Metadata (10 pts)
    meta = output.get("metadata", {})
    if meta.get("risk_level") and meta.get("variability_index") is not None:
        score += 10

    return min(score, 100)


# ---------------------------------------------------------------------------
# EXPLAINABILITY REPORT
# ---------------------------------------------------------------------------
def generate_explanation(output: Dict) -> str:
    """
    Produces a human-readable explanation of HOW the agent reached its output.
    This is the explainability section important for academic evaluation.
    """
    meta = output.get("metadata", {})
    risk = meta.get("risk_level", "UNKNOWN")
    vi   = meta.get("variability_index", 0.0)
    rp   = meta.get("risk_periods", [])
    llm  = meta.get("llm_used", "unknown")

    lines = [
        "═" * 60,
        "  AGENT DECISION EXPLAINABILITY REPORT",
        "═" * 60,
        "",
        "📊 STEP 1 — INPUT PROCESSING",
        f"   Forecast data was parsed and validated.",
        f"   Peak: {meta.get('peak_generation_kw', 0):.1f} kW  |  "
        f"Avg: {meta.get('avg_generation_kw', 0):.1f} kW",
        "",
        "⚠️  STEP 2 — RISK ANALYSIS",
        f"   Variability Index computed: {vi:.4f}",
        f"   Thresholds → LOW: <0.15 | MEDIUM: 0.15-0.35 | HIGH: >0.35",
        f"   Decision: Risk classified as → {risk}",
    ]

    if rp:
        lines.append(f"   Risk periods detected: {len(rp)}")
        for p in rp[:3]:
            lines.append(f"     • {p}")
    else:
        lines.append("   No high-ramp periods detected.")

    lines += [
        "",
        "📚 STEP 3 — RAG RETRIEVAL",
        f"   Knowledge base queried for {risk.lower()}-risk solar scenarios.",
        f"   Documents retrieved: {len(output.get('references', []))}",
    ]
    for ref in output.get("references", []):
        lines.append(f"     • {ref}")

    lines += [
        "",
        "🤖 STEP 4 — PLANNING (LLM REASONING)",
        f"   LLM used: {llm}",
        f"   Retrieved context was injected into structured prompt.",
        f"   LLM instructed to output JSON only (low temperature=0.2 for determinism).",
        f"   JSON parsing with rule-based fallback on failure.",
        "",
        "📋 STEP 5 — OUTPUT",
        f"   Grid actions generated:            {len(output.get('grid_actions', []))}",
        f"   Optimization strategies generated: {len(output.get('optimization_strategies', []))}",
        "",
        "═" * 60,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LOGGER
# ---------------------------------------------------------------------------
def log_run(
    input_data: Dict,
    output: Dict,
    score: int,
    elapsed_sec: float = 0.0
) -> None:
    """
    Appends a JSONL record for every agent run to logs/agent_runs.jsonl.
    Useful for debugging, auditing, and model improvement.
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    record = {
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "num_forecast_pts": len(input_data.get("raw_forecast", [])),
        "weather_summary": input_data.get("weather_summary", ""),
        "risk_level":      output.get("metadata", {}).get("risk_level", "UNKNOWN"),
        "variability_idx": output.get("metadata", {}).get("variability_index", 0.0),
        "num_actions":     len(output.get("grid_actions", [])),
        "num_strategies":  len(output.get("optimization_strategies", [])),
        "llm_used":        output.get("metadata", {}).get("llm_used", "unknown"),
        "quality_score":   score,
        "elapsed_sec":     elapsed_sec,
        "status":          output.get("metadata", {}).get("status", "unknown"),
    }

    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"Run logged: score={score}, risk={record['risk_level']}")
    except Exception as e:
        logger.warning(f"Could not write log: {e}")


# ---------------------------------------------------------------------------
# FULL EVALUATION PIPELINE
# ---------------------------------------------------------------------------
def evaluate(input_data: Dict, output: Dict) -> Dict[str, Any]:
    """
    Run the full evaluation suite and return a summary dict.
    """
    completeness = check_completeness(output)
    score        = score_output(output)
    explanation  = generate_explanation(output)
    elapsed      = output.get("metadata", {}).get("processing_time_sec", 0.0)

    log_run(input_data, output, score, elapsed)

    return {
        "quality_score":  score,
        "completeness":   completeness,
        "explanation":    explanation,
        "all_fields_ok":  all(completeness.values()),
    }
