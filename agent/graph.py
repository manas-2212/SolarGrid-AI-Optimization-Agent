"""
agent/graph.py
==============
Full LangGraph workflow for the Solar Grid Optimization Agent.

Nodes (in order):
  1. input_processor    → validates + summarises forecast data
  2. risk_analyzer      → detects variability & risk periods
  3. rag_retriever      → fetches relevant KB docs
  4. planner            → generates grid actions + strategies
  5. output_generator   → formats final structured JSON response

State flows from node to node via a typed TypedDict.
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

import requests
from langgraph.graph import END, StateGraph

from rag.knowledge_base import get_retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AGENT STATE  — every field is optional so partial updates work cleanly
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    # --- Inputs ---
    raw_forecast: List[Dict]          # list of {hour, generation_kw, ...}
    weather_summary: str              # optional free-text weather note

    # --- Intermediate ---
    forecast_summary: str
    peak_generation: float
    avg_generation: float
    variability_index: float
    risk_level: str                   # LOW / MEDIUM / HIGH
    risk_periods: List[str]
    retrieved_docs: List[str]

    # --- Final outputs ---
    grid_actions: List[str]
    optimization_strategies: List[str]
    references: List[str]
    final_output: Dict[str, Any]

    # --- Meta ---
    error: Optional[str]
    llm_used: str


# ---------------------------------------------------------------------------
# LLM CALLER  — Mistral free API with local fallback
# ---------------------------------------------------------------------------
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL   = "open-mistral-7b"   # free tier model


def call_llm(prompt: str, system: str = "", max_tokens: int = 700) -> str:
    """
    Calls Mistral free-tier API. Falls back to a rule-based response if:
      - No API key is set
      - The API call fails for any reason
    """
    api_key = os.getenv("MISTRAL_API_KEY", "")

    if api_key:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            resp = requests.post(
                MISTRAL_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MISTRAL_MODEL,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.2,    # low temp → deterministic, factual
                },
                timeout=30
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            logger.info(f"LLM call succeeded ({len(content)} chars)")
            return content

        except Exception as e:
            logger.warning(f"LLM API call failed: {e}. Using rule-based fallback.")

    # -----------------------------------------------------------------------
    # FALLBACK: deterministic rule-based text generation
    # -----------------------------------------------------------------------
    return _rule_based_fallback(prompt)


def _rule_based_fallback(prompt: str) -> str:
    """
    If the LLM is unavailable, produce a structured response from
    keywords in the prompt. This guarantees the pipeline never crashes.
    """
    p = prompt.lower()

    if "risk" in p and "high" in p:
        return (
            "HIGH RISK DETECTED: Significant solar generation variability observed. "
            "Immediate action required: activate spinning reserves and dispatch battery storage. "
            "Notify grid operators of potential ramp event within next 2 hours."
        )
    elif "risk" in p and "medium" in p:
        return (
            "MEDIUM RISK: Moderate variability in solar forecast. "
            "Recommended actions: prepare demand response programs and monitor generation every 15 minutes."
        )
    elif "strateg" in p or "optim" in p:
        return (
            "Optimization strategies: "
            "1. Schedule battery charging during peak solar hours (10:00-14:00). "
            "2. Activate demand response to shift flexible loads. "
            "3. Coordinate with neighboring grid zones for power exchange. "
            "4. Pre-position spinning reserves for evening ramp event."
        )
    else:
        return (
            "Grid operations analysis complete. "
            "Monitor solar output continuously and maintain adequate reserves. "
            "Refer to grid management guidelines for specific action thresholds."
        )


# ---------------------------------------------------------------------------
# NODE 1: INPUT PROCESSOR
# ---------------------------------------------------------------------------
def input_processor_node(state: AgentState) -> AgentState:
    """
    Validates the raw forecast list and computes basic statistics.
    Sets: forecast_summary, peak_generation, avg_generation
    """
    logger.info("[NODE 1] Input Processor")

    forecast = state.get("raw_forecast", [])
    weather   = state.get("weather_summary", "No weather data provided.")

    if not forecast:
        return {**state, "error": "No forecast data provided.", "llm_used": "none"}

    try:
        generations = [float(h.get("generation_kw", 0)) for h in forecast]
        peak_gen    = max(generations)
        avg_gen     = sum(generations) / len(generations)
        total_kwh   = sum(generations)   # assuming each row = 1 hour

        # Find peak hour
        peak_hour_idx = generations.index(peak_gen)
        peak_hour     = forecast[peak_hour_idx].get("hour", f"Hour {peak_hour_idx}")

        summary = (
            f"Forecast covers {len(forecast)} hours. "
            f"Peak generation: {peak_gen:.1f} kW at {peak_hour}. "
            f"Average generation: {avg_gen:.1f} kW. "
            f"Total estimated output: {total_kwh:.1f} kWh. "
            f"Weather context: {weather}"
        )

        logger.info(f"Forecast summary built: peak={peak_gen}, avg={avg_gen:.1f}")

        return {
            **state,
            "forecast_summary": summary,
            "peak_generation":  peak_gen,
            "avg_generation":   avg_gen,
            "error":            None,
            "llm_used":         "none"
        }

    except Exception as e:
        logger.error(f"Input processor error: {e}")
        return {**state, "error": str(e)}


# ---------------------------------------------------------------------------
# NODE 2: RISK ANALYZER
# ---------------------------------------------------------------------------
def risk_analyzer_node(state: AgentState) -> AgentState:
    """
    Computes variability index and identifies high-risk time windows.
    Uses LLM to produce a natural-language risk narrative.
    Sets: variability_index, risk_level, risk_periods, risk_analysis (in final_output later)
    """
    logger.info("[NODE 2] Risk Analyzer")

    if state.get("error"):
        return state

    forecast    = state.get("raw_forecast", [])
    peak_gen    = state.get("peak_generation", 0)
    avg_gen     = state.get("avg_generation", 0)
    generations = [float(h.get("generation_kw", 0)) for h in forecast]

    # --- Variability Index: std dev normalised by peak ---
    if peak_gen > 0 and len(generations) > 1:
        mean  = sum(generations) / len(generations)
        var   = sum((x - mean) ** 2 for x in generations) / len(generations)
        std   = var ** 0.5
        vi    = std / peak_gen
    else:
        vi = 0.0

    # --- Ramp rate detection (MW/hr change between consecutive hours) ---
    risk_periods = []
    for i in range(1, len(generations)):
        delta = abs(generations[i] - generations[i - 1])
        ramp_pct = (delta / peak_gen * 100) if peak_gen > 0 else 0
        if ramp_pct > 20:
            hour_label = forecast[i].get("hour", f"Hour {i}")
            risk_periods.append(
                f"{hour_label}: ramp of {delta:.1f} kW ({ramp_pct:.0f}% of peak)"
            )

    # --- Risk classification ---
    if vi > 0.35 or len(risk_periods) >= 3:
        risk_level = "HIGH"
    elif vi > 0.15 or len(risk_periods) >= 1:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # --- LLM prompt for narrative risk analysis ---
    system_prompt = (
        "You are a solar grid risk analyst. "
        "Respond concisely in 3-4 sentences. Focus on actionable insights."
    )
    user_prompt = (
        f"Analyse this solar forecast risk profile:\n"
        f"- Variability Index: {vi:.3f} (threshold: 0.15=medium, 0.35=high)\n"
        f"- Risk Level: {risk_level}\n"
        f"- Risk Periods: {risk_periods if risk_periods else 'None detected'}\n"
        f"- Peak Generation: {peak_gen:.1f} kW\n"
        f"- Average Generation: {avg_gen:.1f} kW\n\n"
        f"Explain the main risks for grid operators and what they should watch for."
    )

    risk_narrative = call_llm(user_prompt, system=system_prompt, max_tokens=200)

    logger.info(f"Risk analysis done: VI={vi:.3f}, level={risk_level}, periods={len(risk_periods)}")

    return {
        **state,
        "variability_index": round(vi, 4),
        "risk_level":        risk_level,
        "risk_periods":      risk_periods,
        "llm_used":          "mistral-7b" if os.getenv("MISTRAL_API_KEY") else "rule-based",
        # store for final output
        "_risk_narrative":   risk_narrative,
    }


# ---------------------------------------------------------------------------
# NODE 3: RAG RETRIEVER
# ---------------------------------------------------------------------------
def rag_retriever_node(state: AgentState) -> AgentState:
    """
    Builds a retrieval query from the risk profile, fetches relevant
    knowledge base chunks, and stores them for the planner.
    Sets: retrieved_docs, references
    """
    logger.info("[NODE 3] RAG Retriever")

    if state.get("error"):
        return state

    risk_level  = state.get("risk_level", "MEDIUM")
    vi          = state.get("variability_index", 0.0)
    risk_periods = state.get("risk_periods", [])

    # Build a targeted retrieval query
    query_parts = [f"solar grid {risk_level.lower()} risk management"]
    if vi > 0.30:
        query_parts.append("high variability storage dispatch strategies")
    if risk_periods:
        query_parts.append("sudden ramp event grid balancing demand response")
    query_parts.append("optimization recommendations energy storage frequency regulation")

    query = " ".join(query_parts)
    logger.info(f"RAG query: {query}")

    try:
        retriever = get_retriever(k=4)
        docs      = retriever.invoke(query)

        retrieved_texts = []
        references      = []
        for doc in docs:
            retrieved_texts.append(doc.page_content.strip())
            title = doc.metadata.get("title", "Solar Grid Knowledge Base")
            if title not in references:
                references.append(title)

        logger.info(f"Retrieved {len(retrieved_texts)} chunks from {len(references)} documents.")

        return {
            **state,
            "retrieved_docs": retrieved_texts,
            "references":     references,
        }

    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        return {
            **state,
            "retrieved_docs": ["Knowledge base unavailable. Using built-in heuristics."],
            "references":     ["Internal Grid Management Guidelines"],
            "error":          None,   # non-fatal — continue pipeline
        }


# ---------------------------------------------------------------------------
# NODE 4: PLANNER
# ---------------------------------------------------------------------------
def planner_node(state: AgentState) -> AgentState:
    """
    Uses LLM + retrieved context to generate concrete grid_actions
    and optimization_strategies.
    """
    logger.info("[NODE 4] Planner")

    if state.get("error"):
        return state

    risk_level    = state.get("risk_level", "MEDIUM")
    vi            = state.get("variability_index", 0.0)
    risk_periods  = state.get("risk_periods", [])
    forecast_sum  = state.get("forecast_summary", "")
    retrieved     = state.get("retrieved_docs", [])

    # Concatenate top-3 context chunks
    context = "\n\n---\n\n".join(retrieved[:3]) if retrieved else "No context available."

    system_prompt = (
        "You are an expert power grid engineer specialising in solar integration. "
        "You MUST respond with ONLY a valid JSON object. No markdown. No explanation outside JSON. "
        "Use this exact schema:\n"
        '{"grid_actions": ["action1", "action2", "action3"], '
        '"optimization_strategies": ["strategy1", "strategy2", "strategy3"]}\n'
        "Each action/strategy must be specific, actionable, and under 20 words."
    )

    user_prompt = (
        f"FORECAST CONTEXT:\n{forecast_sum}\n\n"
        f"RISK ASSESSMENT:\n"
        f"- Risk Level: {risk_level}\n"
        f"- Variability Index: {vi:.3f}\n"
        f"- Risk Periods: {risk_periods if risk_periods else 'None'}\n\n"
        f"KNOWLEDGE BASE EXCERPTS:\n{context}\n\n"
        f"Generate 3 immediate grid actions and 3 optimization strategies for this situation."
    )

    raw_response = call_llm(user_prompt, system=system_prompt, max_tokens=400)

    # --- Parse JSON from LLM response ---
    grid_actions           = []
    optimization_strategies = []

    try:
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?", "", raw_response).strip().rstrip("` ")
        data  = json.loads(clean)
        grid_actions            = data.get("grid_actions", [])
        optimization_strategies = data.get("optimization_strategies", [])

    except (json.JSONDecodeError, ValueError):
        logger.warning("LLM returned non-JSON. Applying rule-based plan.")
        grid_actions, optimization_strategies = _rule_based_plan(risk_level, vi, risk_periods)

    # Ensure we always have content
    if not grid_actions:
        grid_actions, optimization_strategies = _rule_based_plan(risk_level, vi, risk_periods)

    logger.info(f"Planner produced {len(grid_actions)} actions, {len(optimization_strategies)} strategies.")

    return {
        **state,
        "grid_actions":            grid_actions,
        "optimization_strategies": optimization_strategies,
    }


def _rule_based_plan(risk_level: str, vi: float, risk_periods: list):
    """Deterministic fallback plans indexed by risk level."""
    actions = {
        "HIGH": [
            "Activate spinning reserves immediately (15-20% of solar capacity).",
            "Dispatch battery storage to compensate for ramp-down events.",
            "Initiate demand response — notify industrial consumers to reduce load.",
        ],
        "MEDIUM": [
            "Place gas peaker units on 10-minute standby.",
            "Begin pre-charging battery storage during current solar peak.",
            "Issue 2-hour advance warning to demand response aggregators.",
        ],
        "LOW": [
            "Continue normal grid operations with standard monitoring.",
            "Log current generation data for forecast model improvement.",
            "Schedule battery maintenance during next low-generation window.",
        ],
    }
    strategies = {
        "HIGH": [
            "Deploy fast-response lithium-ion BESS to smooth ramp events.",
            "Enable automatic generation control (AGC) for real-time balancing.",
            "Pre-position pumped hydro for evening peak discharge.",
        ],
        "MEDIUM": [
            "Schedule EV fleet charging during 10:00-14:00 solar peak window.",
            "Use smart inverter reactive power to stabilise voltage.",
            "Coordinate power export to neighboring grids during midday surplus.",
        ],
        "LOW": [
            "Optimise battery state-of-charge for maximum afternoon arbitrage.",
            "Run day-ahead unit commitment with updated solar forecast.",
            "Review curtailment thresholds for next 24-hour period.",
        ],
    }
    level = risk_level if risk_level in actions else "MEDIUM"
    return actions[level], strategies[level]


# ---------------------------------------------------------------------------
# NODE 5: OUTPUT GENERATOR
# ---------------------------------------------------------------------------
def output_generator_node(state: AgentState) -> AgentState:
    """
    Assembles the final structured JSON output.
    This is the single authoritative output of the entire pipeline.
    """
    logger.info("[NODE 5] Output Generator")

    if state.get("error"):
        return {
            **state,
            "final_output": {
                "forecast_summary":        "Error processing forecast.",
                "risk_analysis":           state.get("error", "Unknown error"),
                "grid_actions":            ["Check input data format and retry."],
                "optimization_strategies": ["Resolve input error before proceeding."],
                "references":              [],
                "metadata": {
                    "risk_level":        "UNKNOWN",
                    "variability_index": 0.0,
                    "llm_used":          "none",
                    "status":            "error",
                }
            }
        }

    risk_narrative = state.get("_risk_narrative", "Risk analysis completed.")

    final_output = {
        "forecast_summary":        state.get("forecast_summary", ""),
        "risk_analysis":           risk_narrative,
        "grid_actions":            state.get("grid_actions", []),
        "optimization_strategies": state.get("optimization_strategies", []),
        "references":              state.get("references", []),
        "metadata": {
            "risk_level":        state.get("risk_level", "UNKNOWN"),
            "variability_index": state.get("variability_index", 0.0),
            "risk_periods":      state.get("risk_periods", []),
            "peak_generation_kw": state.get("peak_generation", 0),
            "avg_generation_kw":  state.get("avg_generation", 0),
            "llm_used":           state.get("llm_used", "unknown"),
            "status":             "success",
        }
    }

    logger.info("Output generator: final JSON assembled.")
    return {**state, "final_output": final_output}


# ---------------------------------------------------------------------------
# LANGGRAPH WORKFLOW BUILDER
# ---------------------------------------------------------------------------
def build_agent_graph() -> StateGraph:
    """
    Assembles and compiles the full LangGraph StateGraph.
    Returns a compiled graph ready for .invoke()
    """
    workflow = StateGraph(AgentState)

    # Register all nodes
    workflow.add_node("input_processor",  input_processor_node)
    workflow.add_node("risk_analyzer",    risk_analyzer_node)
    workflow.add_node("rag_retriever",    rag_retriever_node)
    workflow.add_node("planner",          planner_node)
    workflow.add_node("output_generator", output_generator_node)

    # Define the execution edges (linear pipeline)
    workflow.set_entry_point("input_processor")
    workflow.add_edge("input_processor",  "risk_analyzer")
    workflow.add_edge("risk_analyzer",    "rag_retriever")
    workflow.add_edge("rag_retriever",    "planner")
    workflow.add_edge("planner",          "output_generator")
    workflow.add_edge("output_generator", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT (for testing this module directly)
# ---------------------------------------------------------------------------
def run_agent(forecast_data: List[Dict], weather_summary: str = "") -> Dict:
    """
    Public interface: feed forecast data, get structured JSON output.
    """
    graph = build_agent_graph()

    initial_state: AgentState = {
        "raw_forecast":    forecast_data,
        "weather_summary": weather_summary,
    }

    t0     = time.time()
    result = graph.invoke(initial_state)
    elapsed = time.time() - t0

    output = result.get("final_output", {})
    output.setdefault("metadata", {})["processing_time_sec"] = round(elapsed, 2)

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick smoke test
    sample = [
        {"hour": "06:00", "generation_kw": 50},
        {"hour": "07:00", "generation_kw": 200},
        {"hour": "08:00", "generation_kw": 450},
        {"hour": "09:00", "generation_kw": 680},
        {"hour": "10:00", "generation_kw": 850},
        {"hour": "11:00", "generation_kw": 920},
        {"hour": "12:00", "generation_kw": 980},
        {"hour": "13:00", "generation_kw": 950},
        {"hour": "14:00", "generation_kw": 870},
        {"hour": "15:00", "generation_kw": 700},
        {"hour": "16:00", "generation_kw": 400},
        {"hour": "17:00", "generation_kw": 180},
        {"hour": "18:00", "generation_kw": 60},
        {"hour": "19:00", "generation_kw": 0},
    ]

    out = run_agent(sample, weather="Partly cloudy, moderate wind, 28°C")
    print(json.dumps(out, indent=2))
