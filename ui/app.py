"""
ui/app.py
=========
Streamlit UI for the Solar Grid Optimization Agent.
Run with: streamlit run ui/app.py
"""

import json
import os
import sys
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname('agent/'), ".."))

from agent.evaluator import evaluate
from agent.graph import run_agent


# PAGE CONFIG

st.set_page_config(
    page_title="Solar Grid AI Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  .main { background-color: #0f1117; }
  .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }

  /* Risk badge */
  .risk-high   { background:#ff4444; color:white; padding:4px 12px;
                 border-radius:20px; font-weight:700; font-size:0.9rem; }
  .risk-medium { background:#ff9900; color:white; padding:4px 12px;
                 border-radius:20px; font-weight:700; font-size:0.9rem; }
  .risk-low    { background:#00cc66; color:white; padding:4px 12px;
                 border-radius:20px; font-weight:700; font-size:0.9rem; }

  /* Action card */
  .action-card {
    background: rgba(255,255,255,0.05);
    border-left: 4px solid #00aaff;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.92rem;
  }
  .strat-card {
    background: rgba(255,255,255,0.05);
    border-left: 4px solid #00cc66;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.92rem;
  }

  /* Score ring */
  .score-box {
    text-align: center;
    font-size: 3rem;
    font-weight: 900;
    color: #00aaff;
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR — INPUT SECTION
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/solar-panel.png",
    width=70
)
st.sidebar.title("⚡ Solar Grid Agent")
st.sidebar.caption("Agentic AI · LangGraph · RAG · Mistral")
st.sidebar.markdown("---")

input_method = st.sidebar.radio(
    "Input Method",
    ["📊 Use Demo Data", "✏️ Manual Entry", "📁 Upload CSV"],
    index=0
)

weather_input = st.sidebar.text_area(
    "Weather Summary (optional)",
    placeholder="e.g. Partly cloudy, 28°C, moderate wind from SW",
    height=80
)

api_key = st.sidebar.text_input(
    "Mistral API Key (optional)",
    type="password",
    help="Get a free key at mistral.ai. Leave blank for rule-based fallback."
)
if api_key:
    os.environ["MISTRAL_API_KEY"] = api_key

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About:** Milestone 2 of the Solar Energy AI project. "
    "Uses LangGraph + FAISS RAG + Mistral LLM for grid optimization."
)

# ---------------------------------------------------------------------------
# DEMO DATA
# ---------------------------------------------------------------------------
DEMO_FORECAST = [
    {"hour": "06:00", "generation_kw": 45},
    {"hour": "07:00", "generation_kw": 180},
    {"hour": "08:00", "generation_kw": 420},
    {"hour": "09:00", "generation_kw": 670},
    {"hour": "10:00", "generation_kw": 880},
    {"hour": "11:00", "generation_kw": 950},
    {"hour": "12:00", "generation_kw": 980},
    {"hour": "13:00", "generation_kw": 960},
    {"hour": "14:00", "generation_kw": 870},
    {"hour": "15:00", "generation_kw": 680},
    {"hour": "16:00", "generation_kw": 380},
    {"hour": "17:00", "generation_kw": 150},
    {"hour": "18:00", "generation_kw": 40},
    {"hour": "19:00", "generation_kw": 0},
]


# MAIN PAGE

st.title("🌞 Solar Grid AI Optimization Agent")
st.caption("Powered by LangGraph · FAISS RAG · Open-Source LLMs")
st.markdown("---")

# --- Build forecast DataFrame from selected input method ---
forecast_df = None

if input_method == "📊 Use Demo Data":
    forecast_df = pd.DataFrame(DEMO_FORECAST)
    st.info("Using built-in demo forecast (14-hour sunny day with afternoon ramp-down).")

elif input_method == "✏️ Manual Entry":
    st.subheader("Enter Hourly Forecast")
    num_hours = st.slider("Number of hours", 4, 24, 12)
    rows = []
    cols = st.columns(3)
    for i in range(num_hours):
        hour_label = f"{(6 + i) % 24:02d}:00"
        with cols[i % 3]:
            val = st.number_input(
                f"{hour_label}", min_value=0.0, max_value=5000.0,
                value=float(max(0, 500 * abs(6 - abs(i - 6)))),
                key=f"hr_{i}"
            )
            rows.append({"hour": hour_label, "generation_kw": val})
    forecast_df = pd.DataFrame(rows)

elif input_method == "📁 Upload CSV":
    uploaded = st.file_uploader(
        "Upload CSV with columns: hour, generation_kw",
        type=["csv"]
    )
    if uploaded:
        forecast_df = pd.read_csv(uploaded)
        if "hour" not in forecast_df.columns or "generation_kw" not in forecast_df.columns:
            st.error("CSV must have columns: 'hour' and 'generation_kw'")
            forecast_df = None
        else:
            st.success(f"Loaded {len(forecast_df)} rows from CSV.")


# FORECAST PREVIEW CHART
# ---------------------------------------------------------------------------
if forecast_df is not None:
    st.subheader("📈 Forecast Preview")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df["hour"],
        y=forecast_df["generation_kw"],
        mode="lines+markers",
        line=dict(color="#00aaff", width=3),
        marker=dict(size=7),
        fill="tozeroy",
        fillcolor="rgba(0, 170, 255, 0.12)",
        name="Solar Generation (kW)"
    ))
    fig.update_layout(
        template="plotly_dark",
        height=280,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Hour",
        yaxis_title="Generation (kW)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- RUN AGENT BUTTON ---
    st.markdown("---")
    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("🚀 Run AI Agent", type="primary", use_container_width=True)
    with col_note:
        st.caption(
            "The agent runs a 5-node LangGraph pipeline: "
            "Input → Risk → RAG → Plan → Output"
        )

    if run_clicked:
        forecast_list = forecast_df.to_dict(orient="records")


        with st.spinner("🔧 Loading knowledge base (first run builds FAISS index)..."):
            from rag.knowledge_base import load_knowledge_base
            load_knowledge_base()

        # Run the agent
        progress = st.progress(0, text="Running agent pipeline...")
        node_labels = [
            "Processing input...",
            "Analysing risk...",
            "Retrieving knowledge...",
            "Planning actions...",
            "Generating output..."
        ]


        import threading
        result_holder = {}

        def _run():
            result_holder["output"] = run_agent(
                forecast_list,
                weather_summary=weather_input
            )

        t = threading.Thread(target=_run)
        t.start()
        for i, label in enumerate(node_labels):
            time.sleep(0.4)
            progress.progress((i + 1) * 20, text=label)
        t.join()
        progress.empty()

        output = result_holder.get("output", {})
        if not output:
            st.error("Agent returned empty output. Check logs.")
            st.stop()


        st.session_state["last_output"] = output
        st.session_state["last_input"]  = {
            "raw_forecast": forecast_list,
            "weather_summary": weather_input
        }


# DISPLAY RESULTS (if available in session state)

if "last_output" in st.session_state:
    output   = st.session_state["last_output"]
    metadata = output.get("metadata", {})
    risk     = metadata.get("risk_level", "UNKNOWN")

    # --- RUN EVALUATION ---
    eval_result = evaluate(
        st.session_state.get("last_input", {}),
        output
    )

    st.markdown("---")
    st.header("📋 Agent Output")

    # ---- TOP METRICS ROW ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Risk Level", risk)
    with c2:
        st.metric(
            "Variability Index",
            f"{metadata.get('variability_index', 0.0):.4f}"
        )
    with c3:
        st.metric(
            "Peak Generation",
            f"{metadata.get('peak_generation_kw', 0):.0f} kW"
        )
    with c4:
        st.metric(
            "Quality Score",
            f"{eval_result['quality_score']}/100"
        )

    st.markdown("---")

    # ---- TWO-COLUMN LAYOUT ----
    left, right = st.columns(2)

    with left:
        st.subheader("📝 Forecast Summary")
        st.write(output.get("forecast_summary", "N/A"))

        st.subheader("⚠️ Risk Analysis")
        st.write(output.get("risk_analysis", "N/A"))

        if metadata.get("risk_periods"):
            st.subheader("🕐 Risk Periods")
            for rp in metadata["risk_periods"]:
                st.warning(rp)

    with right:
        st.subheader("⚡ Immediate Grid Actions")
        for i, action in enumerate(output.get("grid_actions", []), 1):
            st.markdown(
                f'<div class="action-card">⚡ <b>Action {i}:</b> {action}</div>',
                unsafe_allow_html=True
            )

        st.subheader("🔧 Optimization Strategies")
        for i, strat in enumerate(output.get("optimization_strategies", []), 1):
            st.markdown(
                f'<div class="strat-card">🔧 <b>Strategy {i}:</b> {strat}</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ---- REFERENCES ----
    st.subheader("📚 Knowledge References")
    refs = output.get("references", [])
    if refs:
        for ref in refs:
            st.markdown(f"• {ref}")
    else:
        st.write("No references retrieved.")

    # ---- EXPLAINABILITY ----
    with st.expander("🔍 Agent Explainability Report", expanded=False):
        st.code(eval_result["explanation"], language=None)

    # ---- RAW JSON ----
    with st.expander("📄 Raw JSON Output", expanded=False):
        clean_output = {k: v for k, v in output.items() if not k.startswith("_")}
        st.json(clean_output)

    # ---- EVALUATION ----
    with st.expander("📊 Evaluation & Quality Check", expanded=False):
        st.subheader(f"Quality Score: {eval_result['quality_score']}/100")

        # Completeness table
        completeness = eval_result["completeness"]
        df_check = pd.DataFrame([
            {"Field": k, "Status": "✅ Present" if v else "❌ Missing"}
            for k, v in completeness.items()
        ])
        st.table(df_check)

        st.write(f"**LLM Used:** {metadata.get('llm_used', 'unknown')}")
        st.write(f"**Processing Time:** {metadata.get('processing_time_sec', 0):.2f}s")
        st.write(f"**Status:** {metadata.get('status', 'unknown')}")

    # ---- RAMP RATE CHART ----
    if "last_input" in st.session_state:
        with st.expander("📉 Generation Ramp Rate Chart", expanded=False):
            fc = st.session_state["last_input"].get("raw_forecast", [])
            if len(fc) > 1:
                hours = [h["hour"] for h in fc[1:]]
                ramps = [
                    abs(fc[i]["generation_kw"] - fc[i - 1]["generation_kw"])
                    for i in range(1, len(fc))
                ]
                fig2 = px.bar(
                    x=hours, y=ramps,
                    labels={"x": "Hour", "y": "Ramp Rate (kW/hr)"},
                    title="Hourly Ramp Rate (absolute change)",
                    color=ramps,
                    color_continuous_scale=["green", "orange", "red"],
                    template="plotly_dark"
                )
                st.plotly_chart(fig2, use_container_width=True)

else:
    if forecast_df is None:
        st.info("👆 Select an input method in the sidebar to get started.")
    else:
        st.info("👆 Click **Run AI Agent** to analyse the forecast above.")
