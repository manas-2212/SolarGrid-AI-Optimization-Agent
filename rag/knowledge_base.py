"""
RAG Knowledge Base Builder
Ingests solar/grid documents, creates embeddings, builds FAISS index.
"""

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document
import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DOMAIN KNOWLEDGE  (embedded directly — no external files needed)
# ---------------------------------------------------------------------------
KNOWLEDGE_DOCUMENTS = [
    {
        "title": "Solar Variability and Grid Impact",
        "content": """
Solar power variability refers to the fluctuations in solar generation caused by changing weather
conditions, cloud cover, and the natural day-night cycle. Variability is classified as:

1. SHORT-TERM VARIABILITY (seconds to minutes): Caused by fast-moving clouds.
   Impact: frequency deviations, voltage fluctuations.
   Mitigation: fast-response battery storage (lithium-ion), flywheel energy storage.

2. MEDIUM-TERM VARIABILITY (hours): Caused by weather fronts, partial overcast conditions.
   Impact: ramping events, unit commitment challenges.
   Mitigation: demand response programs, gas peaker plants on standby, grid-scale batteries.

3. LONG-TERM VARIABILITY (seasonal): Predictable low-generation winters.
   Impact: capacity planning, long-duration storage needs.
   Mitigation: pumped hydro, hydrogen storage, seasonal demand shifting.

KEY METRICS:
- Ramp rate: MW/minute change in output. Values >10% of capacity/min are HIGH risk.
- Variability Index (VI): Standard deviation of 5-min intervals. VI > 0.25 = unstable.
- Forecast Error: MAPE > 15% triggers manual grid operator alerts.

RISK THRESHOLDS:
- LOW RISK: Generation stable ±5%, weather clear, VI < 0.10
- MEDIUM RISK: Generation fluctuating ±10-20%, partly cloudy, VI 0.10-0.25
- HIGH RISK: Generation dropping >30%, storm/heavy cloud, VI > 0.25
"""
    },
    {
        "title": "Grid Balancing Strategies for Solar Integration",
        "content": """
Grid balancing ensures that electricity supply always matches demand in real time.
With high solar penetration, operators must use multiple strategies:

STRATEGY 1 — DEMAND RESPONSE (DR):
- Shift flexible loads (EV charging, HVAC, industrial processes) away from solar ramp-down.
- Industrial DR: pause non-critical machinery 15-30 min before forecast solar dip.
- Residential DR: smart thermostats pre-cool buildings 1 hr before afternoon solar drop.
- Estimated load shift capability: 5-15% of peak demand.

STRATEGY 2 — SPINNING RESERVES:
- Keep conventional generators (gas turbines) synchronized but partially loaded.
- They can ramp up within 10 minutes when solar drops unexpectedly.
- Required spinning reserve = 15-20% of solar capacity during high-variability periods.

STRATEGY 3 — INTERCONNECTION AND POWER TRADING:
- Export excess midday solar to neighboring grids.
- Import power during evening ramp-down ("duck curve" management).
- Requires real-time pricing signals and automatic generation control (AGC).

STRATEGY 4 — VOLTAGE REGULATION:
- Solar inverters must provide reactive power support during cloud events.
- Capacitor banks and STATCOMs stabilize voltage within ±5% of nominal.
- Voltage violations risk equipment damage and outages.

STRATEGY 5 — FREQUENCY REGULATION:
- Normal frequency: 50 Hz (Europe/India) or 60 Hz (Americas).
- Deviation >0.5 Hz triggers automatic protection.
- Solar + battery systems provide synthetic inertia to buffer frequency swings.

DUCK CURVE MANAGEMENT:
The "duck curve" describes the net load shape when solar is high:
- Midday: excess solar → curtailment risk or negative prices.
- Evening ramp: solar drops while demand peaks → requires 2000-3000 MW ramp in 3 hours.
- Action: pre-charge batteries at noon, dispatch at 17:00-20:00.
"""
    },
    {
        "title": "Energy Storage Technologies for Solar Grid Support",
        "content": """
Energy storage is critical for shifting solar generation to match demand patterns.

LITHIUM-ION BATTERIES (BESS):
- Best for: short-duration storage (1-4 hours), frequency regulation, peak shaving.
- Round-trip efficiency: 90-95%.
- Response time: <100 milliseconds (fastest available).
- Cost (2024): $150-250/kWh installed.
- Grid action: Charge 10:00-14:00 (solar peak), discharge 17:00-21:00 (evening peak).
- Degradation: 20% capacity loss after 3000-5000 cycles.

PUMPED HYDRO STORAGE:
- Best for: long-duration (8-20 hours), seasonal balancing, large scale.
- Round-trip efficiency: 70-85%.
- Response time: 1-10 minutes.
- Provides grid inertia (physical flywheel effect).
- Requires suitable topography (two reservoirs, elevation difference).

FLOW BATTERIES (Vanadium Redox):
- Best for: medium-duration (4-12 hours), daily cycling.
- Round-trip efficiency: 65-80%.
- Independent scaling of power (kW) and energy (kWh).
- Long calendar life: 20+ years with electrolyte maintenance.

HYDROGEN STORAGE:
- Best for: seasonal storage, weeks to months.
- Electrolysis efficiency: 60-70%; fuel cell efficiency: 50-60%.
- Combined round-trip: 35-45% (low but acceptable for seasonal balancing).
- Emerging technology; costs declining.

THERMAL STORAGE:
- Molten salt (for CSP plants): stores heat for 6-15 hours.
- Ice storage (buildings): pre-cool during solar peak, reduce AC load in evening.
- Very low cost; underutilized in grid planning.

STORAGE DISPATCH RULES:
1. Charge when solar forecast > 80% capacity AND grid price < $30/MWh.
2. Discharge when solar forecast < 20% capacity OR grid price > $80/MWh.
3. Maintain 20% state-of-charge (SOC) as emergency reserve at all times.
4. Limit charge/discharge rate to 0.5C to maximize battery life.
"""
    },
    {
        "title": "Forecasting Error Management and Uncertainty Handling",
        "content": """
Solar forecast errors are inevitable. Grid operators must build systems that are ROBUST to errors.

FORECAST ERROR TYPES:
- Bias error: systematic over/under-prediction (usually weather model bias).
- Random error: unpredictable, due to cloud movement.
- Ramp error: missing sudden generation changes (most dangerous).

STANDARD ERROR METRICS:
- MAE (Mean Absolute Error): average absolute deviation in MW.
- RMSE (Root Mean Square Error): penalizes large errors more.
- MAPE (Mean Absolute Percentage Error): percentage-based, useful for comparison.
- Skill Score: compares forecast to naive persistence model.

UNCERTAINTY QUANTIFICATION:
- Probabilistic forecasts: provide P10, P50, P90 percentiles instead of single value.
- P10 = pessimistic scenario (plan for this in reliability analysis).
- P90 = optimistic scenario (use for curtailment planning).
- Ensemble forecasts: average of multiple NWP models reduces error 15-25%.

OPERATIONAL RESPONSE TO HIGH UNCERTAINTY:
1. If forecast uncertainty (P90-P10) > 30% of capacity: activate demand response standby.
2. If MAPE > 15%: increase spinning reserves by 10%.
3. If ramp rate forecast > 10% capacity/min: pre-position fast storage (batteries).
4. If morning cloud cover > 70%: delay morning load curtailment; wait for actual generation.

FALLBACK PROTOCOLS:
- Primary: ML-based forecast (LSTM, XGBoost).
- Secondary: Numerical Weather Prediction (NWP) model output.
- Tertiary: Persistence forecast (assume today = yesterday) + manual adjustment.
- Emergency: Manual dispatch based on operator experience.
"""
    },
    {
        "title": "Solar Grid Integration Best Practices and Standards",
        "content": """
Global standards and best practices for integrating solar into the electricity grid.

IEEE STANDARDS:
- IEEE 1547-2018: Standard for interconnection of distributed resources.
  Requires: voltage ride-through, frequency ride-through, reactive power capability.
- IEEE 2030.8: Smart grid interoperability for energy storage.

IEC STANDARDS:
- IEC 61724: Photovoltaic system performance monitoring.
- IEC 61727: Grid connection requirements for PV systems.

GRID CODE REQUIREMENTS (India - CEA Grid Code):
- Solar plants >1 MW must provide reactive power in range 0.95 pf lead/lag.
- Frequency response: reduce output by 40% if frequency > 50.5 Hz.
- Low Voltage Ride Through (LVRT): remain connected during faults.
- Ramp rate limit: maximum 10% of rated capacity per minute.
- Forecasting: Day-ahead and intra-day forecasts mandatory for plants >50 MW.

BEST PRACTICES FOR GRID OPERATORS:
1. FORECASTING: Update solar forecasts every 15-30 minutes using NWP + ML hybrid.
2. SCHEDULING: Day-ahead unit commitment using probabilistic solar forecasts.
3. REAL-TIME: Automatic Generation Control (AGC) with solar+storage coordination.
4. PROTECTION: Set inverter trip settings to match grid code ride-through requirements.
5. COMMUNICATION: SCADA systems with <5 second data latency for solar plants.
6. PLANNING: Study grid hosting capacity before approving new solar interconnections.

CURTAILMENT MANAGEMENT:
- Curtailment = deliberate reduction of solar output to avoid overloading grid.
- Minimize curtailment through: storage, demand response, grid upgrades.
- Curtailment order (merit): first curtail highest-cost, non-contracted solar.
- Track curtailment hours; >5% annual curtailment signals need for grid investment.

ANCILLARY SERVICES FROM SOLAR+STORAGE:
- Frequency Regulation: Batteries respond in <1 second (vs. 5-10 min for gas).
- Voltage Support: Smart inverters provide dynamic reactive power.
- Black Start: Battery-backed solar can restart grid sections after blackout.
- Synthetic Inertia: Grid-forming inverters simulate rotational inertia.
"""
    }
]


def build_knowledge_base(persist_dir: str = "rag/faiss_index") -> FAISS:
    """
    Build FAISS vector store from domain knowledge documents.
    Returns the vectorstore object.
    """
    logger.info("Building RAG knowledge base...")

    # 1. Convert raw dicts to LangChain Document objects
    raw_docs = []
    for item in KNOWLEDGE_DOCUMENTS:
        doc = Document(
            page_content=item["content"].strip(),
            metadata={"title": item["title"], "source": "solar_grid_kb"}
        )
        raw_docs.append(doc)

    # 2. Split into chunks for better retrieval granularity
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info(f"Split into {len(chunks)} chunks from {len(raw_docs)} documents.")

    # 3. Embed using a small, fast HuggingFace model (free, runs locally)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 4. Build FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5. Persist to disk so we don't rebuild every run
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)
    logger.info(f"FAISS index saved to: {persist_dir}")

    return vectorstore


def load_knowledge_base(persist_dir: str = "rag/faiss_index") -> FAISS:
    """
    Load existing FAISS index from disk. Build if not found.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    index_file = os.path.join(persist_dir, "index.faiss")
    if os.path.exists(index_file):
        logger.info(f"Loading existing FAISS index from {persist_dir}")
        return FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        logger.info("No existing index found. Building fresh knowledge base...")
        return build_knowledge_base(persist_dir)


def get_retriever(persist_dir: str = "rag/faiss_index", k: int = 3):
    """
    Returns a LangChain retriever for use inside LangGraph nodes.
    k = number of chunks to retrieve per query.
    """
    vs = load_knowledge_base(persist_dir)
    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vs = build_knowledge_base()
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke("What should I do when solar generation drops suddenly?")
    print("\n=== TEST RETRIEVAL ===")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc.metadata['title']}")
        print(doc.page_content[:300], "...")
