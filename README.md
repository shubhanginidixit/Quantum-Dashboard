# Neutron – Hybrid Quantum-Classical Disease Detection Platform

> **SIH Problem Statement SIH26139**
> Google ADK Micro-Agents + Streamlit Dashboard

---

## 🚀 Live Dashboard

**[▶ Open Neutron Dashboard (Streamlit Cloud)](https://shubhanginidixit-quantum-dashboard-dashboard-xxxxxx.streamlit.app/)**

> ⚠️ Replace `xxxxxx` with the actual Streamlit Cloud app hash after deployment.
> See [Deploy to Streamlit Cloud](#deploy-to-streamlit-cloud) below.

---

## 📁 Project Structure

```
Quantum/
├── shared/                          # Shared utilities
│   ├── __init__.py
│   ├── config.py                    # Qubit budget, thresholds, collection names
│   ├── schemas.py                   # Pydantic v2 models (strict separation)
│   └── firestore_tools.py           # Firestore R/W with in-memory mock fallback
│
├── preprocessing_agent/             # Agent 1 – Preprocessing
│   ├── __init__.py
│   └── agent.py                     # 7 tools: load → clean → scale → reduce → select → encode → write
│
├── decision_support_agent/          # Agent 2 – Decision Support & Explainability
│   ├── __init__.py
│   └── agent.py                     # 6 tools: fetch → risk label → SHAP → quantum sensitivity → write
│
├── dashboard.py                     # Streamlit interactive dashboard (deployable)
├── smoke_test.py                    # End-to-end test suite (mock Firestore)
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone
git clone https://github.com/shubhanginidixit/Quantum-Dashboard.git
cd Quantum-Dashboard

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY for ADK agents (optional for smoke test)
```

---

## 🧪 Run Smoke Test

The smoke test uses **in-memory mock Firestore** – no GCP credentials needed:

```bash
python smoke_test.py
```

Expected output:
```
═══ (a) Preprocessing Agent ═══
  ✅  load_dataset returns 'loaded'
  ✅  clean_data removes all NaNs
  ...
═══ (b) Decision Support Agent ═══
  ✅  fetch_prediction found the record
  ✅  has top-level 'classical_explanation' key
  ✅  has top-level 'quantum_sensitivity' key
  ...
═══ (c) FR-6.2: recalc_with_threshold ═══
  ✅  probability_score unchanged at 0.72
  ...
  🎉  All tests passed!
```

---

## 🤖 Run ADK Agents

```bash
# Preprocessing Agent
adk run preprocessing_agent

# Decision Support Agent
adk run decision_support_agent
```

> Requires `GOOGLE_API_KEY` set in `.env` for Gemini model access.

---

## 📊 Run Dashboard Locally

```bash
streamlit run dashboard.py
```

---

## ☁️ Deploy to Streamlit Cloud

To get a **direct link** without running code locally:

1. Push this repo to GitHub (already done).
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New app"** → select repo `shubhanginidixit/Quantum-Dashboard`.
4. Set **Main file path** to `dashboard.py`.
5. Click **Deploy** → you get a permanent public URL.

---

## 🗺️ Functional Requirement Mappings

| FR     | Description                              | Implementation                                             |
|--------|------------------------------------------|------------------------------------------------------------|
| FR-1   | Data ingestion & preprocessing           | `preprocessing_agent` tools 1-3                            |
| FR-2   | Dimensionality reduction for qubits      | `reduce_dimensionality` (PCA/AE, 8-20 qubits)             |
| FR-3   | Quantum feature encoding                 | `encode_quantum_features` (Angle/Amplitude/ZZFeatureMap)   |
| FR-4   | Classical explainability                 | `generate_classical_explanation` (SHAP)                    |
| FR-5   | Quantum sensitivity analysis             | `generate_quantum_sensitivity` (parameter shift/ε-perturb) |
| FR-6.1 | Risk classification                      | `compute_risk_label`                                       |
| FR-6.2 | Threshold re-calc (no score mutation)    | `recalc_with_threshold` – score stays frozen               |
| FR-7   | Structural explanation separation        | `classical_explanation` ≠ `quantum_sensitivity` (always)   |

---

## 🏗️ Structural Separation Rules

```
DecisionSupportSchema
├── classical_explanation: ClassicalExplanation   ← SHAP values, feature names
│                                                    (SEPARATE sub-model)
└── quantum_sensitivity:  QuantumSensitivity      ← per-qubit sensitivities
                                                    (SEPARATE sub-model)
```

These are **always two distinct Pydantic sub-models** stored as **two separate
top-level keys** in the Firestore document. They must **never** be merged or
flattened into a single dictionary.

---

## 📝 Placeholder Mocks

| Component                  | Status        | Notes                                                    |
|----------------------------|---------------|----------------------------------------------------------|
| DICOM/FASTA data loader    | 🔲 Stubbed    | `load_dataset` generates synthetic tabular data           |
| Autoencoder reduction      | 🔲 Stubbed    | Falls back to PCA + synthetic reconstruction loss         |
| SHAP explainer             | 🔲 Stubbed    | Random SHAP values (needs trained model `predict` fn)     |
| Quantum circuit backend    | 🔲 Stubbed    | Simulated parameter-shift sensitivities                   |
| Firestore (no GCP creds)   | ✅ Mock        | In-memory dict store – fully functional for testing       |

---

## 📄 License

MIT – Built for Smart India Hackathon 2026.
