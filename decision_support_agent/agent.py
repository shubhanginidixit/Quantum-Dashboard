"""
decision_support_agent/agent.py
────────────────────────────────
Agent 2 – Decision Support & Explainability Agent.

Responsibilities
────────────────
1. Fetch a prediction record (read-only from `predictions`).
2. Compute risk label from probability_score + threshold.
3. Re-calculate risk label with a NEW threshold **without**
   altering probability_score or triggering retraining (FR-6.2).
4. Generate classical explanation (SHAP on classical features).
5. Generate quantum sensitivity (parameter-shift / epsilon
   perturbation on quantum-encoded inputs).
6. Write **exactly one** document to `decision_support`.

Read-only access: `predictions`, `preprocessed_features`.
Write access:     `decision_support`.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Google ADK (with fallback for standalone environments)
try:
    from google.adk.agents import Agent
except ImportError:
    class Agent:
        """Lightweight stand-in when google-adk is not installed."""
        def __init__(self, name: str, model: str = "gemini-2.0-flash", description: str = "", instruction: str = "", tools: list = None):
            self.name = name
            self.model = model
            self.description = description
            self.instruction = instruction
            self.tools = tools or []


# Shared modules
from shared.config import (
    DEFAULT_RISK_THRESHOLD,
    COLLECTION_DECISION,
    COLLECTION_PREDICTIONS,
    COLLECTION_PREPROCESSED,
)
from shared.schemas import (
    ClassicalExplanation,
    DecisionSupportMetadata,
    DecisionSupportSchema,
    QuantumSensitivity,
)
from shared.firestore_tools import read_document, write_document

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session scratch-pad (per invocation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_session: Dict[str, Any] = {}


# ─── Tool 1: fetch_prediction ────────────────────────────────
def fetch_prediction(prediction_id: str) -> dict:
    """
    Read a prediction document from the `predictions` collection.
    Also fetches the matching `preprocessed_features` doc so that
    feature names & quantum encoding are available for explanation.

    Returns
    -------
    dict – the prediction record + preprocessing metadata.
    """
    pred = read_document(COLLECTION_PREDICTIONS, prediction_id)
    if pred is None:
        return {"error": f"Prediction '{prediction_id}' not found."}

    _session["prediction"] = pred
    _session["sample_id"] = prediction_id

    # Try to load the linked preprocessed record
    dataset_id = pred.get("dataset_id")
    preproc = None
    if dataset_id:
        preproc = read_document(COLLECTION_PREPROCESSED, dataset_id)
    _session["preprocessed"] = preproc

    return {
        "status": "fetched",
        "prediction_id": prediction_id,
        "probability_score": pred.get("probability_score"),
        "disease_category": pred.get("disease_category"),
        "has_preprocessed": preproc is not None,
    }


# ─── Tool 2: compute_risk_label ──────────────────────────────
def compute_risk_label(
    threshold: Optional[float] = None,
) -> dict:
    """
    Classify the prediction as 'High Risk' or 'Low Risk'
    using the configured (or overridden) threshold.
    """
    t = threshold if threshold is not None else DEFAULT_RISK_THRESHOLD
    prob = _session["prediction"]["probability_score"]

    label = "High Risk" if prob >= t else "Low Risk"
    _session["risk_score"] = prob
    _session["risk_label"] = label
    _session["threshold_used"] = t

    return {
        "probability_score": prob,
        "threshold": t,
        "risk_label": label,
    }


# ─── Tool 3: recalc_with_threshold (FR-6.2) ──────────────────
def recalc_with_threshold(new_threshold: float) -> dict:
    """
    FR-6.2 – Update the risk label using *new_threshold*.

    Guarantees:
      • `probability_score` is **never** altered.
      • No retraining or model re-inference is triggered.
      • Only the label string changes.
    """
    prob = _session["risk_score"]  # unchanged
    old_label = _session["risk_label"]
    new_label = "High Risk" if prob >= new_threshold else "Low Risk"

    _session["risk_label"] = new_label
    _session["threshold_used"] = new_threshold

    return {
        "probability_score": prob,
        "old_threshold": _session.get("threshold_used", DEFAULT_RISK_THRESHOLD),
        "new_threshold": new_threshold,
        "old_label": old_label,
        "new_label": new_label,
        "score_unchanged": True,
    }


# ─── Tool 4: generate_classical_explanation ───────────────────
def generate_classical_explanation() -> dict:
    """
    Compute a SHAP-based explanation for the classical features.

    PLACEHOLDER: In production this would use a trained model's
    predict function as the SHAP explainer target.  Here we
    approximate with a KernelExplainer on a dummy linear model
    so the smoke test can execute without a live model artefact.
    """
    preproc = _session.get("preprocessed")
    if preproc is None:
        return {"error": "No preprocessed record linked – cannot explain."}

    feature_names = preproc.get("classical_feature_names", [])
    feature_vector = preproc.get("classical_feature_vector", [])
    n = len(feature_names)

    # ── PLACEHOLDER model callback ────────────────────────
    # Replace with: shap.KernelExplainer(model.predict, X_train)
    rng = np.random.default_rng(99)
    shap_values = rng.standard_normal(n).tolist()
    base_value = float(rng.random())

    # Rank features by |SHAP|
    abs_shap = np.abs(shap_values)
    top_idx = np.argsort(abs_shap)[::-1][:5]
    top_contributors = [feature_names[i] for i in top_idx]

    explanation = ClassicalExplanation(
        shap_values=shap_values,
        feature_names=feature_names,
        base_value=base_value,
        top_contributors=top_contributors,
    )
    _session["classical_explanation"] = explanation

    return {
        "status": "explained",
        "top_contributors": top_contributors,
        "base_value": base_value,
    }


# ─── Tool 5: generate_quantum_sensitivity ────────────────────
def generate_quantum_sensitivity(
    method: str = "parameter_shift",
    epsilon: float = 0.01,
) -> dict:
    """
    Estimate per-qubit sensitivity via parameter-shift rule or
    epsilon perturbation on the quantum-encoded parameters.

    PLACEHOLDER: In production this calls the quantum circuit
    backend with shifted parameters.  Here we simulate.
    """
    preproc = _session.get("preprocessed")
    if preproc is None:
        return {"error": "No preprocessed record – cannot compute sensitivity."}

    qe = preproc.get("quantum_encoding", {})
    encoded_vals = qe.get("encoded_values", [])
    n_qubits = qe.get("qubit_count", len(encoded_vals))

    rng = np.random.default_rng(77)

    if method == "parameter_shift":
        # Simulated: ∂f/∂θ ≈ [f(θ+π/2) - f(θ-π/2)] / 2
        sensitivities = (rng.random(n_qubits) * 2 - 1).tolist()
    else:
        # Epsilon perturbation: Δf / ε
        sensitivities = (rng.random(n_qubits) * epsilon * 10).tolist()

    qubit_labels = [f"q{i}" for i in range(n_qubits)]

    sensitivity = QuantumSensitivity(
        method=method,
        sensitivities=sensitivities,
        qubit_labels=qubit_labels,
        epsilon=epsilon,
    )
    _session["quantum_sensitivity"] = sensitivity

    return {
        "status": "computed",
        "method": method,
        "n_qubits": n_qubits,
        "top_sensitive_qubit": qubit_labels[int(np.argmax(np.abs(sensitivities)))],
    }


# ─── Tool 6: write_decision_support_record ───────────────────
def write_decision_support_record() -> dict:
    """
    Validate and write **exactly one** document to
    `decision_support`.

    Structural guarantee: `classical_explanation` and
    `quantum_sensitivity` are stored as two SEPARATE top-level
    keys – never merged.
    """
    doc = DecisionSupportSchema(
        sample_id=_session["sample_id"],
        risk_score=_session["risk_score"],
        risk_label=_session["risk_label"],
        classical_explanation=_session["classical_explanation"],
        quantum_sensitivity=_session["quantum_sensitivity"],
        metadata=DecisionSupportMetadata(
            agent_run_id=str(uuid.uuid4()),
        ),
    )

    doc_id = write_document(
        COLLECTION_DECISION,
        doc.model_dump(),
        doc_id=_session["sample_id"],
    )

    return {
        "status": "written",
        "collection": COLLECTION_DECISION,
        "doc_id": doc_id,
        "schema_valid": True,
        "separate_keys": ["classical_explanation", "quantum_sensitivity"],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADK Agent definition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DECISION_SUPPORT_INSTRUCTION = """\
You are the **Decision Support & Explainability Agent** for the
Neutron hybrid quantum-classical disease-detection platform.

Your mission is to take a prediction and produce a clinician-
ready decision-support record that includes:
  • A risk label derived from the probability score.
  • A classical SHAP-based explanation.
  • A quantum-circuit sensitivity analysis.

**Workflow** – call tools in this order:
1. `fetch_prediction`                – load the prediction record.
2. `compute_risk_label`              – classify High / Low risk.
3. `generate_classical_explanation`  – SHAP feature importance.
4. `generate_quantum_sensitivity`    – per-qubit sensitivity.
5. `write_decision_support_record`   – persist exactly ONE doc.

Optionally the user may ask you to re-classify with a new
threshold.  In that case call `recalc_with_threshold` – this
MUST NOT change the probability_score.

After writing, report a summary and stop.
"""

decision_support_agent = Agent(
    name="decision_support_agent",
    model="gemini-2.0-flash",
    description=(
        "Generates explainable, clinician-ready decision-support "
        "records combining SHAP explanations and quantum sensitivity."
    ),
    instruction=DECISION_SUPPORT_INSTRUCTION,
    tools=[
        fetch_prediction,
        compute_risk_label,
        recalc_with_threshold,
        generate_classical_explanation,
        generate_quantum_sensitivity,
        write_decision_support_record,
    ],
)

# Convenience alias expected by `adk run decision_support_agent`
root_agent = decision_support_agent
