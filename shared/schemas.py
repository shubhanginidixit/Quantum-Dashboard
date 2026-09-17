"""
shared/schemas.py
─────────────────
Pydantic v2 models that enforce strict schemas for every
document written to Firestore.

KEY DESIGN RULE
───────────────
`DecisionSupportSchema.classical_explanation` and
`DecisionSupportSchema.quantum_sensitivity` are **two separate
sub-models** (ClassicalExplanation and QuantumSensitivity).
They MUST never be merged into a single dict.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Preprocessing Agent output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class QuantumEncoding(BaseModel):
    """Describes how classical features are encoded for the quantum circuit."""
    type: str = Field(
        ...,
        description="Encoding strategy: 'angle', 'amplitude', or 'zzfeaturemap'",
    )
    qubit_count: int = Field(..., ge=1, description="Number of qubits used")
    encoded_values: List[float] = Field(
        ..., description="Encoded parameter values fed to the circuit"
    )


class PreprocessedFeaturesSchema(BaseModel):
    """
    Single document written to the `preprocessed_features` collection.
    Captures everything downstream agents need about the cleaned,
    reduced, and quantum-encoded feature set.
    """
    dataset_id: str
    disease_category: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Classical side
    classical_feature_vector: List[float]
    classical_feature_names: List[str]

    # Quantum side
    quantum_encoding: QuantumEncoding

    # Reduction metadata
    reduction_method: str = Field(
        ..., description="'pca' or 'autoencoder'"
    )
    explained_variance_or_loss: float = Field(
        ...,
        description="Cumulative explained variance (PCA) or reconstruction loss (AE)",
    )

    @field_validator("classical_feature_vector", "classical_feature_names")
    @classmethod
    def non_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("Must contain at least one element")
        return v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Decision Support Agent output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ClassicalExplanation(BaseModel):
    """SHAP-based feature-importance explanation (classical features only)."""
    shap_values: List[float] = Field(
        ..., description="SHAP values aligned to classical_feature_names"
    )
    feature_names: List[str]
    base_value: float = Field(
        ..., description="SHAP expected/base value for the model output"
    )
    top_contributors: List[str] = Field(
        default_factory=list,
        description="Top-N feature names driving the prediction",
    )


class QuantumSensitivity(BaseModel):
    """
    Parameter-shift / epsilon-perturbation sensitivity analysis
    on the quantum-encoded inputs.
    """
    method: str = Field(
        ..., description="'parameter_shift' or 'epsilon_perturbation'"
    )
    sensitivities: List[float] = Field(
        ..., description="Per-qubit sensitivity magnitudes"
    )
    qubit_labels: List[str] = Field(
        ..., description="Labels like ['q0', 'q1', …]"
    )
    epsilon: float = Field(
        default=0.01, description="Perturbation magnitude used"
    )


class DecisionSupportMetadata(BaseModel):
    """Audit / traceability metadata."""
    model_version: str = "neutron-v0.1"
    agent_run_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DecisionSupportSchema(BaseModel):
    """
    Document written to the `decision_support` collection.

    ┌─────────────────────────────────────────────────────┐
    │  STRUCTURAL RULE                                    │
    │  `classical_explanation` and `quantum_sensitivity`  │
    │  are ALWAYS two **separate** keys.  They must       │
    │  never be flattened or merged.                      │
    └─────────────────────────────────────────────────────┘
    """
    sample_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_label: str = Field(
        ..., description="'High Risk' or 'Low Risk'"
    )

    # ── Structurally separate explanation keys ────────────
    classical_explanation: ClassicalExplanation
    quantum_sensitivity: QuantumSensitivity

    # ── Audit trail ──────────────────────────────────────
    metadata: DecisionSupportMetadata = Field(
        default_factory=DecisionSupportMetadata
    )
