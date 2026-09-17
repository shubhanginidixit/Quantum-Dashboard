"""
shared/config.py
────────────────
Central configuration for the Neutron platform.
All magic numbers and collection names live here so agents
never hard-code them.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()  # picks up .env in project root

# ── Qubit budget ──────────────────────────────────────────────
QUBIT_MIN = 8
QUBIT_MAX = 20

# ── Quantum encoding selection thresholds ─────────────────────
# When reduced feature count is below this → Angle encoding,
# between this and QUBIT_MAX → Amplitude, else → ZZFeatureMap.
ANGLE_ENCODING_UPPER = 10
AMPLITUDE_ENCODING_UPPER = 16

# ── Risk classification ──────────────────────────────────────
DEFAULT_RISK_THRESHOLD: float = float(os.getenv("RISK_THRESHOLD", "0.50"))

# ── Firestore collection names ────────────────────────────────
COLLECTION_PREPROCESSED = "preprocessed_features"
COLLECTION_PREDICTIONS  = "predictions"
COLLECTION_DECISION     = "decision_support"

# ── GCP / Firestore ──────────────────────────────────────────
GCP_PROJECT_ID: str | None = os.getenv("GCP_PROJECT_ID") or None
USE_MOCK_FIRESTORE: bool = GCP_PROJECT_ID is None

# ── ADK / Gemini model ───────────────────────────────────────
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY") or None
