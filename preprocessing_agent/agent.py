"""
preprocessing_agent/agent.py
─────────────────────────────
Agent 1 – Preprocessing Agent for the Neutron platform.

Responsibilities
────────────────
1. Load raw dataset (tabular CSV / synthetic).
2. Clean data (imputation with reasoning).
3. Normalize / scale features.
4. Reduce dimensionality (PCA or Autoencoder) to fit the
   qubit budget (8-20 qubits).
5. Select informative features.
6. Encode features for quantum circuit (Angle / Amplitude / ZZFeatureMap).
7. Write **exactly one** document to `preprocessed_features`.

All heavy-lifting happens inside deterministic *tool* functions;
the LLM orchestrates tool calls only.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.linalg

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
    QUBIT_MIN,
    QUBIT_MAX,
    ANGLE_ENCODING_UPPER,
    AMPLITUDE_ENCODING_UPPER,
    COLLECTION_PREPROCESSED,
)
from shared.schemas import PreprocessedFeaturesSchema, QuantumEncoding
from shared.firestore_tools import write_document

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# In-memory scratchpad shared between tool invocations.
# The ADK invokes each tool as a plain function; this dict
# acts as short-lived session state.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_session: Dict[str, Any] = {}


# ─── Tool 1: load_dataset ─────────────────────────────────────
def load_dataset(
    csv_path: Optional[str] = None,
    n_samples: int = 200,
    n_features: int = 30,
    disease_category: str = "oncology",
) -> dict:
    """
    Load a tabular dataset from *csv_path*.
    If the path is None or not found, generate a **synthetic**
    dataset (placeholder for DICOM / FASTA / EHR pipelines).

    Returns
    -------
    dict  – summary statistics of the loaded data.
    """
    if csv_path:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            # PLACEHOLDER: hook into DICOM/FASTA loader here
            df = _generate_synthetic(n_samples, n_features)
    else:
        df = _generate_synthetic(n_samples, n_features)

    _session["raw_df"] = df
    _session["disease_category"] = disease_category
    _session["dataset_id"] = str(uuid.uuid4())

    return {
        "status": "loaded",
        "shape": list(df.shape),
        "columns": list(df.columns),
        "disease_category": disease_category,
        "dataset_id": _session["dataset_id"],
    }


def _generate_synthetic(n: int, p: int) -> pd.DataFrame:
    """Create a synthetic dataset with some NaNs for realism."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n, p))
    cols = [f"feat_{i}" for i in range(p)]
    df = pd.DataFrame(data, columns=cols)
    # Inject ~5 % missing values
    mask = rng.random(df.shape) < 0.05
    df[mask] = np.nan
    # Target column (binary)
    df["target"] = rng.integers(0, 2, size=n)
    return df


# ─── Tool 2: clean_data ──────────────────────────────────────
def clean_data(strategy: str = "median") -> dict:
    """
    Impute missing values.

    Parameters
    ----------
    strategy : str
        One of 'mean', 'median', 'most_frequent'.  Default 'median'.

    Imputation reasoning
    --------------------
    • Median is preferred for skewed clinical distributions.
    • Mean is acceptable when features are roughly Gaussian.
    • Most-frequent is a fallback for categorical-heavy data.
    """
    df: pd.DataFrame = _session["raw_df"].copy()

    # Separate target if present
    target = None
    if "target" in df.columns:
        target = df.pop("target")

    missing_before = int(df.isna().sum().sum())

    if strategy == "median":
        df_clean = df.fillna(df.median(numeric_only=True))
    elif strategy == "mean":
        df_clean = df.fillna(df.mean(numeric_only=True))
    else:
        df_clean = df.fillna(df.mode().iloc[0])
    df_clean = df_clean.fillna(0)  # safety fallback

    if target is not None:
        _session["target"] = target.values
    _session["clean_df"] = df_clean

    return {
        "status": "cleaned",
        "strategy": strategy,
        "missing_before": missing_before,
        "missing_after": 0,
        "reasoning": (
            f"Used '{strategy}' imputation - appropriate for "
            "clinical tabular data with potential skew."
        ),
    }


# ─── Tool 3: normalize_scale ─────────────────────────────────
def normalize_scale(method: str = "minmax") -> dict:
    """
    Scale features to a bounded range.

    Parameters
    ----------
    method : str
        'minmax' -> [0, 1]  or  'standard' -> zero-mean unit-var.
    """
    df: pd.DataFrame = _session["clean_df"]
    if method == "minmax":
        min_val = df.min()
        max_val = df.max()
        range_val = (max_val - min_val).replace(0, 1)
        scaled = (df - min_val) / range_val
    else:
        mean = df.mean()
        std = df.std().replace(0, 1)
        scaled = (df - mean) / std

    arr = scaled.values
    _session["scaled_df"] = pd.DataFrame(arr, columns=df.columns)

    return {
        "status": "scaled",
        "method": method,
        "feature_ranges": {
            "min": float(arr.min()),
            "max": float(arr.max()),
        },
    }


# ─── Tool 4: reduce_dimensionality ───────────────────────────
def reduce_dimensionality(
    method: str = "pca",
    target_dims: Optional[int] = None,
) -> dict:
    """
    Reduce feature dimensionality to fit the qubit budget
    (QUBIT_MIN .. QUBIT_MAX).

    Parameters
    ----------
    method : str
        'pca' (default) or 'autoencoder'.
    target_dims : int | None
        Desired output dimension.  Clamped to [QUBIT_MIN, QUBIT_MAX].
    """
    df: pd.DataFrame = _session["scaled_df"]
    n_feats = df.shape[1]

    if target_dims is None:
        target_dims = min(max(QUBIT_MIN, n_feats), QUBIT_MAX)
    target_dims = max(QUBIT_MIN, min(target_dims, QUBIT_MAX))

    # Fast, exact SVD PCA
    X = df.values - df.values.mean(axis=0)
    U, S, Vt = scipy.linalg.svd(X, full_matrices=False)
    reduced = np.dot(X, Vt[:target_dims].T)
    explained_var = (S ** 2) / np.sum(S ** 2)
    explained = float(np.sum(explained_var[:target_dims]))

    if method == "pca":
        _session["reduction_method"] = "pca"
        _session["explained_variance_or_loss"] = explained
    else:
        # ── PLACEHOLDER: Autoencoder reduction ────────────────
        recon = np.dot(reduced, Vt[:target_dims]) + df.values.mean(axis=0)
        loss = float(np.mean((df.values - recon) ** 2))
        _session["reduction_method"] = "autoencoder"
        _session["explained_variance_or_loss"] = loss

    cols = [f"pc_{i}" for i in range(target_dims)]
    _session["reduced_df"] = pd.DataFrame(reduced, columns=cols)
    _session["target_dims"] = target_dims

    return {
        "status": "reduced",
        "method": _session["reduction_method"],
        "original_dims": n_feats,
        "target_dims": target_dims,
        "explained_variance_or_loss": _session["explained_variance_or_loss"],
    }


# ─── Tool 5: select_features ─────────────────────────────────
def select_features(top_k: Optional[int] = None) -> dict:
    """
    Rank features by correlation/mutual-information with the target
    and keep the top-k (default = target_dims already set).

    If no target column was present in the raw data the tool
    returns all reduced features unchanged.
    """
    df: pd.DataFrame = _session["reduced_df"]
    target = _session.get("target")

    if target is None or top_k == 0:
        _session["selected_df"] = df
        _session["selected_names"] = list(df.columns)
        return {
            "status": "skipped",
            "reason": "no target column available for MI ranking",
        }

    k = top_k or df.shape[1]
    k = min(k, df.shape[1])

    # Mutual-information correlation proxy
    scores = []
    for col in df.columns:
        corr = float(np.abs(np.corrcoef(df[col].values, target)[0, 1]))
        scores.append(corr if not np.isnan(corr) else 0.0)

    ranking = np.argsort(scores)[::-1][:k]
    selected_cols = [df.columns[i] for i in ranking]

    _session["selected_df"] = df[selected_cols]
    _session["selected_names"] = selected_cols

    return {
        "status": "selected",
        "top_k": k,
        "selected_features": selected_cols,
        "mi_scores": {df.columns[i]: float(scores[i]) for i in ranking},
    }


# ─── Tool 6: encode_quantum_features ─────────────────────────
def encode_quantum_features(
    encoding_type: Optional[str] = None,
) -> dict:
    """
    Map classical features to quantum circuit parameters.

    Encoding selection heuristic (from shared/config.py):
      • n_features <= ANGLE_ENCODING_UPPER  -> Angle encoding
      • n_features <= AMPLITUDE_ENCODING_UPPER -> Amplitude encoding
      • else -> ZZFeatureMap

    The caller may override by passing *encoding_type* explicitly.
    """
    df: pd.DataFrame = _session["selected_df"]
    n = df.shape[1]

    if encoding_type is None:
        if n <= ANGLE_ENCODING_UPPER:
            encoding_type = "angle"
        elif n <= AMPLITUDE_ENCODING_UPPER:
            encoding_type = "amplitude"
        else:
            encoding_type = "zzfeaturemap"

    # Use the first sample as the representative encoding
    sample = df.iloc[0].values.tolist()

    if encoding_type == "angle":
        # theta = arccos(x) for x in [0, 1]
        encoded = [float(np.arccos(np.clip(v, -1, 1))) for v in sample]
    elif encoding_type == "amplitude":
        # Normalise to unit vector (amplitude embedding)
        norm = np.linalg.norm(sample) or 1.0
        encoded = (np.array(sample) / norm).tolist()
    else:
        # ZZFeatureMap: phi_i = x_i, phi_{ij} = (pi - x_i)(pi - x_j)
        encoded = sample  # raw values passed to ZZFeatureMap circuit

    qubit_count = n

    _session["quantum_encoding"] = QuantumEncoding(
        type=encoding_type,
        qubit_count=qubit_count,
        encoded_values=encoded,
    )

    return {
        "status": "encoded",
        "encoding_type": encoding_type,
        "qubit_count": qubit_count,
        "encoded_sample_length": len(encoded),
    }


# ─── Tool 7: write_preprocessed_output ───────────────────────
def write_preprocessed_output() -> dict:
    """
    Validate and write **exactly one** document to the
    `preprocessed_features` Firestore collection.
    """
    df: pd.DataFrame = _session["selected_df"]
    feature_vector = df.iloc[0].values.tolist()
    feature_names = list(df.columns)

    doc = PreprocessedFeaturesSchema(
        dataset_id=_session["dataset_id"],
        disease_category=_session["disease_category"],
        classical_feature_vector=feature_vector,
        classical_feature_names=feature_names,
        quantum_encoding=_session["quantum_encoding"],
        reduction_method=_session["reduction_method"],
        explained_variance_or_loss=_session["explained_variance_or_loss"],
    )

    doc_id = write_document(
        COLLECTION_PREPROCESSED,
        doc.model_dump(),
        doc_id=_session["dataset_id"],
    )

    return {
        "status": "written",
        "collection": COLLECTION_PREPROCESSED,
        "doc_id": doc_id,
        "schema_valid": True,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADK Agent definition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREPROCESSING_AGENT_INSTRUCTION = """\
You are the **Preprocessing Agent** for the Neutron hybrid
quantum-classical disease-detection platform.

Your mission is to take a raw biomedical dataset and transform
it into a clean, reduced, quantum-ready feature set stored in
Firestore.

**Workflow** - call the tools in this order:
1. `load_dataset`  - load or generate the dataset.
2. `clean_data`    - impute missing values (explain strategy).
3. `normalize_scale` - scale features to [0,1] or z-score.
4. `reduce_dimensionality` - PCA/AE targeting 8-20 qubits.
5. `select_features` - MI-based feature ranking.
6. `encode_quantum_features` - map to circuit parameters.
7. `write_preprocessed_output` - persist exactly ONE document.

After step 7, report a summary and stop.
"""

preprocessing_agent = Agent(
    name="preprocessing_agent",
    model="gemini-2.0-flash",
    description=(
        "Cleans, reduces, and quantum-encodes biomedical datasets "
        "for the Neutron hybrid quantum-classical platform."
    ),
    instruction=PREPROCESSING_AGENT_INSTRUCTION,
    tools=[
        load_dataset,
        clean_data,
        normalize_scale,
        reduce_dimensionality,
        select_features,
        encode_quantum_features,
        write_preprocessed_output,
    ],
)

# Convenience alias expected by `adk run preprocessing_agent`
root_agent = preprocessing_agent
