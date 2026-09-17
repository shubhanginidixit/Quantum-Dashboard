"""
smoke_test.py
─────────────
End-to-end smoke test for the Neutron platform.

Runs entirely in-memory (mock Firestore) – no GCP credentials needed.

Test plan
─────────
a) Run Preprocessing Agent tools on synthetic data → confirm
   output in `preprocessed_features`.
b) Seed mock `predictions`, run Decision Support Agent tools →
   confirm separate keys for classical_explanation and
   quantum_sensitivity.
c) Execute `recalc_with_threshold` → assert risk_label changes
   while probability_score stays the same.
"""
from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path so `shared.*` imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force mock Firestore (no GCP creds)
os.environ.pop("GCP_PROJECT_ID", None)

from shared.config import (
    COLLECTION_PREDICTIONS,
    COLLECTION_PREPROCESSED,
    COLLECTION_DECISION,
)
from shared.firestore_tools import (
    read_document,
    seed_document,
    list_documents,
    clear_mock_collection,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_passed = 0
_failed = 0


def _assert(condition: bool, msg: str):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {msg}")
    else:
        _failed += 1
        print(f"  [FAIL] {msg}")



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (a) Preprocessing Agent – tool-level integration test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_preprocessing_agent():
    print("\n=== (a) Preprocessing Agent ===")
    clear_mock_collection(COLLECTION_PREPROCESSED)

    from preprocessing_agent.agent import (
        load_dataset,
        clean_data,
        normalize_scale,
        reduce_dimensionality,
        select_features,
        encode_quantum_features,
        write_preprocessed_output,
        _session,
    )

    # Step 1 – Load synthetic dataset
    res = load_dataset(n_samples=100, n_features=25, disease_category="cardiology")
    _assert(res["status"] == "loaded", "load_dataset returns 'loaded'")
    _assert(res["shape"][1] == 26, "synthetic df has 25 features + 1 target")

    # Step 2 – Clean
    res = clean_data(strategy="median")
    _assert(res["missing_after"] == 0, "clean_data removes all NaNs")
    _assert("median" in res["reasoning"], "imputation reasoning mentions strategy")

    # Step 3 – Scale
    res = normalize_scale(method="minmax")
    _assert(res["method"] == "minmax", "normalize_scale uses minmax")

    # Step 4 – Reduce dimensionality
    res = reduce_dimensionality(method="pca", target_dims=10)
    _assert(8 <= res["target_dims"] <= 20, "qubit budget respected (8-20)")
    _assert(res["method"] == "pca", "reduction method is PCA")

    # Step 5 – Select features
    res = select_features()
    _assert(res["status"] == "selected", "select_features ran successfully")

    # Step 6 – Quantum encoding
    res = encode_quantum_features()
    _assert(res["status"] == "encoded", "encode_quantum_features ran")
    _assert(res["encoding_type"] in ("angle", "amplitude", "zzfeaturemap"),
            f"encoding type is valid: {res['encoding_type']}")

    # Step 7 – Write to Firestore
    res = write_preprocessed_output()
    _assert(res["status"] == "written", "write_preprocessed_output succeeded")
    _assert(res["schema_valid"], "output passed Pydantic validation")

    # Verify document exists in mock store
    doc_id = _session["dataset_id"]
    doc = read_document(COLLECTION_PREPROCESSED, doc_id)
    _assert(doc is not None, f"document {doc_id[:8]}... exists in preprocessed_features")
    _assert("quantum_encoding" in doc, "document contains quantum_encoding")
    _assert("classical_feature_names" in doc, "document contains classical_feature_names")

    return doc_id


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (b) Decision Support Agent – tool-level integration test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_decision_support_agent(dataset_id: str):
    print("\n=== (b) Decision Support Agent ===")
    clear_mock_collection(COLLECTION_DECISION)

    # Seed a mock prediction record
    pred_id = "pred-smoke-001"
    seed_document(COLLECTION_PREDICTIONS, pred_id, {
        "dataset_id": dataset_id,
        "disease_category": "cardiology",
        "probability_score": 0.72,
        "model_version": "mock-v1",
    })

    from decision_support_agent.agent import (
        fetch_prediction,
        compute_risk_label,
        generate_classical_explanation,
        generate_quantum_sensitivity,
        write_decision_support_record,
        _session,
    )

    # Step 1 – Fetch prediction
    res = fetch_prediction(pred_id)
    _assert(res["status"] == "fetched", "fetch_prediction found the record")
    _assert(res["probability_score"] == 0.72, "probability_score matches seed")
    _assert(res["has_preprocessed"], "linked preprocessed doc found")

    # Step 2 – Compute risk label
    res = compute_risk_label(threshold=0.5)
    _assert(res["risk_label"] == "High Risk", "0.72 >= 0.5 -> High Risk")

    # Step 3 – Classical explanation
    res = generate_classical_explanation()
    _assert(res["status"] == "explained", "SHAP explanation generated")
    _assert(len(res["top_contributors"]) > 0, "at least one top contributor")

    # Step 4 – Quantum sensitivity
    res = generate_quantum_sensitivity(method="parameter_shift", epsilon=0.01)
    _assert(res["status"] == "computed", "quantum sensitivity computed")
    _assert(res["n_qubits"] > 0, "qubit count > 0")

    # Step 5 – Write decision support record
    res = write_decision_support_record()
    _assert(res["status"] == "written", "write_decision_support_record succeeded")
    _assert(res["schema_valid"], "output passed Pydantic validation")

    # ── KEY ASSERTION: separate keys ─────────────────────────
    doc = read_document(COLLECTION_DECISION, pred_id)
    _assert(doc is not None, "decision_support doc exists")
    _assert("classical_explanation" in doc, "has top-level 'classical_explanation' key")
    _assert("quantum_sensitivity" in doc, "has top-level 'quantum_sensitivity' key")
    _assert(
        isinstance(doc["classical_explanation"], dict)
        and isinstance(doc["quantum_sensitivity"], dict),
        "both keys are dicts (not merged)",
    )
    # Ensure they are truly separate structures
    ce_keys = set(doc["classical_explanation"].keys())
    qs_keys = set(doc["quantum_sensitivity"].keys())
    _assert(
        "shap_values" in ce_keys and "sensitivities" in qs_keys,
        "classical has shap_values; quantum has sensitivities (separate schemas)",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (c) FR-6.2 – recalc_with_threshold
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_recalc_threshold():
    print("\n=== (c) FR-6.2: recalc_with_threshold ===")

    from decision_support_agent.agent import (
        recalc_with_threshold,
        _session,
    )

    original_score = _session["risk_score"]  # should still be 0.72

    # With threshold 0.5 → High Risk (already set).
    # Now raise threshold to 0.80 → should flip to Low Risk.
    res = recalc_with_threshold(new_threshold=0.80)

    _assert(res["new_label"] == "Low Risk", "0.72 < 0.80 -> flipped to Low Risk")
    _assert(
        res["probability_score"] == original_score,
        f"probability_score unchanged at {original_score}",
    )
    _assert(res["score_unchanged"] is True, "score_unchanged flag is True")

    # Flip back with a very low threshold
    res2 = recalc_with_threshold(new_threshold=0.10)
    _assert(res2["new_label"] == "High Risk", "0.72 >= 0.10 -> back to High Risk")
    _assert(
        res2["probability_score"] == original_score,
        "probability_score STILL unchanged after second recalc",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    print("=" * 52)
    print("   Neutron Platform - Smoke Test Suite")
    print("   Mock Firestore (no GCP creds required)")
    print("=" * 52)

    dataset_id = test_preprocessing_agent()
    test_decision_support_agent(dataset_id)
    test_recalc_threshold()

    print(f"\n{'-'*50}")
    print(f"  Results: {_passed} passed, {_failed} failed")
    if _failed:
        print("  [WARN] Some tests failed - see above.")
        sys.exit(1)
    else:
        print("  [SUCCESS] All tests passed!")
        sys.exit(0)

