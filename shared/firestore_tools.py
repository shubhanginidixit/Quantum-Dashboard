"""
shared/firestore_tools.py
──────────────────────────
Scoped Firestore read / write helpers.

When GCP_PROJECT_ID is unset the module transparently falls back
to an **in-memory mock store** so the smoke-test suite can run
without any cloud credentials.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from shared.config import (
    COLLECTION_DECISION,
    COLLECTION_PREDICTIONS,
    COLLECTION_PREPROCESSED,
    GCP_PROJECT_ID,
    USE_MOCK_FIRESTORE,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# In-memory mock Firestore (used when GCP creds are absent)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MOCK_DB: Dict[str, Dict[str, Any]] = {
    COLLECTION_PREPROCESSED: {},
    COLLECTION_PREDICTIONS: {},
    COLLECTION_DECISION: {},
}


def _serialise(obj: Any) -> Any:
    """Make datetimes JSON-safe for the mock store."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(i) for i in obj]
    return obj


# ── Live Firestore client (lazy) ─────────────────────────────
_fs_client = None


def _get_client():
    global _fs_client
    if _fs_client is None:
        from google.cloud import firestore  # type: ignore
        _fs_client = firestore.Client(project=GCP_PROJECT_ID)
    return _fs_client


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def write_document(
    collection: str,
    data: Dict[str, Any],
    doc_id: Optional[str] = None,
) -> str:
    """
    Write a single document.  Returns the document ID.

    Allowed collections:
      • preprocessed_features  (Preprocessing Agent)
      • decision_support       (Decision Support Agent)
    """
    doc_id = doc_id or str(uuid.uuid4())
    safe = _serialise(data)

    if USE_MOCK_FIRESTORE:
        _MOCK_DB.setdefault(collection, {})[doc_id] = safe
    else:
        _get_client().collection(collection).document(doc_id).set(safe)

    return doc_id


def read_document(collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Read a single document by ID.

    Allowed read-only collections for Decision Support Agent:
      • predictions
      • preprocessed_features
    """
    if USE_MOCK_FIRESTORE:
        return _MOCK_DB.get(collection, {}).get(doc_id)
    else:
        snap = _get_client().collection(collection).document(doc_id).get()
        return snap.to_dict() if snap.exists else None


def list_documents(
    collection: str,
    limit: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """Return up to *limit* documents from *collection*."""
    if USE_MOCK_FIRESTORE:
        items = dict(list(_MOCK_DB.get(collection, {}).items())[:limit])
        return items
    else:
        docs = _get_client().collection(collection).limit(limit).stream()
        return {d.id: d.to_dict() for d in docs}


def seed_document(collection: str, doc_id: str, data: Dict[str, Any]) -> str:
    """
    Seed helper – identical to write_document but semantically
    indicates the data is synthetic / test-only.
    """
    return write_document(collection, data, doc_id=doc_id)


def clear_mock_collection(collection: str) -> None:
    """Reset a mock collection (test helper)."""
    if USE_MOCK_FIRESTORE:
        _MOCK_DB[collection] = {}
