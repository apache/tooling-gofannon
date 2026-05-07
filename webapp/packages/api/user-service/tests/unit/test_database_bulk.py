"""Unit tests for the bulk DB APIs (save_many / delete_many / get_many).

Tests the memory backend behavior; the CouchDB backend is exercised by
integration tests against a real CouchDB instance (out of scope here).
"""
from __future__ import annotations

import pytest

from services.database_service.memory import MemoryDBService

pytestmark = pytest.mark.unit


@pytest.fixture
def db() -> MemoryDBService:
    return MemoryDBService()


# --- save_many ----------------------------------------------------------

def test_save_many_inserts_all_new_docs(db: MemoryDBService) -> None:
    docs = [{"_id": "a", "value": 1}, {"_id": "b", "value": 2}]
    results = db.save_many("test", docs)
    assert all(r["ok"] for r in results)
    assert len(results) == 2
    assert {r["id"] for r in results} == {"a", "b"}
    # And they're actually stored.
    assert db.get("test", "a")["value"] == 1


def test_save_many_empty_input_returns_empty_list(db: MemoryDBService) -> None:
    assert db.save_many("test", []) == []


def test_save_many_doc_without_id_marked_failed(db: MemoryDBService) -> None:
    docs = [{"_id": "a", "value": 1}, {"value": 2}]  # second has no _id
    results = db.save_many("test", docs)
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False
    assert "missing _id" in results[1]["error"]


def test_save_many_overwrites_existing_doc(db: MemoryDBService) -> None:
    db.save("test", "a", {"_id": "a", "value": "old"})
    db.save_many("test", [{"_id": "a", "value": "new"}])
    assert db.get("test", "a")["value"] == "new"


# --- get_many -----------------------------------------------------------

def test_get_many_returns_all_existing_docs(db: MemoryDBService) -> None:
    db.save("test", "a", {"_id": "a", "value": 1})
    db.save("test", "b", {"_id": "b", "value": 2})
    out = db.get_many("test", ["a", "b"])
    assert out["a"]["value"] == 1
    assert out["b"]["value"] == 2


def test_get_many_missing_doc_maps_to_none(db: MemoryDBService) -> None:
    db.save("test", "a", {"_id": "a", "value": 1})
    out = db.get_many("test", ["a", "missing"])
    assert out["a"]["value"] == 1
    assert out["missing"] is None


def test_get_many_empty_input(db: MemoryDBService) -> None:
    assert db.get_many("test", []) == {}


def test_get_many_preserves_order_of_keys(db: MemoryDBService) -> None:
    """Result dict iterates in input order — predictable for callers
    that zip results with their key list."""
    for k in ["a", "b", "c"]:
        db.save("test", k, {"_id": k})
    out = db.get_many("test", ["c", "a", "b"])
    assert list(out.keys()) == ["c", "a", "b"]


# --- delete_many --------------------------------------------------------

def test_delete_many_removes_all_listed_docs(db: MemoryDBService) -> None:
    for k in ["a", "b", "c"]:
        db.save("test", k, {"_id": k})
    results = db.delete_many("test", ["a", "b"])
    assert all(r["ok"] for r in results)
    # a and b are gone, c remains.
    assert db.get_many("test", ["a", "b", "c"]) == {
        "a": None, "b": None, "c": {"_id": "c"},
    }


def test_delete_many_idempotent_on_missing_docs(db: MemoryDBService) -> None:
    """Deleting a doc that doesn't exist is a successful no-op, not an
    error.  Important for retry scenarios."""
    results = db.delete_many("test", ["never-existed"])
    assert results[0]["ok"] is True


def test_delete_many_empty_input(db: MemoryDBService) -> None:
    assert db.delete_many("test", []) == []
