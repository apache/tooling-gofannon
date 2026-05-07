"""Tests for the round-trip-count-bounded data_store_service paths.

These tests use a counting-wrapper around MemoryDBService that records
each backend method call.  The asserts verify that the new bulk paths
make the expected number of calls, not just that they produce the
right output.  The point is to lock in the perf invariants so a
future refactor doesn't quietly regress them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from services.data_store_service import DataStoreService
from services.database_service.memory import MemoryDBService

pytestmark = pytest.mark.unit


class CountingDB(MemoryDBService):
    """MemoryDBService that counts every public method call by name.

    Lets tests assert "this service-layer operation made K backend
    calls" without instrumenting the service code.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls: Dict[str, int] = {}

    def _bump(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def get(self, db_name: str, doc_id: str) -> Dict[str, Any]:
        self._bump("get")
        return super().get(db_name, doc_id)

    def save(self, db_name: str, doc_id: str, doc: Dict[str, Any]) -> Dict[str, Any]:
        self._bump("save")
        return super().save(db_name, doc_id, doc)

    def delete(self, db_name: str, doc_id: str) -> None:
        self._bump("delete")
        super().delete(db_name, doc_id)

    def get_many(self, db_name: str, doc_ids: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        self._bump("get_many")
        return super().get_many(db_name, doc_ids)

    def save_many(self, db_name: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._bump("save_many")
        return super().save_many(db_name, docs)

    def delete_many(self, db_name: str, doc_ids: List[str]) -> List[Dict[str, Any]]:
        self._bump("delete_many")
        return super().delete_many(db_name, doc_ids)

    def find(self, db_name: str, selector, fields=None, limit: int = 10000):
        self._bump("find")
        return super().find(db_name, selector, fields, limit)


@pytest.fixture
def svc():
    db = CountingDB()
    return DataStoreService(db), db


# --- set() optimistic path: one save call ------------------------------

def test_set_new_key_does_one_save(svc) -> None:
    """Writing a brand-new key should be a single backend call.
    The old shape did get() then save() = 2 calls."""
    s, db = svc
    s.set("user1", "ns", "key1", "value1")
    assert db.calls.get("save") == 1
    assert db.calls.get("get", 0) == 0  # no pre-read


# --- set_many: bounded backend calls regardless of N -------------------

def test_set_many_does_one_get_many_and_one_save_many(svc) -> None:
    """Bulk write of N items should make 2 backend calls total
    (one fetch, one save) — not N or 2N."""
    s, db = svc
    items = [("ns", f"key{i}", f"value{i}", None) for i in range(50)]
    s.set_many("user1", items)
    assert db.calls.get("get_many") == 1
    assert db.calls.get("save_many") == 1
    # And critically: no per-key save() or get() calls.
    assert db.calls.get("save", 0) == 0
    assert db.calls.get("get", 0) == 0


def test_set_many_correctly_persists_all_values(svc) -> None:
    """Sanity: the bulk path actually writes the data."""
    s, db = svc
    items = [("ns", f"key{i}", f"value{i}", None) for i in range(5)]
    n = s.set_many("user1", items)
    assert n == 5
    out = s.get_many("user1", "ns", [f"key{i}" for i in range(5)])
    assert out == {f"key{i}": f"value{i}" for i in range(5)}


# --- get_many: one backend call ----------------------------------------

def test_get_many_uses_bulk_backend_call(svc) -> None:
    s, db = svc
    items = [("ns", f"key{i}", f"value{i}", None) for i in range(10)]
    s.set_many("user1", items)
    db.calls.clear()

    s.get_many("user1", "ns", [f"key{i}" for i in range(10)])
    assert db.calls.get("get_many") == 1
    assert db.calls.get("get", 0) == 0


# --- get_all: drops inline access tracking saves -----------------------

def test_get_all_with_agent_name_does_no_inline_saves(svc) -> None:
    """Old behavior: one find() + N save() calls for access tracking.
    New behavior: one find() and access tracking is queued in the
    accumulator (no immediate backend calls)."""
    s, db = svc
    items = [("ns", f"key{i}", f"value{i}", None) for i in range(20)]
    s.set_many("user1", items)
    db.calls.clear()

    s.get_all("user1", "ns", agent_name="agent_a")
    # find() does the read; no per-doc saves should fire.
    assert db.calls.get("find") == 1
    assert db.calls.get("save", 0) == 0


# --- clear_namespace: one find + one delete_many -----------------------

def test_clear_namespace_uses_bulk_delete(svc) -> None:
    s, db = svc
    items = [("ns", f"key{i}", f"value{i}", None) for i in range(15)]
    s.set_many("user1", items)
    db.calls.clear()

    n = s.clear_namespace("user1", "ns")
    assert n == 15
    assert db.calls.get("delete_many") == 1
    assert db.calls.get("delete", 0) == 0


def test_clear_namespace_actually_removes_docs(svc) -> None:
    s, db = svc
    s.set_many("user1", [("ns", "k1", "v1", None), ("ns", "k2", "v2", None)])
    s.clear_namespace("user1", "ns")
    out = s.get_many("user1", "ns", ["k1", "k2"])
    assert out == {}


# --- set() conflict retry ----------------------------------------------

def test_set_retries_once_on_conflict(svc) -> None:
    """When the optimistic write loses, set() should re-fetch and
    retry exactly once.  Total backend calls: save (fail), get
    (re-fetch), save (succeed) = 3."""
    s, db = svc
    # Pre-populate the doc so the optimistic write hits a conflict.
    # We do this by manually injecting a doc that the service-layer
    # set() doesn't know exists.
    doc_id = s._make_doc_id("user1", "ns", "key1")
    db.dbs["agent_data_store"] = {
        doc_id: {
            "_id": doc_id, "userId": "user1", "namespace": "ns",
            "key": "key1", "value": "old", "metadata": {},
            "accessCount": 0, "createdAt": "2020-01-01T00:00:00",
        },
    }

    # Hack: the memory backend's save() succeeds even on conflict.
    # To exercise the retry path we need a mock that fails the first
    # save and succeeds the second.  Patch save() inline.
    save_calls = {"n": 0}
    original_save = db.save

    def conflicting_save(db_name, doc_id_, doc_):
        save_calls["n"] += 1
        if save_calls["n"] == 1:
            from fastapi import HTTPException
            raise HTTPException(status_code=409, detail="conflict")
        return original_save(db_name, doc_id_, doc_)

    db.save = conflicting_save  # type: ignore

    result = s.set("user1", "ns", "key1", "new_value")

    assert save_calls["n"] == 2  # one fail + one succeed
    assert result["value"] == "new_value"


# --- access tracking accumulator -------------------------------------

def test_access_accumulator_records_doc_ids(svc) -> None:
    """get_all with agent_name should populate the accumulator's buffer."""
    s, db = svc
    s.set_many("user1", [("ns", "k1", "v1", None), ("ns", "k2", "v2", None)])
    db.calls.clear()

    s.get_all("user1", "ns", agent_name="agent_x")

    # Accumulator buffer should contain entries for the read docs;
    # background flush hasn't fired yet (we don't await).
    buf = s._access_accumulator._buffer
    assert len(buf) == 2
    for entry in buf.values():
        assert entry["agent_name"] == "agent_x"
        assert entry["delta"] == 1


def test_access_accumulator_collapses_repeat_reads(svc) -> None:
    """Reading the same doc twice within a flush window should
    collapse into one buffer entry with delta=2."""
    s, _ = svc
    s.set("user1", "ns", "k1", "v1")
    s.get_many("user1", "ns", ["k1"], agent_name="agent_x")
    s.get_many("user1", "ns", ["k1"], agent_name="agent_x")

    buf = s._access_accumulator._buffer
    assert len(buf) == 1
    entry = next(iter(buf.values()))
    assert entry["delta"] == 2
