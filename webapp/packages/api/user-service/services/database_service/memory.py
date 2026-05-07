from typing import Any, Dict, List, Optional
from fastapi import HTTPException
from .base import DatabaseService


class MemoryDBService(DatabaseService):
    """In-memory dictionary implementation for testing or when no DB is configured."""

    def __init__(self):
        self.dbs: Dict[str, Dict[str, Any]] = {}
        print("Using in-memory database service.")

    def get(self, db_name: str, doc_id: str) -> Dict[str, Any]:
        if db_name not in self.dbs or doc_id not in self.dbs[db_name]:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in '{db_name}'")
        return self.dbs[db_name][doc_id]

    def save(self, db_name: str, doc_id: str, doc: Dict[str, Any]) -> Dict[str, Any]:
        if db_name not in self.dbs:
            self.dbs[db_name] = {}
        self.dbs[db_name][doc_id] = doc
        return {"id": doc_id, "rev": "memory-rev"}

    def delete(self, db_name: str, doc_id: str):
        if db_name in self.dbs and doc_id in self.dbs[db_name]:
            del self.dbs[db_name][doc_id]
        else:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found for deletion.")

    def list_all(self, db_name: str) -> List[Dict[str, Any]]:
        return list(self.dbs.get(db_name, {}).values())

    def save_many(
        self,
        db_name: str,
        docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if db_name not in self.dbs:
            self.dbs[db_name] = {}
        results: List[Dict[str, Any]] = []
        for doc in docs:
            doc_id = doc.get("_id")
            if not doc_id:
                results.append({"ok": False, "id": None, "error": "missing _id"})
                continue
            self.dbs[db_name][doc_id] = doc
            results.append({"ok": True, "id": doc_id, "rev": "memory-rev"})
        return results

    def delete_many(
        self,
        db_name: str,
        doc_ids: List[str],
    ) -> List[Dict[str, Any]]:
        store = self.dbs.get(db_name, {})
        results: List[Dict[str, Any]] = []
        for doc_id in doc_ids:
            store.pop(doc_id, None)  # idempotent — missing is fine
            results.append({"ok": True, "id": doc_id})
        return results

    def get_many(
        self,
        db_name: str,
        doc_ids: List[str],
    ) -> Dict[str, "Optional[Dict[str, Any]]"]:
        store = self.dbs.get(db_name, {})
        return {doc_id: store.get(doc_id) for doc_id in doc_ids}