"""Inspect what is actually stored in the index.   python diagnose_index.py

Reads chroma_db/ and doc_registry.json directly (no models, no API keys) and
reports, per file, how many chunks exist, how many are blank, and a text sample.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import chromadb

BASE = Path(__file__).resolve().parent
DB = BASE / "chroma_db"
REG = BASE / "doc_registry.json"


def main() -> int:
    if not DB.exists():
        print(f"No index at {DB}. Run the app and press 'Index documents' first.")
        return 1
    client = chromadb.PersistentClient(path=str(DB))
    cols = client.list_collections()
    print("collections:", [(getattr(c, 'name', c)) for c in cols])
    for c in cols:
        col = client.get_collection(getattr(c, "name", c))
        data = col.get(include=["documents", "metadatas"])
        docs, metas = data["documents"] or [], data["metadatas"] or []
        print(f"\n== collection {col.name!r}: {len(docs)} chunks")
        per = defaultdict(lambda: {"n": 0, "blank": 0, "chars": 0, "sample": ""})
        for d, m in zip(docs, metas):
            f = (m or {}).get("file_name", "?")
            p = per[f]
            p["n"] += 1
            text = d or ""
            p["chars"] += len(text.strip())
            if not text.strip():
                p["blank"] += 1
            elif not p["sample"]:
                p["sample"] = " ".join(text.split())[:140]
        for f, p in sorted(per.items()):
            flag = "  <-- ALL BLANK" if p["blank"] == p["n"] else ("  <-- some blank" if p["blank"] else "")
            print(f"  {f}: {p['n']} chunks, {p['blank']} blank, {p['chars']} chars{flag}")
            if p["sample"]:
                print(f"      sample: {p['sample']!r}")
    if REG.exists():
        reg = json.loads(REG.read_text())
        print("\nregistry:")
        for path, e in reg.items():
            print(f"  {Path(path).name}: schema={e.get('schema')} chunks={e.get('chunks')}")
    else:
        print("\n(no doc_registry.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
