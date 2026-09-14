from __future__ import annotations

import json
import sqlite3
from pathlib import Path


db = Path(r"D:\Study\DSC2026\LegalIR\cache\exp112_task_adaptive_retrieval\sources.sqlite")
con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
print(con.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall())
print("counts", con.execute("SELECT source,COUNT(*),MIN(LENGTH(payload)),MAX(LENGTH(payload)) FROM sources GROUP BY source ORDER BY source").fetchall())
print("signatures", con.execute("SELECT source,signature,COUNT(*) FROM sources GROUP BY source,signature ORDER BY source,COUNT(*) DESC").fetchall())
for qid, source, payload, signature in con.execute(
    "SELECT q,source,payload,signature FROM sources LIMIT 5"
):
    values = json.loads(payload)
    print(qid, source, signature, len(values), values[:2])
con.close()

jina_db = Path(r"D:\Study\DSC2026\sota\results\research_v2_forensic\research_v2_jina_boundary\evidence_ab_scores.sqlite")
jina = sqlite3.connect(f"file:{jina_db.as_posix()}?mode=ro", uri=True)
print("jina tables", jina.execute("SELECT name,sql FROM sqlite_master WHERE type='table'").fetchall())
for table in [row[0] for row in jina.execute("SELECT name FROM sqlite_master WHERE type='table'")]:
    print(table, jina.execute(f"SELECT COUNT(*) FROM {table}").fetchone())
    print(jina.execute(f"SELECT * FROM {table} LIMIT 2").fetchall())
jina.close()
