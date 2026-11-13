"""Bước 3: nạp graph.json vào Neo4j bằng LangChain (Neo4jGraph.query, Cypher thuần).

Nạp bằng Cypher UNWIND thay cho add_graph_documents để giữ chunk_ids trên nút/cạnh và đặt nhãn Chunk trực tiếp; schema
giống schema mà add_graph_documents(include_source=True, baseEntityLabel=True) tạo ra (LEGAL_ONTOLOGY.md mục 6).
Neo4j chạy bằng docker-compose.yml ở thư mục gốc (có APOC cho refresh_schema/GraphCypherQAChain).
    (:Chunk {id: chunk_id, text, header, doc_key})-[:MENTIONS]->(:__Entity__:<LOẠI> {id, description, chunk_ids})
    (:__Entity__)-[:<LOẠI_QUAN_HỆ> {description, weight, chunk_ids}]->(:__Entity__)

    .venv-lc/Scripts/python -m lc_app.graph.load_neo4j [--reset]
--reset xóa toàn bộ dữ liệu trong database đích trước khi nạp.
"""
from __future__ import annotations

import argparse
import json

from langchain_neo4j import Neo4jGraph

from ..config import Config
from .ontology import ENTITY_TYPES, RELATION_TYPES


BATCH = 1000   # số dòng mỗi giao dịch: đồ thị đầy đủ có hàng chục nghìn nút/cạnh, một giao dịch dễ hết bộ nhớ heap


def batched(neo4j: Neo4jGraph, cypher: str, rows: list[dict]) -> None:
    for start in range(0, len(rows), BATCH):
        neo4j.query(cypher, {"rows": rows[start:start + BATCH]})


def load(neo4j: Neo4jGraph, graph: dict, chunks: dict[str, dict]) -> None:
    neo4j.query("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.id IS UNIQUE")
    neo4j.query("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
    batched(neo4j, "UNWIND $rows AS row MERGE (c:Chunk {id: row.chunk_id}) "
                   "SET c.text = row.text, c.header = row.header, c.doc_key = row.doc_key",
            [{k: c[k] for k in ("chunk_id", "text", "header", "doc_key")} for c in chunks.values()])
    for kind in ENTITY_TYPES:   # nhãn không truyền được bằng tham số Cypher nếu không có APOC: mỗi loại một lệnh
        batched(neo4j, f"UNWIND $rows AS row MERGE (e:__Entity__ {{id: row.id}}) SET e:{kind}, "
                       "e.description = row.description, e.chunk_ids = row.chunk_ids",
                [n for n in graph["nodes"] if n["type"] == kind])
    batched(neo4j, "UNWIND $rows AS row MATCH (c:Chunk {id: row.chunk}), (e:__Entity__ {id: row.entity}) "
                   "MERGE (c)-[:MENTIONS]->(e)",
            [{"chunk": c, "entity": n["id"]} for n in graph["nodes"] for c in n["chunk_ids"]])
    for kind in RELATION_TYPES:
        batched(neo4j, f"UNWIND $rows AS row MATCH (s:__Entity__ {{id: row.source}}), "
                       f"(t:__Entity__ {{id: row.target}}) MERGE (s)-[r:{kind}]->(t) "
                       "SET r.description = row.description, r.weight = row.weight, r.chunk_ids = row.chunk_ids",
                [e for e in graph["edges"] if e["type"] == kind])


def main() -> None:
    parser = argparse.ArgumentParser(description="Nạp đồ thị vào Neo4j.")
    parser.add_argument("--reset", action="store_true", help="xóa dữ liệu cũ trong database đích")
    args = parser.parse_args()
    config = Config.from_env()
    graph = json.loads((config.graph_dir / "graph.json").read_text(encoding="utf-8"))
    ids = {c for n in graph["nodes"] for c in n["chunk_ids"]}
    chunks = {}
    with config.chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["chunk_id"] in ids:
                chunks[record["chunk_id"]] = record
    neo4j = Neo4jGraph(**config.neo4j, refresh_schema=False)   # schema đọc lại sau khi nạp
    if args.reset:
        neo4j.query("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS")
    load(neo4j, graph, chunks)
    counts = neo4j.query("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY n DESC")
    rels = neo4j.query("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY n DESC")
    print("nút:", {c["label"]: c["n"] for c in counts})
    print("cạnh:", {r["type"]: r["n"] for r in rels})


if __name__ == "__main__":
    main()
