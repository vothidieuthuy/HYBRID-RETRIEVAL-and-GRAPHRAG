"""Kiểm tra đồ thị trong Neo4j có đúng cấu trúc LEGAL_ONTOLOGY.md và khớp graph.json không.

Mỗi mục in ĐẠT / CẢNH BÁO / LỖI kèm tối đa vài ví dụ. LỖI = sai ontology hoặc lệch dữ liệu nguồn; CẢNH BÁO = đúng
ontology nhưng nên xem lại (thường là lỗi trích xuất của LLM).

    .venv-lc/Scripts/python -m lc_app.graph.check
"""
from __future__ import annotations

import json
import re
import sys

from langchain_neo4j import Neo4jGraph

from ..config import Config
from .build import grounded
from .ontology import DOMAIN_RANGE, ENTITY_TYPES, GENERIC_NAMES, RELATION_TYPES, UNIT_NAME

ARTICLE = UNIT_NAME
DOC_NUMBER = re.compile(r"\d+/\d{4}/[\wĐđ-]+")
SHOW = 5


class Report:
    def __init__(self) -> None:
        self.errors = 0

    def line(self, ok: bool, title: str, bad: list, level: str = "LỖI") -> None:
        status = "ĐẠT" if ok else level
        self.errors += (not ok) and level == "LỖI"
        print(f"[{status:9}] {title}" + ("" if ok else f": {len(bad)} trường hợp"))
        for item in ([] if ok else bad[:SHOW]):
            print(f"              - {item}")


def main() -> None:
    config = Config.from_env()
    neo4j = Neo4jGraph(**config.neo4j, refresh_schema=False)
    q = neo4j.query
    graph = json.loads((config.graph_dir / "graph.json").read_text(encoding="utf-8"))
    report = Report()
    print("== 1. Loại nút và quan hệ có đúng ontology không")
    labels = {r["label"] for r in q("CALL db.labels() YIELD label RETURN label")}
    extra = labels - {"__Entity__", "Chunk", *ENTITY_TYPES}
    report.line(not extra, "chỉ có nhãn Chunk, __Entity__ và 4 loại nút", sorted(extra))
    wrong = q("MATCH (e:__Entity__) WITH e, [l IN labels(e) WHERE l <> '__Entity__'] AS ls "
              "WHERE size(ls) <> 1 RETURN e.id AS id, ls")
    report.line(not wrong, "mỗi thực thể đúng một loại", [f"{r['id']}: {r['ls']}" for r in wrong])
    types = {r["t"] for r in q("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t")}
    extra = types - {"MENTIONS", *RELATION_TYPES}
    report.line(not extra, "chỉ có MENTIONS và 5 loại quan hệ", sorted(extra))

    print("== 2. Miền/đích của quan hệ (ai được nối với ai)")
    for kind, (domain, target) in DOMAIN_RANGE.items():
        rows = q(f"MATCH (s)-[:{kind}]->(t) RETURN s.id AS s, t.id AS t, "
                 "[l IN labels(s) WHERE l <> '__Entity__'][0] AS ls, [l IN labels(t) WHERE l <> '__Entity__'][0] AS lt")
        bad = [f"{r['s']} ({r['ls']}) -{kind}-> {r['t']} ({r['lt']})" for r in rows
               if r["ls"] not in domain or r["lt"] not in target]
        report.line(not bad, f"{kind}: {'/'.join(sorted(domain))} -> {'/'.join(sorted(target))} ({len(rows)} cạnh)",
                    bad, "CẢNH BÁO")
    bad = q("MATCH (a)-[:MENTIONS]->(b) WHERE NOT a:Chunk OR NOT b:__Entity__ RETURN count(*) AS n")[0]["n"]
    report.line(bad == 0, "MENTIONS chỉ đi từ Chunk tới thực thể", [f"{bad} cạnh sai"])

    print("== 3. Khớp với graph.json (dữ liệu đã gộp)")
    counts = {r["t"]: r["n"] for r in q("MATCH (e:__Entity__) RETURN [l IN labels(e) WHERE l <> '__Entity__'][0] "
                                         "AS t, count(*) AS n")}
    expected = {}
    for node in graph["nodes"]:
        expected[node["type"]] = expected.get(node["type"], 0) + 1
    report.line(counts == expected, "số nút theo loại", [f"Neo4j {counts} ≠ graph.json {expected}"])
    rels = {r["t"]: r["n"] for r in q("MATCH ()-[r]->() WHERE type(r) <> 'MENTIONS' RETURN type(r) AS t, "
                                      "count(*) AS n")}
    expected = {}
    for edge in graph["edges"]:
        expected[edge["type"]] = expected.get(edge["type"], 0) + 1
    report.line(rels == expected, "số cạnh theo loại", [f"Neo4j {rels} ≠ graph.json {expected}"])

    print("== 4. Nguồn (mỗi thực thể/cạnh truy được về chunk)")
    orphan = q("MATCH (e:__Entity__) WHERE NOT ()-[:MENTIONS]->(e) RETURN e.id AS id")
    report.line(not orphan, "mọi thực thể có ít nhất một chunk nguồn (MENTIONS)", [r["id"] for r in orphan])
    units = set()
    with config.chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("graph_unit"):
                units.add(record["chunk_id"])
    chunks = [r["id"] for r in q("MATCH (c:Chunk) RETURN c.id AS id")]
    report.line(set(chunks) <= units, "nút Chunk đều là text unit (graph_unit = true)",
                sorted(set(chunks) - units))
    mismatch = q("MATCH (e:__Entity__) OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e) WITH e, collect(c.id) AS src "
                 "WHERE apoc.coll.sort(src) <> apoc.coll.sort(e.chunk_ids) RETURN e.id AS id")
    report.line(not mismatch, "thuộc tính chunk_ids khớp cạnh MENTIONS", [r["id"] for r in mismatch])
    no_source = q("MATCH (s)-[r]->(t) WHERE type(r) <> 'MENTIONS' AND (r.chunk_ids IS NULL OR size(r.chunk_ids) = 0) "
                  "RETURN s.id + ' -' + type(r) + '-> ' + t.id AS e")
    report.line(not no_source, "mọi cạnh có chunk_ids nguồn", [r["e"] for r in no_source])

    print("== 5. Quy tắc đặt tên")
    entities = q("MATCH (e:__Entity__) RETURN e.id AS id, [l IN labels(e) WHERE l <> '__Entity__'][0] AS t")
    report.line(not [e for e in entities if e["id"].casefold() in GENERIC_NAMES], "không có nút chung chung",
                [e["id"] for e in entities if e["id"].casefold() in GENERIC_NAMES])
    bad = [e["id"] for e in entities if e["t"] in ("VAN_BAN", "DIEU_KHOAN") and re.search(r"\bsố \d", e["id"])]
    report.line(not bad, "tên văn bản không có chữ \"số\"", bad)
    bad = [e["id"] for e in entities if e["t"] == "DIEU_KHOAN" and not ARTICLE.match(e["id"])]
    report.line(not bad, "tên Điều dạng \"Điều <số> <tên văn bản>\" hoặc \"Phụ lục <số> <tên văn bản>\"", bad)
    bad = [e["id"] for e in entities if e["t"] == "DIEU_KHOAN" and re.search(r"[Kk]hoản|[Đđ]iểm \w", e["id"])]
    report.line(not bad, "không có nút cấp Khoản/Điểm", bad)
    names = {}
    for e in entities:
        names.setdefault(e["id"].casefold(), []).append(e["id"])
    bad = [v for v in names.values() if len(v) > 1]
    report.line(not bad, "không có hai nút chỉ khác hoa/thường", bad)
    bad = [e["id"] for e in entities if e["t"] == "VAN_BAN" and not DOC_NUMBER.search(e["id"])]
    report.line(not bad, "văn bản có số hiệu", bad, "CẢNH BÁO")

    print("== 6. Mô tả (chỉ dựa trên đoạn văn)")
    # Điều chỉ được nhắc tới (không có CHUA đi vào) mà mô tả nêu "quy định về <chủ đề>": hợp lệ khi chủ đề nằm cùng
    # mệnh đề với số Điều trong đoạn văn nguồn ("Hạn mức công nhận đất ở theo quy định tại khoản 5 Điều 141"),
    # là đoán khi đoạn văn chỉ liệt kê số Điều ("..., gồm: khoản 4 Điều 50, khoản 1 Điều 55, ...")
    rows = q("MATCH (d:DIEU_KHOAN) WHERE NOT ()-[:CHUA]->(d) MATCH (c:Chunk)-[:MENTIONS]->(d) "
             "RETURN d.id AS id, d.description AS description, collect(c.text) AS texts")
    guessed = [f"{r['id']}: {r['description'][:120]}" for r in rows
               if not grounded(r["id"], r["description"], r["texts"])]
    report.line(not guessed, "mô tả Điều chỉ được nhắc tới không đoán nội dung (chủ đề phải cùng câu với số Điều)",
                guessed, "CẢNH BÁO")

    print("== 7. Cấu trúc đồ thị")
    alone = q("MATCH (e:__Entity__) WHERE NOT (e)-[:CHUA|HUONG_DAN|THAM_CHIEU|SUA_DOI|AP_DUNG_CHO]-() RETURN e.id AS id")
    report.line(not alone, "thực thể nào cũng có ít nhất một quan hệ ontology", [r["id"] for r in alone], "CẢNH BÁO")
    no_parent = q("MATCH (c:Chunk)-[:MENTIONS]->(d:DIEU_KHOAN) WITH c, collect(d) AS ds "
                  "WHERE none(d IN ds WHERE ()-[:CHUA]->(d)) RETURN c.id AS id")
    report.line(not no_parent, "mỗi chunk có Điều được nối CHUA từ văn bản của nó", [r["id"] for r in no_parent],
                "CẢNH BÁO")
    print(f"\n{report.errors} mục LỖI." if report.errors else "\nKhông có mục LỖI.")
    sys.exit(1 if report.errors else 0)


if __name__ == "__main__":
    main()
