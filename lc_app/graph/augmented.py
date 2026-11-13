"""Chế độ graph-augmented-hybrid (truy hồi lai tăng cường bằng đồ thị): hybrid-rrf giữ nguyên thứ tự; với `seeds` chunk
đầu (hạt giống), đi một bước trên đồ thị (LEGAL_ONTOLOGY.md) để tìm căn cứ thứ hai mà câu hỏi không nêu ra, rồi chèn ngay
sau hạt giống. (Tên cũ của chế độ: hybrid-graph-amend.)

Hai loại bước nhảy (ứng viên của cả hai được xếp chung theo cosine với câu hỏi, mỗi hạt giống lấy tối đa `per_seed`):
  SUA_DOI   : Điều của hạt giống là đích của cạnh SUA_DOI -> chunk nhắc tới cả bên sửa đổi lẫn Điều bị sửa (nội dung
              sửa đổi, bổ sung, bãi bỏ).
  DAN_CHIEU : chunk hạt giống nhắc tới (MENTIONS) một Điều khác -> chunk thuộc chính Điều đó (nội dung được dẫn chiếu,
              ví dụ "theo quy định tại Điều 133 Luật Đất đai", "trường hợp quy định tại khoản 3 Điều 14 Nghị định này").
Không có ngân sách riêng: chunk chèn vào cạnh tranh trong cùng ngân sách ngữ cảnh. Điều của một chunk xác định từ metadata
của chính chunk (dieu, document_number); nút DIEU_KHOAN có id dạng "Điều <số> <loại văn bản> <số hiệu>".
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_neo4j import Neo4jGraph
from pydantic import ConfigDict, PrivateAttr

AMENDED = """
UNWIND $seeds AS s
WITH s WHERE s.dieu IS NOT NULL
MATCH (c:Chunk {id: s.id})-[:MENTIONS]->(x:DIEU_KHOAN)
WHERE x.id STARTS WITH s.dieu + ' ' AND x.id ENDS WITH s.num
MATCH (x)<-[:SUA_DOI]-(a)<-[:MENTIONS]-(m:Chunk)-[:MENTIONS]->(x)
WHERE m.id <> s.id
RETURN s.id AS seed, collect(DISTINCT m.id) AS chunks, collect(DISTINCT a.id) AS via
"""

REFERENCED = """
UNWIND $seeds AS s
MATCH (c:Chunk {id: s.id})-[:MENTIONS]->(y:DIEU_KHOAN)
WHERE s.dieu IS NULL OR NOT (y.id STARTS WITH s.dieu + ' ' AND y.id ENDS WITH s.num)
RETURN s.id AS seed, collect(DISTINCT y.id) AS targets
"""

ARTICLE = re.compile(r"^Điều (\S+) .*?(\S+)$")   # "Điều 133 Luật Đất đai 31/2024/QH15" -> ("133", "31/2024/QH15")


class GraphAugmentedHybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base: BaseRetriever
    graph: Neo4jGraph
    documents: dict[str, Document]       # chunk_id -> Document truy hồi được (cùng bộ với hybrid)
    store: FAISS
    seeds: int = 5
    per_seed: int = 1
    _row: dict[str, int] = PrivateAttr(default_factory=dict)
    _home: dict[tuple[str, str], list[str]] = PrivateAttr(default_factory=dict)

    def similarity(self, query: str, chunk_ids: set[str]) -> dict[str, float]:
        if not self._row:
            self._row.update({doc_id: i for i, doc_id in self.store.index_to_docstore_id.items()})
        vector = np.asarray(self.store.embeddings.embed_query(query), dtype="float32")
        return {c: float(vector @ self.store.index.reconstruct(self._row[c])) for c in chunk_ids if c in self._row}

    def article_chunks(self, node_id: str) -> list[str]:
        """Các chunk thuộc chính Điều có id nút `node_id` (theo metadata dieu + document_number của chunk)."""
        if not self._home:
            home = defaultdict(list)
            for doc_id, doc in self.documents.items():
                if doc.metadata.get("dieu") and doc.metadata.get("document_number"):
                    home[(str(doc.metadata["dieu"]), doc.metadata["document_number"])].append(doc_id)
            self._home.update(home)
        match = ARTICLE.match(node_id)
        return self._home.get(match.groups(), []) if match else []

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        ranked = self.base.invoke(query, config={"callbacks": run_manager.get_child()})
        seeds = []
        for d in ranked[:self.seeds]:
            own = d.metadata.get("dieu") and d.metadata.get("document_number")
            seeds.append({"id": d.id, "dieu": f"Điều {d.metadata['dieu']}" if own else None,
                          "num": d.metadata["document_number"] if own else None})
        # candidates[seed][chunk] = (quan hệ, qua nút nào); SUA_DOI được ghi trước nên giữ nhãn khi trùng
        candidates: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
        if seeds:
            for row in self.graph.query(AMENDED, {"seeds": seeds}):
                via = ", ".join(row["via"])
                for chunk in row["chunks"]:
                    if chunk in self.documents:
                        candidates[row["seed"]].setdefault(chunk, ("SUA_DOI", via))
            for row in self.graph.query(REFERENCED, {"seeds": seeds}):
                for target in row["targets"]:
                    for chunk in self.article_chunks(target):
                        if chunk != row["seed"]:
                            candidates[row["seed"]].setdefault(chunk, ("DAN_CHIEU", target))
        cosine = self.similarity(query, {c for found in candidates.values() for c in found})
        result, placed = [], set()
        for doc in ranked:
            if doc.id in placed:
                continue
            result.append(doc)
            placed.add(doc.id)
            found = candidates.get(doc.id, {})
            for chunk in sorted((c for c in found if c not in placed), key=lambda c: -cosine.get(c, 0.0))[:self.per_seed]:
                extra = self.documents[chunk]
                rel, via = found[chunk]
                result.append(Document(page_content=extra.page_content, id=extra.id,
                                       metadata={**extra.metadata, "graph_seed": doc.id, "graph_rel": rel,
                                                 "graph_via": via}))
                placed.add(chunk)
        return result
