"""Cấu hình chung; đọc từ biến môi trường (tiền tố LC_) hoặc .env, mặc định theo cấu hình đã chốt của đề tài."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)


@dataclass(frozen=True)
class Config:
    chunks_path: Path = ROOT / "data/chunk/chunk_v1/clean_chunk_v1/chunks.jsonl"   # đầu ra kb_pipeline.clean_chunks
    index_dir: Path = ROOT / "data/chunk/chunk_v1/clean_chunk_v1/index"   # chỉ mục đi cùng phiên bản chunk
    testset: Path = ROOT / "data/ragas_testset_50.csv"
    embedding_model: str = "AITeamVN/Vietnamese_Embedding"
    embedding_max_tokens: int = 1024      # chunk dài nhất ~900 token: không cắt mất nội dung
    retrieval_mode: str = "hybrid-rrf"    # dense | bm25 | hybrid-rrf
    candidates: int = 100                 # số kết quả mỗi retriever con đưa vào RRF
    rrf_c: int = 60
    depth: int = 40                      # số chunk lấy ra trước khi cắt theo ngân sách
    context_chars: int = 8000             # ngân sách ngữ cảnh đưa vào LLM (ký tự)
    llm_model: str = "gpt-4o-mini"
    # Đồ thị (GraphRAG theo LEGAL_ONTOLOGY.md): kết quả trích xuất ở graph_dir, đồ thị đã gộp nạp vào Neo4j
    graph_dir: Path = ROOT / "data/chunk/chunk_v1/clean_chunk_v1/graph_full"
    graph_llm_model: str = "gpt-4o-mini"
    # cache LLM dùng chung cho mọi graph_dir (demo, chạy thử, toàn bộ): chunk đã trích ở lần trước không tốn tiền lại
    graph_cache: Path = ROOT / "data/chunk/chunk_v1/clean_chunk_v1/graph_llm_cache.sqlite"
    graph_seeds: int = 5                  # graph-augmented-hybrid: số chunk đầu của hybrid được mở rộng qua đồ thị (sửa đổi, dẫn chiếu)

    @classmethod
    def from_env(cls) -> "Config":
        env = os.getenv
        return cls(
            embedding_model=env("LC_EMBEDDING_MODEL", cls.embedding_model),
            retrieval_mode=env("LC_RETRIEVAL_MODE", cls.retrieval_mode),
            context_chars=int(env("LC_CONTEXT_CHARS", str(cls.context_chars))),
            llm_model=env("OPENAI_GENERATION_MODEL", env("OPENAI_MODEL", cls.llm_model)),
            graph_dir=Path(env("LC_GRAPH_DIR", str(cls.graph_dir))),
            graph_llm_model=env("LC_GRAPH_MODEL", cls.graph_llm_model),
        )

    @property
    def neo4j(self) -> dict:
        """Tham số kết nối cho langchain_neo4j.Neo4jGraph, đọc từ .env (NEO4J_URI/USER/PASSWORD/DATABASE)."""
        return {"url": os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"), "username": os.getenv("NEO4J_USER", "neo4j"),
                "password": os.environ["NEO4J_PASSWORD"], "database": os.getenv("NEO4J_DATABASE", "neo4j")}

    @property
    def faiss_dir(self) -> Path:
        return self.index_dir / ("faiss_" + self.embedding_model.replace("/", "__"))
