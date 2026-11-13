"""Bốn chế độ truy hồi dựng bằng thành phần của LangChain.

dense      : HuggingFaceEmbeddings (Vietnamese_Embedding, vector chuẩn hóa) + FAISS tích vô hướng (= cosine)
bm25       : BM25Retriever (rank_bm25, k1=1,2, b=0,75) với tách từ tiếng Việt (lc_app.vietnamese)
hybrid-rrf : EnsembleRetriever = Reciprocal Rank Fusion chuẩn của dense và bm25:
             score(d) = Σ 1 / (c + rank_i(d)), c=60
graph-augmented-hybrid (truy hồi lai tăng cường bằng đồ thị, cần Neo4j): GraphAugmentedHybridRetriever
             (lc_app.graph.augmented) giữ nguyên thứ tự hybrid-rrf, chèn sau mỗi chunk đầu một chunk tìm qua đồ thị
             LEGAL_ONTOLOGY.md: nội dung sửa đổi Điều của chunk (SUA_DOI) hoặc Điều mà chunk dẫn chiếu tới (MENTIONS).
             Tên cũ: hybrid-graph-amend.
Câu hỏi tổng hợp dùng GraphRAG global (lc_app.graph.global_search), không phải retriever chunk.
"""
from __future__ import annotations

from collections import defaultdict

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.storage import LocalFileStore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from .config import Config
from .loader import LegalChunkLoader
from .vietnamese import tokenize

BASE_MODES = ("dense", "bm25", "hybrid-rrf")
GRAPH_MODE = "graph-augmented-hybrid"            # cần Neo4j (lc_app.graph)
MODES = BASE_MODES + (GRAPH_MODE,)


class ChunkRRF(EnsembleRetriever):
    """RRF chuẩn, score(d) = Σ 1 / (c + rank_i(d)), cộng theo chunk_id (Document.id).

    EnsembleRetriever gốc gộp theo page_content khi id_key=None, nên các chunk khác nhau có nội dung trùng hệt
    (vài dòng phụ lục) bị nhập làm một và cộng dồn điểm. Thứ tự khi bằng điểm giữ như bản gốc (dense trước).
    """

    def weighted_reciprocal_rank(self, doc_lists: list[list[Document]]) -> list[Document]:
        scores: dict[str, float] = defaultdict(float)
        first: dict[str, Document] = {}
        for docs, weight in zip(doc_lists, self.weights, strict=True):
            for rank, doc in enumerate(docs, start=1):
                scores[doc.id] += weight / (rank + self.c)
                first.setdefault(doc.id, doc)
        return sorted(first.values(), key=lambda doc: scores[doc.id], reverse=True)


def load_documents(config: Config) -> list[Document]:
    return LegalChunkLoader(config.chunks_path).load()


def load_embeddings(config: Config) -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model, model_kwargs={"device": "cpu"},
                                       encode_kwargs={"normalize_embeddings": True, "batch_size": 8})
    embeddings._client.max_seq_length = config.embedding_max_tokens   # mặc định của mô hình là 512
    return embeddings


def build_faiss(config: Config, documents: list[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    # vector của từng chunk được ghi vào cache trên đĩa ngay khi nhúng xong: lỗi giữa chừng không phải nhúng lại
    cached = CacheBackedEmbeddings.from_bytes_store(
        embeddings, LocalFileStore(str(config.index_dir / "embedding_cache")),
        namespace=config.embedding_model, key_encoder="sha256", batch_size=64)
    store = FAISS.from_documents(documents, cached, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    store.embedding_function = embeddings   # truy vấn luôn nhúng trực tiếp, không qua cache
    # FAISS.save_local (C++) không mở được đường dẫn có dấu tiếng Việt trên Windows: tự ghi bằng Python
    config.faiss_dir.mkdir(parents=True, exist_ok=True)
    (config.faiss_dir / "faiss.bin").write_bytes(store.serialize_to_bytes())
    return store


def load_faiss(config: Config, embeddings: HuggingFaceEmbeddings) -> FAISS:
    path = config.faiss_dir / "faiss.bin"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có chỉ mục {path}: chạy python -m lc_app.build_index trước.")
    # tệp pickle do chính build_index tạo ra
    return FAISS.deserialize_from_bytes(path.read_bytes(), embeddings, allow_dangerous_deserialization=True,
                                        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)


class Retrievers:
    """Dựng một lần (nhúng truy vấn dùng chung mô hình), lấy retriever theo tên chế độ."""

    def __init__(self, config: Config, documents: list[Document] | None = None):
        self.config = config
        documents = documents if documents is not None else load_documents(config)
        self.store = load_faiss(config, load_embeddings(config))
        self.dense = self.store.as_retriever(search_kwargs={"k": config.candidates})
        self.bm25 = BM25Retriever.from_documents(documents, preprocess_func=tokenize, k=config.candidates,
                                                 bm25_params={"k1": 1.2, "b": 0.75})
        # EnsembleRetriever bắt buộc có weights; trọng số bằng nhau = RRF chuẩn không trọng số
        self.hybrid = ChunkRRF(retrievers=[self.dense, self.bm25], weights=[1.0, 1.0], c=config.rrf_c)
        self._graph_augmented = None

    @property
    def graph_augmented(self) -> BaseRetriever:
        if self._graph_augmented is None:
            from langchain_neo4j import Neo4jGraph

            from .graph.augmented import GraphAugmentedHybridRetriever
            self._graph_augmented = GraphAugmentedHybridRetriever(
                base=self.hybrid, graph=Neo4jGraph(**self.config.neo4j, refresh_schema=False),
                documents={doc.id: doc for doc in self.bm25.docs}, store=self.store, seeds=self.config.graph_seeds)
        return self._graph_augmented

    def get(self, mode: str) -> BaseRetriever:
        if mode not in MODES:
            raise ValueError(f"Chế độ {mode!r} không hợp lệ; chọn một trong {MODES}.")
        if mode == GRAPH_MODE:
            return self.graph_augmented
        return {"dense": self.dense, "bm25": self.bm25, "hybrid-rrf": self.hybrid}[mode]
