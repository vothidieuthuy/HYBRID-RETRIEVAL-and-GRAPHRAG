"""Dựng chỉ mục FAISS (một lần, chạy CPU, không gọi API).

    .venv-lc/Scripts/python -m lc_app.build_index
"""
from __future__ import annotations

import json
import time

from .config import Config
from .retrievers import build_faiss, load_documents, load_embeddings


def main() -> None:
    config = Config.from_env()
    documents = load_documents(config)
    print(f"{len(documents)} chunk từ {config.chunks_path}", flush=True)
    started = time.perf_counter()
    build_faiss(config, documents, load_embeddings(config))
    manifest = {"chunks": len(documents), "chunks_path": str(config.chunks_path),
                "embedding_model": config.embedding_model, "max_tokens": config.embedding_max_tokens,
                "distance": "inner_product (vector chuẩn hóa = cosine)",
                "seconds": round(time.perf_counter() - started, 1)}
    (config.faiss_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print(f"Đã lưu {config.faiss_dir} ({manifest['seconds']} giây)")


if __name__ == "__main__":
    main()
