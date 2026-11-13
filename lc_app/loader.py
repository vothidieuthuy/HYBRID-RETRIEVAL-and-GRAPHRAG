"""Nạp chunk của KB (chunks.jsonl, đã chia theo Điều/Khoản) thành Document của LangChain.

page_content = embed_text (header văn bản/Điều + ghi chú sửa đổi + nội dung): dùng để nhúng và đánh chỉ mục BM25.
metadata["text"] = nội dung gốc: phần đưa vào LLM và dùng khi đánh giá.
Chỉ lấy chunk truy hồi được, còn hiệu lực, ở cấp 0 (thẻ văn bản) và 2 (Điều/Khoản).
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

KEPT = ("level", "doc_key", "document_number", "document_type", "doc_title", "issue_date", "effective_date",
        "validity_status", "dieu", "dieu_title", "khoan", "header", "source_file")


class LegalChunkLoader(BaseLoader):
    def __init__(self, path: str | Path, levels: tuple[int, ...] = (0, 2), include_inactive: bool = False):
        self.path, self.levels, self.include_inactive = Path(path), levels, include_inactive

    def lazy_load(self) -> Iterator[Document]:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["level"] not in self.levels or not record.get("retrievable"):
                    continue
                if not self.include_inactive and not record.get("active", True):
                    continue
                text, header = record["text"], record.get("header") or ""
                embed_text = record.get("embed_text") or text
                metadata = {key: record.get(key) for key in KEPT}
                metadata["khoan"] = ", ".join(metadata["khoan"]) if metadata["khoan"] else None
                # ghi chú "đã được sửa đổi/bãi bỏ bởi ..." nằm giữa header và nội dung trong embed_text
                metadata["notes"] = embed_text[len(header):len(embed_text) - len(text)].strip() or None
                metadata["text"] = text
                yield Document(page_content=embed_text, metadata=metadata, id=record["chunk_id"])


ANNEX = re.compile(r"(?i:phụ lục)(?:\s+([IVXLC]+|\d+)\b)?")


def citation(doc: Document) -> str:
    meta = doc.metadata
    annex = None
    if not meta.get("dieu"):   # chunk phụ lục: tên phụ lục ở đoạn cuối của header ("... > Phụ lục I TRÌNH TỰ ...]")
        match = ANNEX.search((meta.get("header") or "").rsplit(" > ", 1)[-1])
        annex = match and ("Phụ lục " + match.group(1) if match.group(1) else "Phụ lục")
    location = " ".join(x for x in (f"Điều {meta['dieu']}" if meta.get("dieu") else annex,
                                    f"Khoản {meta['khoan']}" if meta.get("khoan") else None) if x)
    return " - ".join(x for x in (meta.get("doc_title") or "Văn bản chưa rõ tên", meta.get("document_number"),
                                  location or None) if x)
