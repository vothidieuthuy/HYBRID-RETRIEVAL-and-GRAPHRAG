"""Đánh giá một bộ chunk mà không cần dựng FAISS: cấu trúc, độ phủ testset và truy hồi BM25.

1. Cấu trúc: phân bố độ dài chunk truy hồi được, chunk quá ngắn/quá dài, chunk vượt giới hạn token của mô hình
   nhúng (ước lượng 3 ký tự/token), nội dung trùng lặp, dòng text nguồn không nằm trong chunk nào (mất nội dung).
2. Độ phủ testset (oracle): với mỗi câu hỏi của ragas_testset_50, chọn các chunk khớp reference_contexts nhiều
   nhất trong ngân sách ngữ cảnh -> trần recall mà bất kỳ retriever nào cũng không vượt được. Câu có trần thấp
   là câu mà KB hiện tại không còn văn bản căn cứ.
3. Truy hồi BM25: cùng tokenizer và cùng cách chấm điểm với lc_app.evaluate_ir (Hit@k, MRR@10, Recall@ngân sách).

    python -m kb_pipeline.evaluate_chunks [--chunks DIR] [--extracted DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from langchain_community.retrievers import BM25Retriever

from lc_app.config import Config
from lc_app.evaluate_ir import grams, read_testset, score
from lc_app.loader import LegalChunkLoader
from lc_app.vietnamese import tokenize

ROOT = Path(__file__).resolve().parents[1]
SHORT, LONG, CHARS_PER_TOKEN = 300, 2400, 3.0
MARKERS = ("[BẢNG]", "[/BẢNG]")
DROPPED = re.compile(r"^Nơi nhận|\b(?:TM|KT|TL|Q)\.\s")


def percentile(values: list[int], q: float) -> int:
    return values[min(len(values) - 1, int(len(values) * q))] if values else 0


def structure(records: list[dict], extracted: Path, max_tokens: int) -> dict:
    usable = [r for r in records if r["retrievable"] and r.get("active", True)]
    sizes = sorted(len(r["text"]) for r in usable if r["level"] == 2)
    embed_tokens = [len(r["embed_text"]) / CHARS_PER_TOKEN for r in usable]
    texts = Counter(r["text"] for r in usable)
    # dòng nguồn không xuất hiện trong chunk nào (kể cả chunk không truy hồi): nội dung bị rơi khi chia
    # So sau khi bỏ khoảng trắng: dòng dài bị cắt theo câu khi chia chunk vẫn được tính là có mặt.
    squash = lambda s: "".join(s.split())  # noqa: E731
    by_source: dict[str, list[str]] = {}
    for r in records:
        parts = by_source.setdefault(r["source_file"], [])
        parts.append(squash(r["text"]))
        for heading in (r.get("chuong"), r.get("muc")):  # tiêu đề Chương/Mục nằm trong header, không trong text
            if heading:
                parts.extend(squash(x) for x in [heading, *heading.split(". ", 1)])
    lost, dropped_signature = {}, [0]
    for txt in sorted(extracted.glob("*/*/*.txt")):
        source = txt.relative_to(extracted).as_posix()
        joined = "\n".join(by_source.get(source, []))
        lines = [l for l in txt.read_text(encoding="utf-8").splitlines() if l and l not in MARKERS]
        # dòng dài bị cắt theo câu (và chunk sau được chèn tiêu đề phụ lục): xét từng câu
        missing = [l for l in lines if squash(l) not in joined
                   and not all(squash(s) in joined for s in re.split(r"(?<=[.;:])\s+", l) if s)]
        signature = [l for l in missing if DROPPED.search(l)]  # bảng nơi nhận / chữ ký: bỏ có chủ đích
        dropped_signature[0] += len(signature)
        missing = [l for l in missing if l not in signature]
        if missing:
            lost[f"{txt.parent.parent.name}/{txt.parent.name}"] = {
                "lines": len(missing), "chars": sum(map(len, missing)),
                "share": round(sum(map(len, missing)) / max(sum(map(len, lines)), 1), 4),
                "example": missing[0][:120]}
    return {
        "chunks": len(records), "retrievable_active": len(usable),
        "by_type": dict(Counter(r["chunk_type"] for r in usable)),
        "chars": {"p10": percentile(sizes, .1), "p50": percentile(sizes, .5), "p90": percentile(sizes, .9),
                  "max": sizes[-1] if sizes else 0},
        f"short_<{SHORT}": sum(s < SHORT for s in sizes), f"long_>{LONG}": sum(s > LONG for s in sizes),
        f"over_{max_tokens}_tokens": sum(t > max_tokens for t in embed_tokens),
        "duplicate_texts": sum(n - 1 for n in texts.values() if n > 1),
        "dropped_signature_lines": dropped_signature[0],
        "docs_losing_text": len(lost),
        "lost_text": dict(sorted(lost.items(), key=lambda kv: -kv[1]["chars"])[:15]),
    }


def oracle(questions: list[dict], texts: list[str], budget: int) -> list[dict]:
    """Trần recall: xếp chunk theo số 4-gram trùng với reference rồi lấy trong ngân sách ký tự."""
    chunk_grams = [grams(t) for t in texts]
    rows = []
    for q in questions:
        reference = set().union(*(grams(c) for c in q["contexts"]))
        ranked = sorted(range(len(texts)), key=lambda i: -len(chunk_grams[i] & reference))
        covered, used, best = set(), 0, 0.0
        for i in ranked:
            overlap = chunk_grams[i] & reference
            if not overlap or used >= budget:
                break
            best = max(best, len(overlap) / max(len(reference), 1))
            covered |= overlap
            used += len(texts[i])
        rows.append({"id": q["id"], "question": q["question"][:90],
                     "best_chunk_coverage": round(best, 3),
                     "oracle_recall": round(len(covered) / max(len(reference), 1), 3)})
    return rows


def main() -> None:
    config = Config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=Path, default=config.chunks_path.parent)
    parser.add_argument("--extracted", type=Path, default=ROOT / "data/data/processed_kb/clean_extracted")
    parser.add_argument("--testset", type=Path, default=config.testset)
    args = parser.parse_args()
    out = args.chunks / "eval"
    out.mkdir(exist_ok=True)
    budget = config.context_chars

    records = [json.loads(line) for line in (args.chunks / "chunks.jsonl").open(encoding="utf-8")]
    report = {"structure": structure(records, args.extracted, config.embedding_max_tokens)}

    questions = read_testset(args.testset)
    documents = LegalChunkLoader(args.chunks / "chunks.jsonl").load()
    texts = [d.metadata["text"] for d in documents]
    coverage = oracle(questions, texts, budget)
    report["testset_oracle"] = {
        "mean_oracle_recall": round(sum(r["oracle_recall"] for r in coverage) / len(coverage), 4),
        "questions_without_source(best<0.3)": [r["id"] for r in coverage if r["best_chunk_coverage"] < 0.3]}

    bm25 = BM25Retriever.from_documents(documents, preprocess_func=tokenize, k=config.depth, k1=1.2, b=0.75)
    ranked = [[d.metadata["text"] for d in bm25.invoke(q["question"])] for q in questions]
    metrics, details = score(ranked, questions, budget)
    report["bm25"] = metrics

    with (out / "per_question.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage[0]) + ["bm25_first_relevant_rank", "bm25_recall"])
        writer.writeheader()
        for row, detail in zip(coverage, details):
            writer.writerow({**row, "bm25_first_relevant_rank": detail["first_relevant_rank"],
                             "bm25_recall": detail["recall"]})
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
