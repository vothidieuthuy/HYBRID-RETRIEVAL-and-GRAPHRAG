"""Đánh giá truy hồi (IR, không gọi LLM) trên data/ragas_testset_50.csv, cùng định nghĩa với bản đánh giá cũ.

Chunk liên quan: >= 50% 4-gram âm tiết của chunk nằm trong reference_contexts, hoặc chunk phủ >= 30% 4-gram của
reference_contexts. Độ đo: Hit@1/5/10, MRR@10, Recall@<ngân sách> = tỷ lệ 4-gram của reference_contexts được phủ bởi
các chunk đầu danh sách cho tới khi đủ ngân sách ký tự.

    .venv-lc/Scripts/python -m lc_app.evaluate_ir [--mode dense --mode bm25 --mode hybrid-rrf --mode graph-augmented-hybrid]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .config import ROOT, Config
from .retrievers import BASE_MODES, MODES, Retrievers
from .vietnamese import WORD, normalize


def grams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    words = WORD.findall(normalize(text))
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def read_testset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{"id": row["id"], "question": row["user_input"], "contexts": json.loads(row["reference_contexts"])}
                for row in csv.DictReader(handle)]


def score(ranked: list[list[str]], questions: list[dict], budget: int) -> tuple[dict, list[dict]]:
    """ranked[i] = nội dung các chunk (metadata["text"]) theo thứ tự trả về cho câu hỏi i."""
    totals = dict.fromkeys(("hit@1", "hit@5", "hit@10", "mrr@10", f"recall@{budget}"), 0.0)
    details = []
    for question, texts in zip(questions, ranked):
        reference = set().union(*(grams(c) for c in question["contexts"]))
        relevant, covered, used = [], set(), 0
        for text in texts:
            chunk = grams(text)
            overlap = chunk & reference
            relevant.append(bool(chunk) and (len(overlap) / len(chunk) >= 0.5
                                             or len(overlap) / max(len(reference), 1) >= 0.3))
            if used < budget:
                covered |= overlap
                used += len(text)
        first = next((rank for rank, flag in enumerate(relevant[:10], 1) if flag), None)
        recall = len(covered) / max(len(reference), 1)
        totals["hit@1"] += bool(relevant[:1] and relevant[0])
        totals["hit@5"] += any(relevant[:5])
        totals["hit@10"] += any(relevant[:10])
        totals["mrr@10"] += 1 / first if first else 0.0
        totals[f"recall@{budget}"] += recall
        details.append({"id": question["id"], "first_relevant_rank": first, "recall": round(recall, 4)})
    return {k: round(v / len(questions), 4) for k, v in totals.items()}, details


def main() -> None:
    config = Config.from_env()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--testset", type=Path, default=config.testset)
    parser.add_argument("--out", type=Path, default=ROOT / "results/ir")
    args = parser.parse_args()
    questions = read_testset(args.testset)
    retrievers = Retrievers(config)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, all_details = [], {}
    for mode in args.mode or BASE_MODES:   # graph-augmented-hybrid (cần Neo4j) chỉ chạy khi chỉ định --mode
        retriever = retrievers.get(mode)
        started = time.perf_counter()
        docs = [retriever.invoke(q["question"])[:config.depth] for q in questions]
        ranked = [[d.metadata["text"] for d in top] for top in docs]
        seconds = (time.perf_counter() - started) / len(questions)
        metrics, details = score(ranked, questions, config.context_chars)
        rows.append({"mode": f"langchain/{mode}", "embedding": "-" if mode == "bm25" else config.embedding_model,
                     **metrics, "sec_per_query": round(seconds, 3)})
        for detail, top in zip(details, docs):
            detail["top"] = [d.id for d in top[:10]]
        all_details[mode] = details
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        # ghi sau mỗi chế độ: lỗi ở chế độ sau không làm mất kết quả đã có
        (args.out / "per_question.json").write_text(json.dumps(all_details, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    fields = list(rows[0])
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    lines += ["| " + " | ".join(str(row[f]) for f in fields) + " |" for row in rows]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = args.out / "summary.csv"
    try:
        handle = summary.open("w", encoding="utf-8-sig", newline="")
    except PermissionError:   # file đang mở trong Excel
        summary = summary.with_name(f"summary_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        print(f"summary.csv đang bị khóa, ghi sang {summary.name}")
        handle = summary.open("w", encoding="utf-8-sig", newline="")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
