"""Gắn nhãn cho bộ 24 câu tổng hợp (data/global_questions_24.csv) trên KB hiện tại: khía cạnh (nugget) và đoạn luật liên quan.

1. Tập ứng viên (pooling kiểu TREC): top-<depth> của mọi chế độ truy hồi của lc_app, cộng
   <per-doc> chunk gần câu hỏi nhất (cosine) trong mỗi văn bản của cột van_ban_lien_quan (lượt tìm có chủ đích: câu tổng
   hợp trải trên nhiều văn bản, top-depth có thể không chạm tới văn bản nào đó). Tập ứng viên lưu ở <work>/pool.json
   và được giữ cố định ở các lần chạy sau (chỉ dựng lại khi có --rebuild-pool): bản 04/11/2026 gộp từ 6 chế độ, gồm cả
   hai chế độ đồ thị đã bỏ sau đó (hybrid-graph, hybrid-graph-expand).
2. Khía cạnh (như nugget của TREC RAG 2024): <judge> đọc câu hỏi và các ứng viên, liệt kê các ý mà câu trả lời đầy đủ
   phải có; mỗi ý phải có căn cứ trong các đoạn văn. Lưu ở <work>/aspects_<judge>.json.
3. Chấm từng cặp (câu hỏi, chunk) bằng <judge>, temperature 0: 2 = tự trả lời được ít nhất một khía cạnh (phải nêu khía
   cạnh nào), 1 = cùng chủ đề nhưng chỉ là bối cảnh, 0 = không liên quan. Nhãn lưu ở <work>/labels_<judge>.jsonl; chạy
   lại chỉ chấm các cặp chưa có nhãn. Có nhãn của giám khảo khác (--compare) thì in độ đồng thuận (Cohen's kappa).
4. Ghi data/global_questions_24_labeled.csv: các cột của bộ câu hỏi, khia_canh (JSON: id, nội dung, chunk căn cứ),
   relevant_chunk_ids và reference_contexts (chunk điểm 2, cùng định dạng ragas_testset_50 để lc_app.evaluate_ir đọc
   được), pool_size, n_relevant, n_partial. Khía cạnh không có chunk điểm 2 nào bị bỏ (không trả lời được từ KB).
   Chunk ngoài tập gộp coi như không liên quan (giới hạn chuẩn của pooling).

    .venv-lc/Scripts/python -m lc_app.label_global --plan   # chỉ gộp ứng viên, đếm cặp, không gọi API
    .venv-lc/Scripts/python -m lc_app.label_global [--judge gpt-4.1] [--compare gpt-4o-mini]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import ROOT, Config
from .loader import citation
from .retrievers import MODES, Retrievers

SRC = ROOT / "data/global_questions_24.csv"
DST = ROOT / "data/global_questions_24_labeled.csv"
WORK = ROOT / "data/global24_labels"
PRICE = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.0, 8.0), "gpt-4o": (2.5, 10.0)}
ASPECTS = ChatPromptTemplate.from_messages([("human", """Bạn là chuyên gia pháp luật đất đai, nhà ở Việt Nam, đang xây dựng bộ dữ liệu đánh giá hệ thống hỏi đáp.
Câu hỏi dưới đây là câu hỏi tổng hợp. Hãy liệt kê các KHÍA CẠNH mà một câu trả lời đầy đủ phải đề cập.
- Mỗi khía cạnh là một ý ngắn (một câu), cụ thể, kiểm chứng được, ví dụ "Các trường hợp thu hồi đất vì mục đích quốc phòng, an ninh".
- Chỉ nêu khía cạnh mà câu hỏi yêu cầu VÀ có căn cứ trong các đoạn văn bên dưới; ghi số thứ tự các đoạn căn cứ.
- Không nêu khía cạnh suy đoán, không gộp nhiều ý vào một khía cạnh, không tách một ý thành nhiều khía cạnh trùng nhau.
- Thường có 4–10 khía cạnh.
Trả về JSON: {{"khia_canh": [{{"id": "A1", "noi_dung": "...", "doan": [1, 5]}}]}}

CÂU HỎI: {question}

CÁC ĐOẠN VĂN:
{passages}""")])
GRADE = ChatPromptTemplate.from_messages([("human", """Bạn là chuyên gia pháp luật đất đai, nhà ở Việt Nam, đang gắn nhãn dữ liệu đánh giá hệ thống tìm kiếm.
Câu hỏi là câu hỏi tổng hợp; câu trả lời đầy đủ cần các khía cạnh liệt kê bên dưới.
Hãy đánh giá ĐOẠN VĂN với CÂU HỎI, chỉ dựa vào nội dung đoạn văn, theo thang chặt:
- 2: chính đoạn văn nêu quy định (trường hợp, điều kiện, mức, thủ tục, thẩm quyền...) trả lời trực tiếp ít nhất một khía cạnh; phải ghi các khía cạnh đó.
- 1: cùng chủ đề nhưng chỉ là bối cảnh: định nghĩa chung, dẫn chiếu sang quy định khác, quy định cho đối tượng/trường hợp hẹp không phải trọng tâm câu hỏi, hoặc chỉ nhắc tới chủ đề.
- 0: không liên quan.
Trả về JSON: {{"diem": 0|1|2, "khia_canh": ["A1", ...], "ly_do": "một câu ngắn"}}

CÂU HỎI: {question}

KHÍA CẠNH:
{aspects}

ĐOẠN VĂN ({citation}):
{text}""")])


def read_questions() -> tuple[list[str], list[dict]]:
    with SRC.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames), list(reader)


def build_pool(retrievers: Retrievers, questions: list[dict], depth: int, per_doc: int) -> dict:
    """{id câu hỏi: {chunk_id: [nguồn đưa chunk vào tập gộp]}}"""
    pool = {q["id"]: {} for q in questions}
    for mode in MODES:
        retriever = retrievers.get(mode)
        for q in questions:
            for doc in retriever.invoke(q["user_input"])[:depth]:
                pool[q["id"]].setdefault(doc.id, []).append(mode)
        print(f"[{mode}] xong", flush=True)
    size = retrievers.store.index.ntotal
    for q in questions:
        for number in (v.strip().split()[-1] for v in q["van_ban_lien_quan"].split(";") if v.strip()):
            docs = retrievers.store.similarity_search(q["user_input"], k=per_doc, fetch_k=size,
                                                      filter={"document_number": number})
            if not docs:
                print(f"{q['id']}: không có chunk nào của {number} trong KB", flush=True)
            for doc in docs:
                pool[q["id"]].setdefault(doc.id, []).append("van_ban_lien_quan")
    return pool


def read_labels(path: Path) -> dict:
    if not path.exists():
        return {}
    return {(r["id"], r["chunk_id"]): r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def cost(records, model: str) -> float:
    tokens = [r["tokens"] for r in records if r.get("tokens")]
    price = PRICE.get(model, (0.0, 0.0))
    return (sum(t["input_tokens"] for t in tokens) * price[0] + sum(t["output_tokens"] for t in tokens) * price[1]) / 1e6


def kappa(pairs: list[tuple[int, int]]) -> float:
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[k] * second[k] for k in first) / n / n
    return (observed - expected) / (1 - expected)


def make_aspects(llm, questions: list[dict], pool: dict, documents: dict, path: Path) -> dict:
    aspects = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [q for q in questions if q["id"] not in aspects]
    inputs = []
    for q in todo:
        ids = sorted(pool[q["id"]])
        passages = "\n\n".join(f"[{i}] ({citation(documents[c])}) {documents[c].metadata['text'][:700]}"
                               for i, c in enumerate(ids, 1))
        inputs.append({"question": q["user_input"], "passages": passages})
    for q, reply in zip(todo, (ASPECTS | llm).batch(inputs, config={"max_concurrency": 2})):
        ids = sorted(pool[q["id"]])
        items = json.loads(reply.content)["khia_canh"]
        aspects[q["id"]] = {"khia_canh": [{"id": a["id"], "noi_dung": a["noi_dung"],
                                           "doan": [ids[i - 1] for i in a.get("doan", []) if 0 < i <= len(ids)]}
                                          for a in items],
                            "tokens": reply.usage_metadata}
        path.write_text(json.dumps(aspects, ensure_ascii=False, indent=1), encoding="utf-8")
    return aspects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", action="store_true", help="chỉ gộp ứng viên và đếm cặp")
    parser.add_argument("--rebuild-pool", action="store_true", help="dựng lại tập ứng viên thay vì dùng pool.json")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--per-doc", type=int, default=3, help="số chunk thêm cho mỗi văn bản trong van_ban_lien_quan")
    parser.add_argument("--judge", default="gpt-4.1", help="mô hình viết khía cạnh và chấm")
    parser.add_argument("--compare", default="gpt-4o-mini", help="giám khảo khác để tính độ đồng thuận")
    args = parser.parse_args()
    config = Config.from_env()
    fields, questions = read_questions()
    retrievers = Retrievers(config)
    documents = {doc.id: doc for doc in retrievers.bm25.docs}
    WORK.mkdir(parents=True, exist_ok=True)
    if args.rebuild_pool or not (WORK / "pool.json").exists():
        pool = build_pool(retrievers, questions, args.depth, args.per_doc)
        (WORK / "pool.json").write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        pool = json.loads((WORK / "pool.json").read_text(encoding="utf-8"))
    labels_path = WORK / f"labels_{args.judge}.jsonl"
    labels = read_labels(labels_path)
    todo = [(q, cid) for q in questions for cid in pool[q["id"]] if (q["id"], cid) not in labels]
    print(f"{sum(map(len, pool.values()))} cặp trong tập gộp, {len(todo)} cặp chưa chấm bằng {args.judge}", flush=True)
    if args.plan:
        return

    llm = ChatOpenAI(model=args.judge, temperature=0, max_retries=8, timeout=180,
                     model_kwargs={"response_format": {"type": "json_object"}})
    aspects = make_aspects(llm, questions, pool, documents, WORK / f"aspects_{args.judge}.json")
    listing = {qid: "\n".join(f"{a['id']}. {a['noi_dung']}" for a in v["khia_canh"]) for qid, v in aspects.items()}
    with labels_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), 64):
            batch = todo[start:start + 64]
            replies = (GRADE | llm).batch([{"question": q["user_input"], "aspects": listing[q["id"]],
                                            "citation": citation(documents[cid]),
                                            "text": documents[cid].metadata["text"]} for q, cid in batch],
                                          config={"max_concurrency": 6})
            for (q, cid), reply in zip(batch, replies):
                parsed = json.loads(reply.content)
                known = {a["id"] for a in aspects[q["id"]]["khia_canh"]}
                covered = [a for a in parsed.get("khia_canh") or [] if a in known]
                score = int(parsed["diem"])
                if score == 2 and not covered:   # điểm 2 phải chỉ ra được khía cạnh
                    score = 1
                record = {"id": q["id"], "chunk_id": cid, "diem": score, "khia_canh": covered if score == 2 else [],
                          "ly_do": parsed.get("ly_do", ""), "tokens": reply.usage_metadata}
                labels[(q["id"], cid)] = record
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"đã chấm {start + len(batch)}/{len(todo)}", flush=True)

    with DST.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["khia_canh", "relevant_chunk_ids", "reference_contexts",
                                                             "pool_size", "n_relevant", "n_partial"])
        writer.writeheader()
        dropped = 0
        for q in questions:
            scored = [labels[(q["id"], cid)] for cid in sorted(pool[q["id"]])]
            relevant = [r["chunk_id"] for r in scored if r["diem"] == 2]
            support = {a["id"]: [r["chunk_id"] for r in scored if a["id"] in r["khia_canh"]]
                       for a in aspects[q["id"]]["khia_canh"]}
            kept = [{"id": a["id"], "noi_dung": a["noi_dung"], "chunk_ids": support[a["id"]]}
                    for a in aspects[q["id"]]["khia_canh"] if support[a["id"]]]
            dropped += len(aspects[q["id"]]["khia_canh"]) - len(kept)
            writer.writerow({**q, "khia_canh": json.dumps(kept, ensure_ascii=False),
                             "relevant_chunk_ids": json.dumps(relevant, ensure_ascii=False),
                             "reference_contexts": json.dumps([documents[c].metadata["text"] for c in relevant],
                                                              ensure_ascii=False),
                             "pool_size": len(scored), "n_relevant": len(relevant),
                             "n_partial": sum(r["diem"] == 1 for r in scored)})
    total = cost(labels.values(), args.judge) + cost(aspects.values(), args.judge)
    print(f"ghi {DST.name}; bỏ {dropped} khía cạnh không có chunk căn cứ; chi phí {args.judge} (cộng dồn) ${total:.2f}")
    other = read_labels(WORK / f"labels_{args.compare}.jsonl")
    pairs = [(other[k]["diem"], labels[k]["diem"]) for k in labels if k in other and k[1] in pool[k[0]]]
    if pairs:
        binary = [(a == 2, b == 2) for a, b in pairs]
        print(f"đồng thuận {args.compare} vs {args.judge} trên {len(pairs)} cặp: "
              f"kappa 3 mức {kappa(pairs):.3f}, kappa liên quan/không (điểm 2) {kappa(binary):.3f}; "
              f"ma trận {sorted(Counter(pairs).items())}")


if __name__ == "__main__":
    main()
