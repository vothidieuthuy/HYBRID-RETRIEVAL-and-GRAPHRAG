"""Độ chính xác trích dẫn Điều: nhận định có nêu "Điều X văn bản Y" có đúng với nguyên văn Điều X trong KB không.

Dùng cho câu trả lời đã sinh (mặc định results_global, chế độ hybrid-rrf và graph-global). RAGAS faithfulness của
graph-global chỉ đối chiếu với báo cáo cộng đồng (do LLM viết), không với luật; phép đo này đối chiếu thẳng với luật.
1. Rút nhận định: LLM đọc câu trả lời, liệt kê các nhận định gắn với một Điều cụ thể của một văn bản; số hiệu văn bản
   quy về danh mục văn bản của KB; với hybrid, [n] được quy về Điều của đoạn nguồn thứ n.
2. Tra Điều: các chunk của KB cùng số hiệu văn bản và cùng số Điều (cả Điều của "Quy định kèm theo" nếu nhận định nói
   vậy), nối nguyên văn.
3. Chấm: LLM xét Điều đó có quy định nội dung của nhận định không: dung | sai; không tìm thấy Điều trong KB -> khong_co.
Kết quả: tỷ lệ đúng trên các nhận định có trích Điều, theo chế độ; so cặp bằng bootstrap theo câu hỏi.

    .venv-lc/Scripts/python -m lc_app.evaluate_citations [--dir results_global] [--judge-model gpt-4.1-mini]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI

from .config import ROOT, Config
from .loader import LegalChunkLoader

PRICE = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.0, 8.0), "gpt-4o-mini": (0.15, 0.60)}
ARTICLE_CHARS = 5000
EXTRACT = """Đọc CÂU TRẢ LỜI về pháp luật dưới đây và liệt kê các nhận định pháp lý được gắn với một Điều CỤ THỂ của một văn bản.
- Chỉ lấy nhận định có nêu số Điều (trực tiếp, ví dụ "theo Điều 79 Luật Đất đai", hoặc qua số nguồn [n] khi DANH SÁCH NGUỒN cho biết nguồn [n] là Điều nào).
- "van_ban": số hiệu văn bản lấy từ DANH MỤC VĂN BẢN (ví dụ "Luật Đất đai 2024" -> "31/2024/QH15"); không xác định được thì để "".
- "kem_theo": true nếu nhận định nói Điều của "Quy định kèm theo" một quyết định.
- "nhan_dinh": nội dung nhận định, viết lại ngắn gọn đủ ý (một câu).
- Mỗi cặp (Điều, văn bản) nêu nhiều ý thì tách thành các nhận định riêng; tối đa 15 nhận định.
Trả về duy nhất JSON: {{"claims": [{{"nhan_dinh": "...", "dieu": "79", "van_ban": "31/2024/QH15", "kem_theo": false}}]}}

DANH MỤC VĂN BẢN (số hiệu | tên):
{catalog}
{sources}
CÂU TRẢ LỜI:
{answer}"""
JUDGE = """Bạn là chuyên gia pháp luật đất đai, nhà ở Việt Nam. Xét NHẬN ĐỊNH dưới đây có được ĐIỀU LUẬT (nguyên văn) quy định không.
- "dung": nội dung chính của nhận định có trong Điều này (diễn đạt khác vẫn được).
- "sai": Điều này không quy định nội dung đó, hoặc quy định khác/ngược lại.
Trả về duy nhất JSON: {{"ket_qua": "dung|sai", "ly_do": "một câu"}}

NHẬN ĐỊNH: {claim}

ĐIỀU LUẬT ({ref}):
{text}"""


def bootstrap(values: list[float], seed: int = 0, rounds: int = 10000) -> tuple[float, float]:
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choice(values) for _ in values) / len(values) for _ in range(rounds))
    return means[int(rounds * 0.025)], means[int(rounds * 0.975)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=ROOT / "results_global")
    parser.add_argument("--mode", action="append", help="mặc định hybrid-rrf và graph-global")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    args = parser.parse_args()
    modes = args.mode or ["hybrid-rrf", "graph-global"]
    config = Config.from_env()
    set_llm_cache(SQLiteCache(database_path=str(args.dir / "citation_cache.sqlite")))
    llm = ChatOpenAI(model=args.judge_model, temperature=0, max_retries=8, timeout=180,
                     model_kwargs={"response_format": {"type": "json_object"}})

    articles: dict[tuple[str, str, bool], list[str]] = defaultdict(list)
    titles = {}
    for doc in LegalChunkLoader(config.chunks_path).load():
        m = doc.metadata
        titles[m["document_number"]] = m.get("doc_title") or ""
        if m.get("dieu"):
            articles[(m["document_number"], str(m["dieu"]), "::QD::" in doc.id)].append(m["text"])
    catalog = "\n".join(f"{k} | {v}" for k, v in sorted(titles.items()))

    answers = {m: [json.loads(line) for line in (args.dir / f"answers_{m}.jsonl").open(encoding="utf-8")] for m in modes}
    usage, claims = [], {m: [] for m in modes}
    for mode in modes:
        prompts = []
        for a in answers[mode]:
            sources = ""
            if a["contexts"] and "citation" in a["contexts"][0]:
                sources = "\nDANH SÁCH NGUỒN:\n" + "\n".join(f"[{i}] {c['citation']}"
                                                             for i, c in enumerate(a["contexts"], 1)) + "\n"
            prompts.append(EXTRACT.format(catalog=catalog, sources=sources, answer=a["answer"]))
        for a, reply in zip(answers[mode], llm.batch(prompts, config={"max_concurrency": 8})):
            usage.append(reply.usage_metadata)
            for c in (json.loads(reply.content).get("claims") or [])[:15]:
                dieu = re.sub(r"\D", "", str(c.get("dieu", "")))
                if dieu and c.get("nhan_dinh"):
                    claims[mode].append({"id": a["id"], "claim": c["nhan_dinh"], "dieu": dieu,
                                         "van_ban": str(c.get("van_ban") or ""), "kem_theo": bool(c.get("kem_theo"))})
        jobs = []
        for c in claims[mode]:
            texts = articles.get((c["van_ban"], c["dieu"], c["kem_theo"])) or articles.get((c["van_ban"], c["dieu"], not c["kem_theo"]))
            if not texts:
                c["ket_qua"], c["ly_do"] = "khong_co", "không có Điều này của văn bản trong KB"
                continue
            c["ref"] = f"Điều {c['dieu']}{' Quy định kèm theo' if c['kem_theo'] else ''} {c['van_ban']}"
            jobs.append((c, JUDGE.format(claim=c["claim"], ref=c["ref"], text="\n".join(texts)[:ARTICLE_CHARS])))
        for (c, _), reply in zip(jobs, llm.batch([p for _, p in jobs], config={"max_concurrency": 8})):
            usage.append(reply.usage_metadata)
            verdict = json.loads(reply.content)
            c["ket_qua"] = "dung" if str(verdict.get("ket_qua")).strip().lower() == "dung" else "sai"
            c["ly_do"] = verdict.get("ly_do", "")
        print(f"[{mode}] {len(claims[mode])} nhận định có trích Điều", flush=True)
    (args.dir / "citations.json").write_text(json.dumps(claims, ensure_ascii=False, indent=1), encoding="utf-8")

    ids = [a["id"] for a in answers[modes[0]]]
    lines = [f"# Độ chính xác trích dẫn Điều ({args.dir.name}, giám khảo {args.judge_model})", "",
             "Nhận định có nêu Điều cụ thể, đối chiếu với nguyên văn Điều trong KB. Đúng = Điều đó quy định nội dung "
             "nhận định; sai = Điều có trong KB nhưng không quy định nội dung đó; không có = Điều/văn bản không có "
             "trong KB.", "",
             "| Chế độ | Nhận định có trích Điều | TB mỗi câu | Đúng | Sai | Không có trong KB | Tỷ lệ đúng |",
             "|---|---|---|---|---|---|---|"]
    per_q = {}
    for mode in modes:
        cs = claims[mode]
        count = {k: sum(c["ket_qua"] == k for c in cs) for k in ("dung", "sai", "khong_co")}
        lines.append(f"| {mode} | {len(cs)} | {len(cs) / len(ids):.1f} | {count['dung']} | {count['sai']} | "
                     f"{count['khong_co']} | {count['dung'] / max(len(cs), 1):.3f} |")
        per_q[mode] = {}
        for qid in ids:
            mine = [c for c in cs if c["id"] == qid]
            if mine:
                per_q[mode][qid] = sum(c["ket_qua"] == "dung" for c in mine) / len(mine)
    if len(modes) == 2:
        both = [q for q in ids if q in per_q[modes[0]] and q in per_q[modes[1]]]
        diffs = [per_q[modes[1]][q] - per_q[modes[0]][q] for q in both]
        if diffs:
            low, high = bootstrap(diffs)
            lines.append(f"\nTỷ lệ đúng theo câu, {modes[1]} − {modes[0]} ({len(both)} câu cả hai có trích Điều): "
                         f"{sum(diffs) / len(diffs):+.3f} [KTC 95% {low:+.3f}, {high:+.3f}]")
    price = PRICE.get(args.judge_model, (0, 0))
    cost = sum(u["input_tokens"] * price[0] + u["output_tokens"] * price[1] for u in usage if u) / 1e6
    lines.append(f"\nChi phí ~${cost:.2f} (tính cả lần gọi lấy từ cache).")
    (args.dir / "citations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
