"""Đánh giá câu hỏi tổng hợp (data/global_questions_24_labeled.csv, không có đáp án mẫu): graph-global so với hybrid-rrf.

1. Sinh câu trả lời: hybrid-rrf (và các chế độ truy hồi khác) qua lc_app.chain; graph-global qua
   lc_app.graph.global_search (map-reduce trên báo cáo cộng đồng). Cùng mô hình sinh (config.llm_model), cùng ngân sách
   ngữ cảnh cuối 8.000 ký tự (chunk với hybrid, các ý của bước map với graph-global).
2. So cặp như bài báo GraphRAG (Edge và cộng sự, 2024): bốn tiêu chí đầy đủ, đa dạng, giúp hiểu và hành động, trực tiếp;
   chấm hai lượt đảo thứ tự, một bên chỉ thắng khi thắng cả hai lượt; kiểm định dấu (bỏ hòa).
3. Độ phủ khía cạnh (như nugget recall của TREC RAG 2024): tỷ lệ khía cạnh của cột khia_canh mà câu trả lời nêu được;
   so cặp bằng bootstrap.
Giám khảo gpt-4.1 (khác mô hình sinh). Cache LLM SQLite ở <out>/llm_cache.sqlite: chạy lại không tốn tiền.

    .venv-lc/Scripts/python -m lc_app.evaluate_global [--mode hybrid-rrf --mode graph-global] [--limit 2]
Đầu ra (results_global): answers_<mode>.jsonl, answers.csv, judge.json, coverage.json, summary.md, run.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from math import comb
from pathlib import Path

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI

from .chain import build_chain
from .config import ROOT, Config
from .loader import citation
from .retrievers import MODES, Retrievers

OUT = ROOT / "results_global"
TESTSET = ROOT / "data/global_questions_24_labeled.csv"
BASELINE = "hybrid-rrf"
PRICE = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1": (2.0, 8.0), "gpt-4.1-mini": (0.40, 1.60), "gpt-4o": (2.5, 10.0)}
CRITERIA = {
    "comprehensiveness": "Đầy đủ: câu trả lời bao quát được bao nhiêu khía cạnh và chi tiết của câu hỏi?",
    "diversity": "Đa dạng: câu trả lời nêu được bao nhiêu góc nhìn, nhóm trường hợp hoặc văn bản khác nhau?",
    "empowerment": "Giúp hiểu và hành động: câu trả lời giúp người đọc hiểu vấn đề và biết cần làm gì, căn cứ ở đâu?",
    "directness": "Trực tiếp: câu trả lời đi thẳng vào câu hỏi, rõ ràng, không lan man?",
}
JUDGE = """Bạn là giám khảo độc lập, đánh giá hai câu trả lời về pháp luật đất đai, nhà ở Việt Nam cho cùng một câu hỏi.

Với mỗi tiêu chí dưới đây, chọn câu trả lời tốt hơn: "1", "2" hoặc "tie" (ngang nhau).
{criteria}

Lưu ý:
- Độ dài không phải là ưu điểm: câu trả lời dài hơn không mặc nhiên tốt hơn; thông tin thừa hoặc lặp lại không được cộng điểm.
- Câu trả lời từ chối hoặc nói "chưa đủ thông tin" thì kém hơn câu trả lời có nội dung phù hợp.
- Không đánh giá văn phong, định dạng markdown hay cách ghi nguồn ([1], [Báo cáo: ...]).

Trả về duy nhất JSON: {{"comprehensiveness": "1|2|tie", "diversity": "1|2|tie", "empowerment": "1|2|tie", "directness": "1|2|tie", "reason": "một câu"}}

CÂU HỎI: {question}

=== CÂU TRẢ LỜI 1 ===
{first}

=== CÂU TRẢ LỜI 2 ===
{second}"""
COVERAGE = """Bạn là giám khảo đánh giá câu trả lời về pháp luật đất đai, nhà ở Việt Nam.
Dưới đây là câu hỏi, danh sách KHÍA CẠNH mà câu trả lời đầy đủ cần nêu, và một câu trả lời.
Với từng khía cạnh, xác định câu trả lời có nêu nội dung của khía cạnh đó không (nêu đúng ý chính là đủ, không cần đúng từng chữ; chỉ nhắc tên chủ đề mà không có nội dung thì không tính).
Trả về duy nhất JSON: {{"covered": ["A1", ...]}}

CÂU HỎI: {question}

KHÍA CẠNH:
{aspects}

CÂU TRẢ LỜI:
{answer}"""


def read_testset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{"id": r["id"], "question": r["user_input"], "aspects": json.loads(r["khia_canh"])}
                for r in csv.DictReader(handle)]


def generate(config: Config, mode: str, questions: list[dict], retrievers: Retrievers | None, level: int) -> list[dict]:
    if mode == "graph-global":
        from .graph.global_search import GlobalSearch
        search = GlobalSearch(config, level=level)
        rows = []
        for q in questions:
            r = search.invoke(q["question"])
            rows.append({"id": q["id"], "mode": mode, "question": q["question"], "answer": r["answer"],
                         "contexts": [{"report": i, "title": search.by_id[i]["title"],
                                       "chunk_ids": search.by_id[i]["chunk_ids"]} for i in r["reports"]],
                         "points": r["points"], "map_calls": r["map_calls"], "usage": r["usage"]})
            print(f"[{mode}] {q['id']}: {r['map_calls']} lô map, {len(r['points'])} ý, {len(r['reports'])} báo cáo",
                  flush=True)
        return rows
    results = build_chain(config, retrievers, mode).batch([q["question"] for q in questions],
                                                          config={"max_concurrency": 4})
    return [{"id": q["id"], "mode": mode, "question": q["question"], "answer": r["answer"],
             "contexts": [{"chunk_id": d.id, "citation": citation(d)} for d in r["documents"]]}
            for q, r in zip(questions, results)]


def sign_test(a: int, b: int) -> float:
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n)


def bootstrap_diff(diffs: list[float], seed: int = 0, rounds: int = 10000) -> tuple[float, float]:
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(rounds))
    return means[int(rounds * 0.025)], means[int(rounds * 0.975)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", action="append", choices=list(MODES) + ["graph-global"])
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--level", type=int, default=1, help="cấp cộng đồng của graph-global")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    modes = args.mode or [BASELINE, "graph-global"]
    if BASELINE not in modes:
        modes.insert(0, BASELINE)
    config = Config.from_env()
    args.out.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(args.out / "llm_cache.sqlite")))
    questions = read_testset(TESTSET)[:args.limit]
    retrievers = Retrievers(config) if any(m != "graph-global" for m in modes) else None

    answers = {}
    for mode in modes:
        answers[mode] = generate(config, mode, questions, retrievers, args.level)
        with (args.out / f"answers_{mode}.jsonl").open("w", encoding="utf-8") as handle:
            for a in answers[mode]:
                handle.write(json.dumps(a, ensure_ascii=False) + "\n")
    with (args.out / "answers.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "question"] + modes)
        for i, q in enumerate(questions):
            writer.writerow([q["id"], q["question"]] + [answers[m][i]["answer"] for m in modes])

    judge = ChatOpenAI(model=args.judge_model, temperature=0, max_retries=8, timeout=180,
                       model_kwargs={"response_format": {"type": "json_object"}})
    criteria = "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    usage = []

    # độ phủ khía cạnh
    jobs = [(m, i) for m in modes for i in range(len(questions))]
    replies = judge.batch([COVERAGE.format(question=questions[i]["question"], answer=answers[m][i]["answer"],
                                           aspects="\n".join(f"{a['id']}. {a['noi_dung']}" for a in questions[i]["aspects"]))
                           for m, i in jobs], config={"max_concurrency": 6})
    coverage = {m: {} for m in modes}
    for (m, i), reply in zip(jobs, replies):
        usage.append(reply.usage_metadata)
        known = [a["id"] for a in questions[i]["aspects"]]
        covered = [a for a in json.loads(reply.content).get("covered", []) if a in known]
        coverage[m][questions[i]["id"]] = {"covered": covered, "total": len(known), "score": len(covered) / len(known)}
    (args.out / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=1), encoding="utf-8")

    # so cặp với hybrid-rrf, hai lượt đảo thứ tự
    pairs = {}
    for mode in (m for m in modes if m != BASELINE):
        jobs = [(i, order) for i in range(len(questions)) for order in ("forward", "swapped")]
        prompts = []
        for i, order in jobs:
            a, b = answers[mode][i]["answer"], answers[BASELINE][i]["answer"]   # A = chế độ, B = hybrid-rrf
            first, second = (a, b) if order == "forward" else (b, a)
            prompts.append(JUDGE.format(criteria=criteria, question=questions[i]["question"], first=first, second=second))
        replies = judge.batch(prompts, config={"max_concurrency": 6})
        verdicts = {}
        for (i, order), reply in zip(jobs, replies):
            usage.append(reply.usage_metadata)
            verdicts.setdefault(questions[i]["id"], {})[order] = json.loads(reply.content)
        per_question = {}
        for qid, v in verdicts.items():
            result = {}
            for key in CRITERIA:
                first = {"1": "A", "2": "B"}.get(str(v["forward"].get(key)).strip().lower(), "tie")
                second = {"1": "B", "2": "A"}.get(str(v["swapped"].get(key)).strip().lower(), "tie")
                result[key] = first if first == second else "tie"
            per_question[qid] = {"result": result, **v}
        pairs[mode] = per_question
    (args.out / "judge.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"# Câu hỏi tổng hợp ({TESTSET.name}, {len(questions)} câu)", "",
             f"Sinh: {config.llm_model}; giám khảo: {args.judge_model}; graph-global cấp cộng đồng {args.level}.", "",
             "## Độ phủ khía cạnh và độ dài", "", "| Chế độ | Độ phủ khía cạnh | Độ dài TB (ký tự) |", "|---|---|---|"]
    for m in modes:
        scores = [coverage[m][q["id"]]["score"] for q in questions]
        length = sum(len(a["answer"]) for a in answers[m]) / len(questions)
        lines.append(f"| {m} | {sum(scores) / len(scores):.3f} | {length:.0f} |")
    for m in (m for m in modes if m != BASELINE):
        diffs = [coverage[m][q["id"]]["score"] - coverage[BASELINE][q["id"]]["score"] for q in questions]
        low, high = bootstrap_diff(diffs)
        lines.append(f"\n{m} − {BASELINE}: {sum(diffs) / len(diffs):+.3f} [KTC 95% {low:+.3f}, {high:+.3f}]")
    for m, per_question in pairs.items():
        lines += ["", f"## So cặp: A = {m}, B = {BASELINE}", "",
                  "| Tiêu chí | A thắng | B thắng | Hòa | p (kiểm định dấu) |", "|---|---|---|---|---|"]
        for key in CRITERIA:
            a = sum(r["result"][key] == "A" for r in per_question.values())
            b = sum(r["result"][key] == "B" for r in per_question.values())
            lines.append(f"| {key} | {a} | {b} | {len(per_question) - a - b} | {sign_test(a, b):.4f} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    gen = [u for m in modes if m == "graph-global" for a in answers[m] for u in a.get("usage", []) if u]
    price = PRICE.get(args.judge_model, (0, 0))
    cost = {"judge_usd": round(sum(u["input_tokens"] * price[0] + u["output_tokens"] * price[1] for u in usage if u) / 1e6, 3),
            "graph_global_generation_usd": round(sum(u["input_tokens"] * 0.15 + u["output_tokens"] * 0.6 for u in gen) / 1e6, 3)}
    (args.out / "run.json").write_text(json.dumps({"modes": modes, "llm": config.llm_model, "judge": args.judge_model,
                                                   "level": args.level, "questions": len(questions), "cost": cost},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(cost))


if __name__ == "__main__":
    main()
