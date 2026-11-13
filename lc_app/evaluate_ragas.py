"""Đánh giá RAGAS (sinh câu trả lời + chấm bằng LLM) trên data/ragas_testset_50.csv cho từng chế độ truy hồi.

1. Sinh câu trả lời bằng chuỗi LangChain của lc_app.chain (cùng retriever, ngân sách ngữ cảnh, prompt, gpt-4o-mini).
2. Chấm bằng RAGAS qua LangChain (LangchainLLMWrapper(ChatOpenAI), LangchainEmbeddingsWrapper(Vietnamese_Embedding)):
   faithfulness, answer_relevancy (ResponseRelevancy), llm_context_precision_with_reference, context_recall.
   Prompt chấm của RAGAS viết bằng tiếng Anh: chuyển sang tiếng Việt một lần bằng adapt_prompts rồi lưu ở
   results/ragas_prompts_vi; các lần chạy sau nạp lại, mọi chế độ chấm bằng cùng một bộ prompt.
Cache LLM SQLite ở <out>/llm_cache.sqlite cho bước sinh câu trả lời: chạy lại cho cùng câu trả lời, không tốn tiền sinh
lại; bước chấm không cache (chấm lại mỗi lần chạy, ~$0,2 cho 50 câu một chế độ với gpt-4o-mini).

    .venv-lc/Scripts/python -m lc_app.evaluate_ragas [--mode dense --mode bm25 --mode hybrid-rrf] [--limit 3]
Đầu ra (results/ragas):
    answers_<mode>.jsonl  câu trả lời đầy đủ: câu hỏi, đáp án mẫu, câu trả lời, các chunk ngữ cảnh (id, nguồn, nội dung)
    answers.csv           mọi chế độ trong một bảng (mở bằng Excel)
    per_question.json     điểm từng câu của từng chế độ
    summary.csv/.md       điểm trung bình, KTC 95% (bootstrap), số câu chấm được; so sánh cặp với hybrid-rrf
Chạy một số chế độ thì kết quả của các chế độ đó được thay/thêm vào; kết quả đã có của chế độ khác giữ nguyên.
    run.json              mô hình sinh, mô hình chấm, token, chi phí ước tính
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import types
from pathlib import Path

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI

from .chain import build_chain
from .config import ROOT, Config
from .loader import citation
from .retrievers import BASE_MODES, MODES, Retrievers

# ragas 0.3.1 (bản cuối không cần instructor/openai<2) import ChatVertexAI từ đường dẫn mà langchain-community 0.4 đã
# bỏ, chỉ để kiểm tra isinstance: thay bằng một lớp rỗng trước khi import ragas
_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault(_vertexai.__name__, _vertexai)

from ragas import EvaluationDataset, RunConfig, evaluate  # noqa: E402
from ragas.cost import get_token_usage_for_openai  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,  # noqa: E402
                           ResponseRelevancy)

OUT = ROOT / "results/ragas"
PROMPTS = ROOT / "results/ragas_prompts_vi"
LANGUAGE = "vietnamese"
PRICE = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60), "gpt-4o": (2.50, 10.0)}   # USD / 1 triệu token


def read_testset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{"id": row["id"], "question": row["user_input"], "reference": row["reference"]}
                for row in csv.DictReader(handle)]


def generate(config: Config, retrievers: Retrievers, mode: str, questions: list[dict]) -> list[dict]:
    chain = build_chain(config, retrievers, mode)
    results = chain.batch([q["question"] for q in questions], config={"max_concurrency": 4})
    return [{"id": q["id"], "mode": mode, "question": q["question"], "reference": q["reference"],
             "answer": r["answer"],
             "contexts": [{"chunk_id": d.id, "citation": citation(d), "text": d.metadata["text"]}
                          for d in r["documents"]]}
            for q, r in zip(questions, results)]


def metrics(judge: LangchainLLMWrapper) -> list:
    """Bốn độ đo, prompt tiếng Việt: nạp từ prompts_vi nếu đã có, chưa có thì adapt_prompts rồi lưu."""
    chosen = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    for metric in chosen:
        folder = PROMPTS / metric.name   # chung cho mọi thư mục kết quả
        try:
            prompts = metric.load_prompts(str(folder), language=LANGUAGE)
        except (FileNotFoundError, ValueError):
            import asyncio
            prompts = asyncio.run(metric.adapt_prompts(language=LANGUAGE, llm=judge))
            folder.mkdir(parents=True, exist_ok=True)
            metric.set_prompts(**prompts)
            metric.save_prompts(str(folder))
        metric.set_prompts(**prompts)
    return chosen


def bootstrap(values: list[float], seed: int = 0, rounds: int = 10000) -> tuple[float, float]:
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choice(values) for _ in values) / len(values) for _ in range(rounds))
    return means[int(rounds * 0.025)], means[int(rounds * 0.975)]


def write_summary(out: Path, scores: dict, names: list[str], title: str) -> list[str]:
    """Ghi summary.csv/.md từ điểm từng câu của mọi chế độ trong scores; trả về các dòng markdown."""
    modes = list(scores)
    summary = []
    for mode in modes:
        line = {"mode": mode}
        for n in names:
            values = [s[n] for s in scores[mode] if s[n] is not None]
            low, high = bootstrap(values) if values else (math.nan, math.nan)
            line[n] = round(sum(values) / len(values), 4) if values else None
            line[f"{n}_ci95"] = f"[{low:.3f}, {high:.3f}]"
            line[f"{n}_n"] = len(values)
        summary.append(line)
    # so sánh cặp với hybrid-rrf (cùng câu hỏi, chỉ câu cả hai chế độ đều chấm được)
    pairs = []
    if "hybrid-rrf" in modes:
        for mode in (m for m in modes if m != "hybrid-rrf"):
            for n in names:
                diffs = [h[n] - o[n] for h, o in zip(scores["hybrid-rrf"], scores[mode])
                         if h[n] is not None and o[n] is not None]
                if diffs:
                    low, high = bootstrap(diffs)
                    pairs.append({"so_sanh": f"hybrid-rrf - {mode}", "metric": n, "n": len(diffs),
                                  "chenh_lech": round(sum(diffs) / len(diffs), 4), "ci95": f"[{low:+.3f}, {high:+.3f}]",
                                  "co_y_nghia": low > 0 or high < 0})
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    md = [f"# {title}\n", "| Chế độ | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    md += [f"| {s['mode']} | " + " | ".join(f"{s[n]:.3f} {s[n + '_ci95']}" if s[n] is not None else "–"
                                         for n in names) + " |" for s in summary]
    if pairs:
        md += ["\n## So sánh cặp (bootstrap 10.000 lần, KTC 95% không chứa 0 = có ý nghĩa)\n",
               "| So sánh | Độ đo | n | Chênh lệch | KTC 95% | Có ý nghĩa |", "|---|---|---|---|---|---|"]
        md += [f"| {p['so_sanh']} | {p['metric']} | {p['n']} | {p['chenh_lech']:+.3f} | {p['ci95']} | "
               f"{'có' if p['co_y_nghia'] else 'không'} |" for p in pairs]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="mô hình chấm của RAGAS")
    parser.add_argument("--limit", type=int, help="chỉ chạy N câu đầu (chạy thử)")
    parser.add_argument("--out", type=Path, default=OUT, help="thư mục kết quả")
    parser.add_argument("--testset", type=Path, help="bộ câu hỏi (mặc định config.testset = ragas_testset_50)")
    args = parser.parse_args()
    modes = args.mode or list(BASE_MODES)
    config = Config.from_env()
    args.out.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(args.out / "llm_cache.sqlite")))
    testset = args.testset or config.testset
    questions = read_testset(testset)[:args.limit]
    retrievers = Retrievers(config)
    # cache=False: RAGAS gọi cùng một prompt song song (answer_relevancy sinh nhiều câu hỏi ngược) -> cache SQLite lỗi
    # trùng khóa và làm các lần gọi giống hệt nhau; chỉ bước sinh câu trả lời dùng cache
    judge = LangchainLLMWrapper(ChatOpenAI(model=args.judge_model, temperature=0, max_retries=8, timeout=180,
                                           cache=False))
    embeddings = LangchainEmbeddingsWrapper(retrievers.store.embeddings)   # Vietnamese_Embedding như truy hồi dense
    chosen = metrics(judge)
    names = [m.name for m in chosen]
    print(f"{len(questions)} câu, chế độ {modes}, sinh {config.llm_model}, chấm {args.judge_model}, độ đo {names}",
          flush=True)

    # kết quả cũ của các chế độ không chạy lần này được giữ lại
    old = args.out / "per_question.json", args.out / "run.json", args.out / "answers.csv"
    scores = json.loads(old[0].read_text(encoding="utf-8")) if old[0].exists() else {}
    usage = json.loads(old[1].read_text(encoding="utf-8")).get("judge_tokens", {}) if old[1].exists() else {}
    rows = []
    if old[2].exists():
        with old[2].open(encoding="utf-8-sig", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["mode"] not in modes]
    start = time.time()
    for mode in modes:
        answers = generate(config, retrievers, mode, questions)
        with (args.out / f"answers_{mode}.jsonl").open("w", encoding="utf-8") as handle:
            for a in answers:
                handle.write(json.dumps(a, ensure_ascii=False) + "\n")
        print(f"[{mode}] đã sinh {len(answers)} câu trả lời ({(time.time() - start) / 60:.1f} phút)", flush=True)
        dataset = EvaluationDataset.from_list([
            {"user_input": a["question"], "response": a["answer"], "reference": a["reference"],
             "retrieved_contexts": [f"{c['citation']}\n{c['text']}" for c in a["contexts"]]} for a in answers])
        result = evaluate(dataset, metrics=chosen, llm=judge, embeddings=embeddings, show_progress=False,
                          run_config=RunConfig(max_workers=8, timeout=300, max_retries=8),
                          token_usage_parser=get_token_usage_for_openai)
        table = result.to_pandas()
        scores[mode] = [{"id": a["id"], **{n: (None if math.isnan(v) else round(float(v), 4))
                                           for n in names for v in [table.iloc[i][n]]}}
                        for i, a in enumerate(answers)]
        tokens = result.total_tokens()
        usage[mode] = {"input": tokens.input_tokens, "output": tokens.output_tokens}
        for a, s in zip(answers, scores[mode]):
            rows.append({"id": a["id"], "mode": mode, "question": a["question"], "reference": a["reference"],
                         "answer": a["answer"], **{n: s[n] for n in names},
                         "sources": "\n".join(f"{c['chunk_id']} | {c['citation']}" for c in a["contexts"])})
        (args.out / "per_question.json").write_text(json.dumps(scores, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{mode}] đã chấm ({(time.time() - start) / 60:.1f} phút): "
              + ", ".join(f"{n} {sum(v for s in scores[mode] if (v := s[n]) is not None) / max(1, sum(s[n] is not None for s in scores[mode])):.3f}"
                          for n in names), flush=True)

    with (args.out / "answers.csv").open("w", encoding="utf-8-sig", newline="") as handle:   # utf-8-sig: Excel đọc đúng dấu
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    modes = list(scores)   # mọi chế độ đã chấm (cũ + lần này)
    md = write_summary(args.out, scores, names, f"RAGAS – {len(questions)} câu {testset.stem}, sinh {config.llm_model}, "
                       f"chấm {args.judge_model}, prompt chấm tiếng Việt")

    price_in, price_out = PRICE.get(args.judge_model, (math.nan, math.nan))
    run = {"generator": config.llm_model, "judge": args.judge_model,
           "language": LANGUAGE, "metrics": names, "questions": len(questions), "modes": modes, "context_chars": config.context_chars,
           "chunks_path": str(config.chunks_path), "testset": str(testset), "judge_tokens": usage,
           "judge_cost_usd": round(sum(u["input"] * price_in + u["output"] * price_out for u in usage.values()) / 1e6, 3),
           "minutes": round((time.time() - start) / 60, 1)}
    (args.out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(md))
    print(f"chi phí chấm ~${run['judge_cost_usd']} (token đếm theo lần gọi gốc, kể cả khi lấy từ cache) -> {args.out}")


if __name__ == "__main__":
    main()
