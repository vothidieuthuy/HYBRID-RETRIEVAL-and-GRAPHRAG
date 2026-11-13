"""Global search của Microsoft GraphRAG (map-reduce trên báo cáo cộng đồng), cho câu hỏi tổng hợp.

Như bản gốc (chế độ tĩnh): lấy mọi báo cáo ở cấp `level`, cộng báo cáo ở cấp thô hơn của các cộng đồng không bị chia
nhỏ tới cấp đó; xáo trộn (seed cố định) rồi chia lô theo ngân sách ký tự. Map: mỗi lô, LLM rút các ý trả lời câu hỏi kèm
điểm quan trọng 0–100 và id báo cáo căn cứ. Reduce: bỏ ý điểm 0, xếp giảm dần, lấy tới hết `reduce_chars` ký tự
(mặc định 8.000 = ngân sách ngữ cảnh của hybrid-rrf), LLM viết câu trả lời cuối.
Kết quả kèm các báo cáo được dẫn (chunk_ids nguồn) để kiểm tra lại với đoạn luật.

    .venv-lc/Scripts/python -m lc_app.graph.global_search "Câu hỏi tổng hợp ..." [--level 1]
"""
from __future__ import annotations

import argparse
import json
import random
import re

from langchain_openai import ChatOpenAI

from ..config import Config

MAP_PROMPT = """Bạn là trợ lý trả lời câu hỏi về pháp luật đất đai, nhà ở Việt Nam dựa trên các báo cáo cộng đồng dưới đây.

Liệt kê các ý trả lời câu hỏi, chỉ dùng thông tin trong các báo cáo. Mỗi ý:
- "description": nội dung ý, nêu cụ thể quy định, điều kiện, trường hợp và Điều/văn bản căn cứ như trong báo cáo; cuối ý ghi [Báo cáo: id, id].
- "score": số nguyên 0–100, mức quan trọng của ý đối với câu hỏi.
Báo cáo không liên quan tới câu hỏi thì bỏ qua; không có ý nào thì trả về danh sách rỗng. Không bịa. Giữ nguyên nghĩa các từ tình thái (phải, được, có thể, không được).
Trả về duy nhất JSON: {{"points": [{{"description": "...", "score": 80}}]}}

CÂU HỎI: {question}

BÁO CÁO (id|tiêu đề|nội dung):
{reports}"""
REDUCE_PROMPT = """Bạn là trợ lý hỏi đáp pháp luật đất đai bằng tiếng Việt. Dưới đây là các ý do nhiều nhà phân tích rút ra từ đồ thị tri thức pháp luật, xếp theo mức quan trọng giảm dần.
Tổng hợp thành MỘT câu trả lời cho câu hỏi:
- Câu đầu tiên trả lời thẳng vào câu hỏi; câu hỏi có nhiều ý hoặc nhiều nhóm trường hợp thì trả lời đủ từng ý, có tiêu đề mục khi phù hợp.
- Chỉ dùng thông tin trong các ý; gộp ý trùng, bỏ ý không liên quan; không nhắc tới "nhà phân tích".
- Giữ dẫn nguồn [Báo cáo: ...] cho từng nhận định và tên Điều/văn bản căn cứ đã có trong các ý; không tự tạo điều luật.
- Ý nào không đủ căn cứ thì nói rõ "Chưa đủ thông tin trong nguồn truy xuất"; không suy đoán.
- Giữ nguyên nghĩa các từ tình thái (phải, được, có thể, không được). Tối đa {max_words} từ.
{style}
CÁC Ý:
{points}

CÂU HỎI: {question}"""
NO_DATA = "Chưa đủ thông tin trong nguồn truy xuất."
CITE = re.compile(r"\[Báo cáo:([^\]]*)\]")


def report_text(report: dict, limit: int) -> str:
    findings = " ".join(f"{f['summary']}: {f['explanation']}" for f in report["findings"])
    return f"{report['summary']} {findings}"[:limit]


class GlobalSearch:
    def __init__(self, config: Config, level: int = 1, batch_chars: int = 12000, report_chars: int = 3000,
                 reduce_chars: int | None = None, max_words: int = 600, seed: int = 42, concurrency: int = 8):
        communities = {c["id"]: c for c in json.loads((config.graph_dir / "communities.json").read_text(encoding="utf-8"))}
        reports = {r["community"]: r for r in json.loads((config.graph_dir / "reports.json").read_text(encoding="utf-8"))}
        # cấp `level` + cộng đồng ở cấp thô hơn không có con (không bị chia tới cấp level)
        self.reports = [r for r in reports.values()
                        if r["level"] == level or (r["level"] < level and not communities[r["community"]]["children"])]
        random.Random(seed).shuffle(self.reports)
        self.by_id = {r["community"]: r for r in self.reports}
        self.batch_chars, self.report_chars, self.max_words = batch_chars, report_chars, max_words
        self.reduce_chars = reduce_chars or config.context_chars
        self.concurrency = concurrency
        self.map_llm = ChatOpenAI(model=config.llm_model, temperature=0, max_retries=8,
                                  model_kwargs={"response_format": {"type": "json_object"}})
        self.reduce_llm = ChatOpenAI(model=config.llm_model, temperature=0, max_retries=8)

    def batches(self) -> list[str]:
        out, current, size = [], [], 0
        for r in self.reports:
            line = f"{r['community']}|{r['title']}|{report_text(r, self.report_chars)}"
            if current and size + len(line) > self.batch_chars:
                out.append("\n".join(current))
                current, size = [], 0
            current.append(line)
            size += len(line) + 1
        return out + (["\n".join(current)] if current else [])

    def invoke(self, question: str, style: str = "") -> dict:
        """style: hướng dẫn trình bày thêm cho bước reduce (vd theo đối tượng người dùng), mặc định không có."""
        replies = self.map_llm.batch([MAP_PROMPT.format(question=question, reports=b) for b in self.batches()],
                                     config={"max_concurrency": self.concurrency})
        points = []
        for reply in replies:
            try:
                items = json.loads(reply.content).get("points") or []
            except json.JSONDecodeError:
                continue
            for item in items:
                try:
                    point = {"description": str(item["description"]), "score": int(item["score"])}
                except (KeyError, TypeError, ValueError):
                    continue
                if point["score"] > 0:
                    points.append(point)
        points.sort(key=lambda p: -p["score"])
        kept, size = [], 0
        for p in points:
            if size + len(p["description"]) > self.reduce_chars:
                break
            kept.append(p)
            size += len(p["description"])
        usage = [r.usage_metadata for r in replies if r.usage_metadata]
        if not kept:
            return {"answer": NO_DATA, "points": [], "reports": [], "map_calls": len(replies), "usage": usage}
        text = "\n\n".join(f"--- Ý {i} (quan trọng {p['score']}) ---\n{p['description']}" for i, p in enumerate(kept, 1))
        reply = self.reduce_llm.invoke(REDUCE_PROMPT.format(points=text, question=question, max_words=self.max_words,
                                                            style=style and style + "\n"))
        cited = sorted({int(x) for p in kept for m in CITE.findall(p["description"]) for x in re.findall(r"\d+", m)
                        if int(x) in self.by_id})
        return {"answer": reply.content.strip(), "points": kept, "reports": cited, "map_calls": len(replies),
                "usage": usage + ([reply.usage_metadata] if reply.usage_metadata else [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--level", type=int, default=1)
    args = parser.parse_args()
    search = GlobalSearch(Config.from_env(), level=args.level)
    result = search.invoke(args.question)
    print(result["answer"])
    print(f"\n{len(search.reports)} báo cáo, {result['map_calls']} lô map, {len(result['points'])} ý vào reduce, "
          f"dẫn {len(result['reports'])} báo cáo")


if __name__ == "__main__":
    main()
