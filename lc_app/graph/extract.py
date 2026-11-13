"""Bước 1: LLM trích thực thể và quan hệ từ từng text unit (chunk graph_unit = true) theo luồng của Microsoft GraphRAG.

Một tin nhắn user chứa toàn bộ prompt (ontology.render_prompt), gleaning 1 vòng (CONTINUE, mặc định của Microsoft),
các vòng nối bằng "##". Gọi ChatOpenAI qua LangChain, cache SQLite dùng chung (config.graph_cache): chạy lại chỉ tốn
tiền cho chunk mới hoặc khi prompt/mô hình đổi.

    .venv-lc/Scripts/python -m lc_app.graph.extract --demo            # 9 chunk demo (chủ đề tách thửa, hợp thửa)
    .venv-lc/Scripts/python -m lc_app.graph.extract --chunk ID [--chunk ID ...]
    .venv-lc/Scripts/python -m lc_app.graph.extract --sample 200      # chạy thử: 200 chunk ngẫu nhiên (seed 0)
    .venv-lc/Scripts/python -m lc_app.graph.extract --all             # toàn bộ chunk graph_unit
Đầu ra trong <graph_dir>:
    extractions.jsonl  mỗi dòng một chunk (thực thể, quan hệ, văn bản trả về của LLM, token), theo thứ tự đầu vào
    run.json           mô hình, mã băm prompt, số chunk, token, chunk lỗi
Chunk lỗi (hết lượt thử lại) được liệt kê và không ghi vào extractions.jsonl; chạy lại cùng lệnh để trích bù.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ..config import Config
from .ontology import COMPLETE, CONTINUE_PROMPT, RECORD, TUPLE, render_prompt

DEMO_CHUNKS = (
    "VN.31-2024-QH15::D220::K1",            # Luật Đất đai: điều kiện tách thửa, hợp thửa
    "VN.31-2024-QH15::D220::K3-4",          # Luật Đất đai: điều kiện hợp thửa; UBND tỉnh quy định chi tiết
    "VN.101-2024-ND-CP::D1",                # NĐ 101/2024: quy định chi tiết, hướng dẫn thi hành Luật Đất đai
    "VN.101-2024-ND-CP::D7::K3",            # NĐ 101/2024: trình tự, thủ tục tách thửa, hợp thửa
    "VN.49-2026-ND-CP::D12::K5",            # NĐ 49/2026: sửa đổi, bổ sung Điều 18 NĐ 101/2024
    "VN.254-2025-QH15::D11::K3",            # NQ 254/2025: lối đi khi tách thửa theo điểm d khoản 1 Điều 220
    "DN.75-2026-QD-UBND::QD::D1",           # QĐ 75/2026 Đà Nẵng: phạm vi điều chỉnh
    "DN.75-2026-QD-UBND::QD::D5::K1-2",     # QĐ 75/2026 Đà Nẵng: điều kiện tách thửa, hợp thửa
    "DN.135-2026-QD-UBND::QD::D18::K3",     # QĐ 135/2026 Đà Nẵng: thủ tục tách thửa, hợp thửa
)
MAX_GLEANINGS = 1
RECORD_RE = re.compile(r"\((.*)\)", re.S)


def load_units(config: Config, ids: list[str] | None) -> list[dict]:
    units = []
    with config.chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("graph_unit") and (ids is None or record["chunk_id"] in ids):
                units.append(record)
    if ids is not None:
        missing = set(ids) - {u["chunk_id"] for u in units}
        if missing:
            raise SystemExit(f"Không phải text unit (graph_unit = false hoặc không tồn tại): {sorted(missing)}")
        units.sort(key=lambda u: ids.index(u["chunk_id"]))
    return units


def doc_label(record: dict) -> str:
    """Tên văn bản đầy đủ như đầu header của chunk nhưng không cắt ngắn trích yếu."""
    kind, number, title = record["document_type"], record["document_number"], record["doc_title"]
    label = (f"{title} {number}" if kind in ("Luật", "Bộ luật")
             else f"{kind} {number} {title.removeprefix(kind).strip()}")
    if record.get("effective_date"):
        year, month, day = record["effective_date"].split("-")
        label += f" (hiệu lực từ {day}/{month}/{year})"
    return label


def full_header(record: dict) -> str:
    """Header như của chunk nhưng không cắt ngắn tên văn bản, tên Chương, tên Điều (header của chunk cắt ở 80–120 ký tự
    để gọn khi nhúng, LLM sẽ chép nguyên chỗ cắt, vd "thi hành Luật Đất…")."""
    parts = record["header"].strip("[]").split(" > ")
    if not record.get("dieu"):   # phụ lục: tên phụ lục bị cắt thì lấy lại dòng đầy đủ trong text của chunk
        lines = [ln.strip() for ln in record["text"].splitlines()]
        spans = [" ".join(lines[i:j]) for i in range(len(lines)) for j in range(i + 1, min(i + 4, len(lines) + 1))]
        rest = [next((s for s in spans if p.endswith("…") and s.startswith(p[:-1]) and len(s) >= len(p)), p)
                for p in parts[1:]]   # tên phụ lục có thể xuống dòng ("PHỤ LỤC" / "DANH MỤC ...")
        return "[" + " > ".join([doc_label(record), *rest]) + "]"
    # phần sau Điều (Khoản, Điểm) ngắn, không bị cắt: lấy nguyên văn từ header của chunk; chunk trọn Điều thì không có
    after = next((parts[i + 1:] for i, p in enumerate(parts) if p.startswith("Điều ")), [])
    path = [doc_label(record), "Quy định kèm theo" if record.get("part") else None, record.get("chuong"),
            record.get("muc"),
            f"Điều {record['dieu']}. {record['dieu_title']}" if record.get("dieu_title") else f"Điều {record['dieu']}",
            *after]
    return "[" + " > ".join(p for p in path if p) + "]"


def unit_text(record: dict) -> str:
    """header đầy đủ + text; không đưa notes (ghi chú sửa đổi do pipeline suy ra từ văn bản khác)."""
    return f"{full_header(record)}\n{record['text']}"


def prompt_hash() -> str:
    """Mã băm của prompt trích xuất + CONTINUE: đổi prompt thì đổi mã (và cache không còn dùng được)."""
    return hashlib.sha256((render_prompt("{input_text}") + CONTINUE_PROMPT).encode("utf-8")).hexdigest()[:12]


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip('"').strip()).strip()


def parse(raw: str) -> tuple[list[dict], list[dict]]:
    """Phân tích chuỗi tuple như _process_result của Microsoft; bản ghi hỏng bị bỏ qua, quan hệ thiếu độ mạnh lấy 1."""
    entities, relations = [], []
    for record in raw.replace(COMPLETE, "").split(RECORD):
        match = RECORD_RE.search(record)
        if not match:
            continue
        fields = [clean(f) for f in match.group(1).split(TUPLE)]
        kind = fields[0].lower()
        if kind == "entity" and len(fields) >= 4:
            entities.append({"name": fields[1], "type": fields[2].upper(), "description": fields[3]})
        elif kind == "relationship" and len(fields) >= 5:
            try:
                weight = float(fields[5]) if len(fields) >= 6 else 1.0
            except ValueError:
                weight = 1.0
            relations.append({"source": fields[1], "target": fields[2], "type": fields[3].upper(),
                              "description": fields[4], "weight": weight})
    return entities, relations


def extract(llm: ChatOpenAI, record: dict) -> dict:
    messages = [HumanMessage(render_prompt(unit_text(record)))]
    replies = [llm.invoke(messages)]
    for _ in range(MAX_GLEANINGS):   # với 1 vòng, Microsoft không hỏi LOOP (còn sót không) sau vòng cuối
        messages += [AIMessage(replies[-1].content), HumanMessage(CONTINUE_PROMPT)]
        replies.append(llm.invoke(messages))
    raw = RECORD.join(r.content.replace(COMPLETE, "").strip() for r in replies)   # nối các vòng bằng "##"
    entities, relations = parse(raw)
    usage = [r.usage_metadata or {} for r in replies]   # bản ghi từ cache giữ token của lần gọi gốc
    return {"chunk_id": record["chunk_id"], "doc_key": record["doc_key"], "entities": entities,
            "relations": relations, "raw": raw,
            "tokens": {"input": sum(u.get("input_tokens", 0) for u in usage),
                       "cached": sum(u.get("input_token_details", {}).get("cache_read", 0) for u in usage),
                       "output": sum(u.get("output_tokens", 0) for u in usage)}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Trích thực thể/quan hệ theo LEGAL_ONTOLOGY.md.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true")
    group.add_argument("--chunk", action="append")
    group.add_argument("--sample", type=int, metavar="N", help="N chunk ngẫu nhiên (seed 0), để chạy thử")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    config = Config.from_env()
    config.graph_dir.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(config.graph_cache)))
    # lỗi tạm thời (429 vượt hạn mức, mạng) được LangChain/OpenAI thử lại với thời gian chờ tăng dần
    llm = ChatOpenAI(model=config.graph_llm_model, temperature=0, max_retries=8, timeout=180)
    ids = list(DEMO_CHUNKS) if args.demo else args.chunk
    units = load_units(config, ids)
    if args.sample:
        units = random.Random(0).sample(units, min(args.sample, len(units)))
    print(f"{len(units)} text unit, mô hình {config.graph_llm_model}, prompt {prompt_hash()} -> {config.graph_dir}",
          flush=True)

    results, failed, start = {}, {}, time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(extract, llm, unit): unit["chunk_id"] for unit in units}
        for done, future in enumerate(as_completed(futures), 1):
            chunk_id = futures[future]
            try:
                results[chunk_id] = future.result()
            except Exception as error:   # một chunk lỗi không làm mất các chunk khác
                failed[chunk_id] = f"{type(error).__name__}: {error}"[:300]
            if done % 25 == 0 or done == len(units):
                elapsed = time.time() - start
                print(f"  {done}/{len(units)} ({len(failed)} lỗi), {elapsed / 60:.1f} phút, "
                      f"còn ~{elapsed / done * (len(units) - done) / 60:.0f} phút", flush=True)

    out: Path = config.graph_dir / "extractions.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for unit in units:
            if unit["chunk_id"] in results:
                handle.write(json.dumps(results[unit["chunk_id"]], ensure_ascii=False) + "\n")
    tokens = {k: sum(r["tokens"][k] for r in results.values()) for k in ("input", "cached", "output")}
    run = {"model": config.graph_llm_model, "prompt_sha": prompt_hash(), "max_gleanings": MAX_GLEANINGS,
           "chunks_path": str(config.chunks_path), "units": len(units), "extracted": len(results),
           "failed": failed, "tokens": tokens, "minutes": round((time.time() - start) / 60, 1)}
    (config.graph_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{sum(len(r['entities']) for r in results.values())} thực thể, "
          f"{sum(len(r['relations']) for r in results.values())} quan hệ, token {tokens} -> {out}")
    if failed:
        print(f"{len(failed)} chunk lỗi (xem run.json), chạy lại cùng lệnh để trích bù:", *list(failed)[:10], sep="\n  ")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
