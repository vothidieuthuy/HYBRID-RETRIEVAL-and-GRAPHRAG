"""Làm sạch chunk sau bước chia và chốt bộ chunk dùng chung cho naive RAG (lc_app) và GraphRAG.

Nguồn: data/chunk/chunk_v1/chunks.jsonl (đầu ra của kb_pipeline.chunk).
Đích:  data/chunk/chunk_v1/clean_chunk_v1/chunks.jsonl, manifest.json, clean_report.csv.

Một file, chung chunk_id, hai góc nhìn:
- retrievable = true : truy hồi (dense/BM25/hybrid của lc_app).
- graph_unit = true  : text unit của GraphRAG = chunk truy hồi được, trừ phụ lục bảng giá/mã/định mức/ví dụ
  (GRAPH_EXCLUDED, lý do ghi ở graph_exclude_reason). Chunk bị loại ghi excluded_by = tên quy tắc.

Quy tắc (chỉ xét chunk truy hồi được, cấp 2):
- form_annex: phụ lục là mẫu biểu (đơn, quyết định, hợp đồng, báo cáo mẫu) mà kb_pipeline.chunk không nhận ra vì
  tiêu đề "Mẫu số 04e. Quyết định ..." nằm cùng dòng với tên mẫu. Xét cả phụ lục: >= 2 tiêu đề "Mẫu số X. ...",
  hoặc đầu phụ lục có "Mẫu số"/"HỢP ĐỒNG MẪU", hoặc trung bình >= 8 chỗ trống ("…", "...") mỗi chunk.
  (Phụ lục nội dung có 0–5 chỗ trống mỗi chunk, phụ lục mẫu 8–28.)
- duplicate: nội dung giống hệt sau chuẩn hóa (chữ thường, bỏ dấu câu/khoảng trắng) -> giữ chunk đầu tiên.
- contained: chunk phụ lục mà phần thân (bỏ dòng dẫn đầu là tên phụ lục) nằm trọn trong một chunk phụ lục dài
  hơn -> giữ chunk dài hơn. Không áp dụng cho Điều/Khoản: hai Điều khác nhau có khoản chép y nhau vẫn là hai
  quy định riêng (vd Điều 7 và Điều 9 QĐ 135/2026).
  Chunk bị khử vẫn được ghi ra (retrievable = false, duplicate_of = chunk giữ lại) để truy vết; chunk_id không đổi.
- placeholder: "(Ban hành) kèm theo Thông tư số /2025/TT-BNNMT ngày tháng năm 2025" trong chính văn bản đó
  -> điền số hiệu và ngày ban hành từ metadata. Sửa ở text, header và embed_text. Các chỗ trống khác
  (vd "Căn cứ Nghị định số /2025/NĐ-CP" trong mẫu tờ trình) không điền vì có thể là văn bản khác.
Chunk trùng giữa các văn bản khác nhau (vd Điều 2 NĐ 49/2026 và NĐ 50/2026) được giữ: khác nguồn, khác hiệu lực.

    python -m kb_pipeline.clean_chunks [--src FILE] [--out DIR] [--force]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ("chunks.jsonl", "manifest.json", "clean_report.csv")
MIN_CONTAINED = 200        # phần thân ngắn hơn thì không xét "nằm trọn" (dễ trùng ngẫu nhiên)
DOC_TYPES = {"TT": "Thông tư", "NĐ": "Nghị định", "QĐ": "Quyết định", "NQ": "Nghị quyết"}
NON_WORD = re.compile(r"\W+")
FORM_HEADING = re.compile(r"^(?:Mẫu số|MẪU SỐ)\s+([\w/-]+)\.\s+\S", re.M)
FORM_MARK = re.compile(r"Mẫu số|HỢP ĐỒNG MẪU|\bMẪU\b")
BLANK = re.compile(r"…|\.{3,}")
FORM_MIN_HEADINGS = 2
FORM_MIN_BLANKS = 8        # trung bình mỗi chunk của phụ lục
# Phụ lục giữ cho truy hồi (tra cứu số liệu) nhưng không đưa vào đồ thị: bảng số/mã cho ra nút vô nghĩa
# ("Cây cói", "Mộ bê tông") làm nhiễu cộng đồng. Phụ lục trình tự thủ tục, danh mục TTHC, hạn mức vẫn vào đồ thị.
GRAPH_EXCLUDED = {
    "DN.61-2026-QD-UBND::PL1": "bảng đơn giá bồi thường",
    "DN.61-2026-QD-UBND::PL2": "bảng đơn giá bồi thường",
    "DN.81-2026-QD-UBND::PL1": "bảng đơn giá bồi thường",
    "DN.81-2026-QD-UBND::PL2": "bảng đơn giá bồi thường",
    "DN.81-2026-QD-UBND::PL3": "bảng đơn giá bồi thường",
    "DN.81-2026-QD-UBND::PL4": "bảng đơn giá bồi thường",
    "DN.81-2026-QD-UBND::PL5": "bảng mật độ cây trồng",
    "DN.74-2026-QD-UBND::PL1": "mẫu bảng tổng hợp",
    "VN.226-2025-ND-CP::PL1": "ví dụ tính giá đất",
    "VN.23-2025-TT-BNNMT::PL1": "danh mục biểu, ký hiệu loại đất",
    "VN.05-2024-TT-BXD@hn::PL2": "định mức chuyên gia",
    "VN.10-2024-TT-BTNMT::PL5": "hướng dẫn kỹ thuật sơ đồ thửa đất",
    "VN.10-2024-TT-BTNMT::PL6": "bảng mã",
    "VN.10-2024-TT-BTNMT::PL7": "bảng mã",
    "VN.357-2025-ND-CP::PL1": "cấu trúc mã định danh",
    "VN.357-2025-ND-CP::PL2": "danh mục trường dữ liệu",
    "VN.357-2025-ND-CP::PL5": "danh mục dữ liệu thu phí",
    "VN.95-2024-ND-CP::PL6": "bảng giá thuê nhà",
    "VN.95-2024-ND-CP::PL7": "công thức tính tiền thuê",
    "VN.95-2024-ND-CP::PL8": "phương pháp tính giá trị còn lại",
    "VN.95-2024-ND-CP::PL10": "hướng dẫn phân cấp nhà",
    "VN.95-2024-ND-CP::PL11": "bảng phân cấp nhà",
    "VN.95-2024-ND-CP::PL12": "bảng tiêu chí phân hạng nhà chung cư",
}


def norm(text: str) -> str:
    return NON_WORD.sub(" ", unicodedata.normalize("NFC", text).lower()).strip()


def body(text: str) -> str:
    """Bỏ dòng dẫn đầu (tên phụ lục, được lặp ở mọi chunk của phụ lục)."""
    return text.split("\n", 1)[1] if "\n" in text else text


def is_candidate(record: dict) -> bool:
    return record["retrievable"] and record["level"] == 2


def annex_of(record: dict) -> str | None:
    return record["chunk_id"].rsplit("::", 1)[0] if record["chunk_type"] == "phu_luc" else None


def mark(record: dict, rule: str, report: list[dict], keep: dict | None = None, detail: str = "") -> None:
    record["retrievable"] = False
    record["excluded_by"] = rule
    if keep:
        record["duplicate_of"] = keep["chunk_id"]
    report.append({"chunk_id": record["chunk_id"], "doc_key": record["doc_key"], "rule": rule,
                   "detail": keep["chunk_id"] if keep else detail, "chars": len(record["text"]),
                   "preview": record["text"][:120].replace("\n", " / ")})


def drop_forms(records: list[dict], report: list[dict]) -> None:
    annexes: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if is_candidate(record) and annex_of(record):
            annexes[annex_of(record)].append(record)
    for annex, chunks in annexes.items():
        headings = {h for r in chunks for h in FORM_HEADING.findall(r["text"])}
        blanks = sum(len(BLANK.findall(r["text"])) for r in chunks) / len(chunks)
        signals = [s for s, hit in (("mẫu", len(headings) >= FORM_MIN_HEADINGS),
                                    ("tiêu đề mẫu", bool(FORM_MARK.search(chunks[0]["text"][:400]))),
                                    ("chỗ trống", blanks >= FORM_MIN_BLANKS)) if hit]
        if signals:
            detail = f"{annex}: {', '.join(signals)} ({len(headings)} mẫu, {blanks:.1f} chỗ trống/chunk)"
            for record in chunks:
                mark(record, "form_annex", report, detail=detail)


def dedupe(records: list[dict], report: list[dict]) -> None:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if is_candidate(record):
            by_doc[record["doc_key"]].append(record)
    for chunks in by_doc.values():
        first: dict[str, dict] = {}
        for record in chunks:
            key = norm(record["text"])
            if key in first:
                mark(record, "duplicate", report, keep=first[key])
            else:
                first[key] = record
        # chunk ngắn xét trước; chunk chứa nó luôn dài hơn nên không thể bị khử ngược lại vào chunk ngắn
        annexes = sorted((r for r in chunks if r["retrievable"] and r["chunk_type"] == "phu_luc"),
                         key=lambda r: len(r["text"]))
        texts = {r["chunk_id"]: norm(r["text"]) for r in annexes}
        for i, record in enumerate(annexes):
            part = norm(body(record["text"]))
            if len(part) < MIN_CONTAINED:
                continue
            keep = next((other for other in annexes[i + 1:]
                         if other["retrievable"] and len(other["text"]) > len(record["text"])
                         and part in texts[other["chunk_id"]]), None)
            if keep:
                mark(record, "contained", report, keep=keep)


def resolve_roots(records: list[dict], report: list[dict]) -> None:
    """Chunk giữ lại ở bước duplicate có thể bị khử tiếp ở bước contained: trỏ thẳng về chunk còn truy hồi."""
    target = {r["chunk_id"]: r["duplicate_of"] for r in records if r.get("duplicate_of")}

    def root(chunk_id: str) -> str:
        while chunk_id in target:
            chunk_id = target[chunk_id]
        return chunk_id

    for record in records:
        if record.get("duplicate_of"):
            record["duplicate_of"] = root(record["duplicate_of"])
    for row in report:
        row["detail"] = root(row["detail"])


def placeholder_pattern(number: str) -> re.Pattern | None:
    match = re.fullmatch(r"\d+/(\d{4})/((TT|NĐ|QĐ|NQ)[\w-]*)", number)
    if not match:
        return None
    year, suffix, kind = match.groups()
    # chỉ cụm "kèm theo <loại> số /năm/..." chắc chắn tự dẫn; "Căn cứ Nghị định số /2025/NĐ-CP" trong mẫu
    # tờ trình có thể là văn bản khác (vd mẫu ở Phụ lục III NĐ 151/2025)
    return re.compile(rf"([Kk]èm theo\s+{DOC_TYPES[kind]} số)\s*/{year}/{re.escape(suffix)}"
                      rf"((\s+ngày)\s+(?:\.+\s*)?tháng\s+(?:\.+\s*)?năm\s+{year})?")


def fill_placeholders(records: list[dict], report: list[dict]) -> None:
    for record in records:
        if record["level"] != 2 or not record.get("document_number"):
            continue
        pattern = placeholder_pattern(record["document_number"])
        if not pattern or not pattern.search(record["text"]):
            continue
        issued = dt.date.fromisoformat(record["issue_date"]) if record.get("issue_date") else None

        def fill(match: re.Match) -> str:
            filled = f"{match.group(1)} {record['document_number']}"
            if match.group(2):   # có cụm ngày bỏ trống: điền ngày ban hành, không có metadata thì giữ nguyên
                filled += (f"{match.group(3)} {issued.day} tháng {issued.month} năm {issued.year}" if issued
                           else match.group(2))
            return filled

        before = pattern.search(record["text"]).group(0)
        for field in ("text", "header", "embed_text"):
            if record.get(field):
                record[field] = pattern.sub(fill, record[field])
        report.append({"chunk_id": record["chunk_id"], "doc_key": record["doc_key"], "rule": "placeholder",
                       "detail": f"{before} -> {pattern.sub(fill, before)}", "chars": len(record["text"]),
                       "preview": ""})


def mark_graph_units(records: list[dict]) -> None:
    seen = {annex_of(r) for r in records}
    missing = sorted(set(GRAPH_EXCLUDED) - seen)
    if missing:   # chunk_id của phụ lục đổi khi chia lại: danh sách phải cập nhật theo
        sys.exit(f"GRAPH_EXCLUDED có phụ lục không còn tồn tại: {missing}")
    for record in records:
        reason = GRAPH_EXCLUDED.get(annex_of(record))
        record["graph_unit"] = is_candidate(record) and not reason
        if is_candidate(record) and reason:
            record["graph_exclude_reason"] = reason


def run(src: Path, out: Path, force: bool) -> None:
    if (out / "chunks.jsonl").exists() and not force:
        sys.exit(f"{out} đã có chunks.jsonl; thêm --force để làm sạch lại")
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:   # chỉ xóa file do bước này tạo (index/, eval/ ... giữ nguyên)
        (out / name).unlink(missing_ok=True)
    records = [json.loads(line) for line in src.open(encoding="utf-8")]
    before = sum(r["retrievable"] for r in records)
    report: list[dict] = []
    drop_forms(records, report)
    dedupe(records, report)
    resolve_roots(records, report)
    fill_placeholders(records, report)
    mark_graph_units(records)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    retrievable = [r for r in records if is_candidate(r)]
    if report:
        with (out / "clean_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report[0]))
            writer.writeheader()
            writer.writerows(report)
    manifest = {"source": str(src), "date": dt.date.today().isoformat(), "chunks": len(records),
                "retrievable_before": before, "retrievable_after": sum(r["retrievable"] for r in records),
                "by_rule": dict(Counter(r["rule"] for r in report)),
                "by_document": dict(Counter(r["doc_key"] for r in report if r["rule"] != "placeholder")),
                "retrievable_by_type": dict(Counter(r["chunk_type"] for r in retrievable)),
                "graph_units": sum(r["graph_unit"] for r in records),
                "graph_excluded": dict(Counter(r["graph_exclude_reason"] for r in retrievable
                                               if r.get("graph_exclude_reason"))),
                "sha256": hashlib.sha256((out / "chunks.jsonl").read_bytes()).hexdigest(),
                "settings": {"min_contained": MIN_CONTAINED, "form_min_headings": FORM_MIN_HEADINGS,
                             "form_min_blanks": FORM_MIN_BLANKS}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Làm sạch chunks.jsonl (khử trùng lặp, điền số hiệu trống).")
    parser.add_argument("--src", type=Path, default=ROOT / "data/chunk/chunk_v1/chunks.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data/chunk/chunk_v1/clean_chunk_v1")
    parser.add_argument("--force", action="store_true", help="ghi đè kết quả làm sạch cũ")
    args = parser.parse_args()
    run(args.src, args.out, args.force)


if __name__ == "__main__":
    main()
