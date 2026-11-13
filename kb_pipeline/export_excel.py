"""Xuất chunks.jsonl ra Excel để rà soát: sheet Chunks (mọi trường + số từ / số ký tự bằng công thức) và Tổng hợp.

    python -m kb_pipeline.export_excel [--chunks FILE] [--out FILE]
(cần openpyxl: chạy bằng Python hệ thống; file .xlsx đang mở trong Excel thì không ghi được)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
FONT = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True, color="FFFFFF")
HEAD_FILL = PatternFill("solid", fgColor="1F4E78")
CALC_FILL = PatternFill("solid", fgColor="DDEBF7")
CALC = ["so_tu_text", "so_ky_tu_text", "so_tu_embed_text"]
WIDTHS = {"chunk_id": 38, "doc_key": 22, "doc_title": 40, "header": 60, "notes": 50, "text": 80,
          "embed_text": 60, "chuong": 30, "muc": 30, "dieu_title": 40, "source_file": 40}


def value(v):
    if isinstance(v, list):
        return ", ".join(map(str, v)) if v else None
    if isinstance(v, str):
        return ILLEGAL_CHARACTERS_RE.sub("", v)
    return v


def words(cell: str) -> str:
    # số từ = số âm tiết tách theo khoảng trắng (xuống dòng coi như khoảng trắng)
    clean = f'TRIM(SUBSTITUTE({cell},CHAR(10)," "))'
    return f'=IF(LEN({clean})=0,0,LEN({clean})-LEN(SUBSTITUTE({clean}," ",""))+1)'


def export(src: Path, out: Path) -> None:
    records = [json.loads(line) for line in src.open(encoding="utf-8")]
    fields = list(dict.fromkeys(k for r in records for k in r))
    wb = Workbook()
    ws = wb.active
    ws.title = "Chunks"
    header = fields + CALC
    ws.append(header)
    col = {name: get_column_letter(i + 1) for i, name in enumerate(header)}
    for n, r in enumerate(records, start=2):
        t, e = f"{col['text']}{n}", f"{col['embed_text']}{n}"
        ws.append([value(r.get(f)) for f in fields] + [words(t), f"=LEN({t})", words(e)])
    for cell in ws[1]:
        cell.font, cell.fill = BOLD, HEAD_FILL
        cell.alignment = Alignment(vertical="center")
    for name, letter in col.items():
        ws.column_dimensions[letter].width = WIDTHS.get(name, 14)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = FONT
    for name in CALC:
        for cell in ws[col[name]][1:]:
            cell.fill = CALC_FILL
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(header))}{len(records) + 1}"

    sm = wb.create_sheet("Tổng hợp")
    last = len(records) + 1
    rng = lambda name: f"Chunks!${col[name]}$2:${col[name]}${last}"  # noqa: E731
    sm.append(["Loại chunk", "Số chunk", "Số chunk truy hồi được", "Số từ trung bình", "Số từ nhỏ nhất",
               "Số từ lớn nhất", "Tổng số từ"])
    types = sorted({r["chunk_type"] for r in records})
    for i, kind in enumerate(types, start=2):
        a = f"A{i}"
        sm.append([kind,
                   f"=COUNTIF({rng('chunk_type')},{a})",
                   f"=COUNTIFS({rng('chunk_type')},{a},{rng('retrievable')},TRUE)",
                   f"=IFERROR(ROUND(AVERAGEIF({rng('chunk_type')},{a},{rng('so_tu_text')}),0),0)",
                   f"=_xlfn.MINIFS({rng('so_tu_text')},{rng('chunk_type')},{a})",
                   f"=_xlfn.MAXIFS({rng('so_tu_text')},{rng('chunk_type')},{a})",
                   f"=SUMIF({rng('chunk_type')},{a},{rng('so_tu_text')})"])
    total = len(types) + 2
    sm.append(["Tất cả", f"=SUM(B2:B{total - 1})", f"=SUM(C2:C{total - 1})",
               f"=IFERROR(ROUND(G{total}/B{total},0),0)", f"=MIN({rng('so_tu_text')})",
               f"=MAX({rng('so_tu_text')})", f"=SUM(G2:G{total - 1})"])
    usable = f"{rng('retrievable')},TRUE,{rng('active')},TRUE"
    sm.append([])
    sm.append(["Chunk truy hồi được (retrievable = TRUE, active = TRUE)"])
    sm.append(["Số chunk", f"=COUNTIFS({usable})"])
    sm.append(["Số từ trung bình", f"=IFERROR(ROUND(AVERAGEIFS({rng('so_tu_text')},{usable}),0),0)"])
    sm.append(["Số chunk > 500 từ", f"=COUNTIFS({usable},{rng('so_tu_text')},\">500\")"])
    sm.append(["Số chunk < 50 từ", f"=COUNTIFS({usable},{rng('so_tu_text')},\"<50\")"])
    sm.append([])
    sm.append(["Ghi chú"])
    sm.append(["Nguồn: " + str(src)])
    sm.append(["Số từ = số âm tiết trong cột text, tách theo khoảng trắng (xuống dòng tính như khoảng trắng)."])
    sm.append(["Cột tô xanh ở sheet Chunks (so_tu_text, so_ky_tu_text, so_tu_embed_text) là công thức, "
               "tự tính lại khi sửa text."])
    for row in sm.iter_rows():
        for cell in row:
            cell.font = FONT
    for cell in sm[1]:
        cell.font, cell.fill = BOLD, HEAD_FILL
    for cell in sm[total]:
        cell.font = Font(name="Arial", size=10, bold=True)
    sm.column_dimensions["A"].width = 34
    for letter in "BCDEFG":
        sm.column_dimensions[letter].width = 18

    wb.calculation.fullCalcOnLoad = True  # Excel tính mọi công thức khi mở file
    try:
        wb.save(out)
    except PermissionError:
        sys.exit(f"Không ghi được {out}: file đang mở trong Excel, hãy đóng rồi chạy lại")
    py_words = [len(r["text"].split()) for r in records]
    print(f"{len(records)} dòng -> {out}")
    print("Kiểm tra (Python, cùng định nghĩa số từ): tổng", sum(py_words),
          "| trung bình", round(sum(py_words) / len(py_words)), "| lớn nhất", max(py_words))


def main() -> None:
    parser = argparse.ArgumentParser(description="Xuất chunks.jsonl ra Excel.")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/chunk/chunk_v1/chunks.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="mặc định: chunks.xlsx cạnh file chunks")
    args = parser.parse_args()
    export(args.chunks, args.out or args.chunks.with_suffix(".xlsx"))


if __name__ == "__main__":
    main()
