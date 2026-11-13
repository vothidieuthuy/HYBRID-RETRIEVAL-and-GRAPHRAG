"""Đọc file Word trong processed_raw_kb ra text thuần, giữ thứ tự đoạn văn và bảng.

Đích: data/data/processed_kb/extracted/<nhóm>/<văn bản>/<role>.txt + metadata.json (chép từ nguồn),
kèm extract_report.csv để rà soát.

Định dạng text: mỗi đoạn văn một dòng; mỗi hàng bảng một dòng (các ô nối bằng " | "), bảng được bao bởi
dòng [BẢNG] và [/BẢNG] để bước chia chunk không nhận nhầm tiêu đề Điều nằm trong bảng. Chú thích cuối trang:
vị trí tham chiếu ghi {{FN:id}}, nội dung lưu ở <role>.footnotes.json (bước clean gắn vào dòng tương ứng).

    python -m kb_pipeline.extract [--src DIR] [--out DIR] [--force]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("01_luat", "02_nghi_dinh", "03_thong_tu", "04_da_nang")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
TABLE_START, TABLE_END = "[BẢNG]", "[/BẢNG]"
ARTICLE = re.compile(r"^Điều\s+(\d+)[a-z]?\s*(?:[.:]|$)")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace(" ", " ").replace("​", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def paragraph_text(para: ElementTree.Element) -> list[str]:
    """Text một đoạn; ngắt dòng mềm (w:br) tách thành nhiều dòng để tiêu đề không dính vào nội dung.

    Tham chiếu chú thích cuối trang được ghi thành {{FN:id}} (nội dung ở <role>.footnotes.json); ký hiệu hiển
    thị tự đặt ngay sau nó (customMarkFollows, vd "[1]") bị bỏ để không lặp."""
    parts: list[str] = []
    skip_mark = False
    for node in para.iter():
        if node.tag == f"{W}footnoteReference":
            parts.append(f"{{{{FN:{node.get(f'{W}id')}}}}}")
            skip_mark = node.get(f"{W}customMarkFollows") in ("1", "true")
        elif node.tag == f"{W}t":
            if skip_mark:
                skip_mark = False
                continue
            parts.append(node.text or "")
        elif node.tag == f"{W}tab":
            parts.append(" ")
        elif node.tag in (f"{W}br", f"{W}cr"):
            parts.append("\n")
    return [line for line in (clean(x) for x in "".join(parts).split("\n")) if line]


def cell_text(cell: ElementTree.Element) -> str:
    return " ".join(line for para in cell.iter(f"{W}p") for line in paragraph_text(para))


def body_lines(body: ElementTree.Element, stats: dict) -> list[str]:
    lines: list[str] = []
    for child in body:
        if child.tag == f"{W}p":
            if child.find(f"{W}pPr/{W}numPr") is not None:
                stats["auto_numbered"] += 1
            lines.extend(paragraph_text(child))
        elif child.tag == f"{W}tbl":
            stats["tables"] += 1
            lines.append(TABLE_START)
            for row in child.iter(f"{W}tr"):
                cells = [cell_text(c) for c in row.findall(f"{W}tc")]
                if any(cells):
                    lines.append(" | ".join(c for c in cells if c))
            lines.append(TABLE_END)
        elif child.tag == f"{W}sdt":  # khối nội dung có kiểm soát: đọc phần thân bên trong
            content = child.find(f"{W}sdtContent")
            if content is not None:
                lines.extend(body_lines(content, stats))
    return lines


def read_footnotes(archive: zipfile.ZipFile) -> dict[str, str]:
    """{id: nội dung} của chú thích cuối trang (bỏ chú thích phân cách và ký hiệu '[n]' đầu nội dung)."""
    if "word/footnotes.xml" not in archive.namelist():
        return {}
    root = ElementTree.fromstring(archive.read("word/footnotes.xml"))
    notes = {}
    for note in root.iter(f"{W}footnote"):
        if note.get(f"{W}type") in ("separator", "continuationSeparator", "continuationNotice"):
            continue
        text = " ".join(line for para in note.iter(f"{W}p") for line in paragraph_text(para))
        text = re.sub(r"^\{\{FN:\d+\}\}\s*", "", text)
        text = re.sub(r"^\[\d+\]\s*", "", text).strip()
        if text:
            notes[note.get(f"{W}id")] = text
    return notes


def docx_to_text(path: Path) -> tuple[str, dict, dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
        footnotes = read_footnotes(archive)
    stats = {"tables": 0, "auto_numbered": 0}
    lines = body_lines(root.find(f"{W}body"), stats)
    text = "\n".join(lines) + "\n"
    numbers = [int(m.group(1)) for m in (ARTICLE.match(line) for line in lines) if m]
    stats.update(characters=len(text), lines=len(lines), article_headings=len(numbers),
                 footnotes=len(footnotes), footnote_refs=text.count("{{FN:"))
    return text, stats, footnotes


def reset_output(out: Path, marker: str, force: bool) -> None:
    """Chỉ xóa thư mục đích khi nó rỗng hoặc do pipeline tạo ra (có file marker), tránh xóa nhầm dữ liệu khác."""
    if out.exists() and any(out.iterdir()):
        if not force:
            sys.exit(f"{out} đã có dữ liệu; thêm --force để làm lại")
        if not (out / marker).exists():
            sys.exit(f"{out} không phải thư mục do pipeline tạo (thiếu {marker}); không xóa")
        shutil.rmtree(out)


def extract(src: Path, out: Path, force: bool) -> None:
    reset_output(out, "extract_report.csv", force)
    rows = []
    for group in GROUPS:
        for folder in sorted(p for p in (src / group).iterdir() if p.is_dir()):
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            target = out / group / folder.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / "metadata.json", target / "metadata.json")
            for entry in meta["files"]:
                text, stats, footnotes = docx_to_text(folder / entry["file"])
                (target / f"{entry['role']}.txt").write_text(text, encoding="utf-8")
                if footnotes:
                    (target / f"{entry['role']}.footnotes.json").write_text(
                        json.dumps(footnotes, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                warnings = []
                if stats["characters"] < 2000:
                    warnings.append("văn bản rất ngắn")
                if stats["auto_numbered"]:
                    warnings.append(f"{stats['auto_numbered']} đoạn đánh số tự động (số thứ tự không có trong text)")
                rows.append({"group": group, "folder": folder.name, "doc_key": meta["doc_key"],
                             "role": entry["role"], **stats, "warnings": "; ".join(warnings)})
    with (out / "extract_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["warnings"]:
            print(f"[cảnh báo] {row['group']}/{row['folder']} ({row['role']}): {row['warnings']}")
    print(f"Đã trích xuất {len(rows)} file vào {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trích xuất text từ file Word của processed_raw_kb.")
    parser.add_argument("--src", type=Path, default=ROOT / "data/data/processed_kb/processed_raw_kb")
    parser.add_argument("--out", type=Path, default=ROOT / "data/data/processed_kb/extracted")
    parser.add_argument("--force", action="store_true", help="xóa và trích xuất lại thư mục đích")
    args = parser.parse_args()
    extract(args.src, args.out, args.force)


if __name__ == "__main__":
    main()
