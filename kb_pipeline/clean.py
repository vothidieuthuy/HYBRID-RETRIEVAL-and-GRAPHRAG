"""Làm sạch text đã trích xuất trước khi chia chunk.

Nguồn: data/data/processed_kb/extracted/<nhóm>/<văn bản>/<role>.txt (+ <role>.footnotes.json, metadata.json).
Đích:  data/data/processed_kb/clean_extracted/ cùng cấu trúc, kèm clean_report.csv (số lần áp dụng từng quy tắc).

Quy tắc (theo thứ tự áp dụng trên từng dòng; dòng [BẢNG]/[/BẢNG] giữ nguyên):
  unicode       NFC; Ð/ð (ký tự Iceland) -> Đ/đ; bỏ ký tự vô hình (zero-width, soft hyphen, BOM, điều khiển)
  footnote      {{FN:id}} -> bỏ ký hiệu, gắn nội dung chú thích vào cuối dòng: "(Ghi chú hợp nhất: ...)"
                với văn bản hợp nhất, "(Chú thích: ...)" với văn bản khác
  note_marker   bỏ ký hiệu chú thích "[n]" giữa dòng (mẫu biểu); giữ dòng nội dung "[n] Ghi ..."
  separator     xóa dòng chỉ gồm gạch/chấm (____, -----, .....); bỏ vạch kẻ trong dòng ("CHÍNH PHỦ -------")
  motto         bỏ quốc hiệu, tiêu ngữ in hoa ở đầu văn bản / mẫu biểu (giữ câu "...nước Cộng hòa xã hội
                chủ nghĩa Việt Nam khóa XV ... thông qua" ở cuối luật)
  dot_leader    dãy chấm điền mẫu "……….." -> "…"
  quotes        ngoặc kép thẳng "..." -> “...” để bước chia chunk nhận đúng đoạn trích dẫn sửa đổi
  doc_number    "102/2024/NĐ- CP", "TT- BTNMT", "31 /2024/QH 15" -> dạng chuẩn; mã thủ tục "1.012783 .H17" -> "1.012783.H17"
  punct_space   bỏ khoảng trắng trước dấu câu ("đất ," -> "đất,")
  signature     bỏ bảng "Nơi nhận" / khối chữ ký (TM./KT. ... CHỦ TỊCH ...) và dòng chữ ký đứng riêng
                ("CHỦ TỊCH QUỐC HỘI (Đã ký) ..."); danh sách đã bỏ ghi ở removed_signatures.csv để rà soát

    python -m kb_pipeline.clean [--src DIR] [--out DIR] [--force]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from kb_pipeline.extract import GROUPS, TABLE_END, TABLE_START, reset_output

ROOT = Path(__file__).resolve().parents[1]
RULES = ("unicode", "footnote", "note_marker", "separator", "motto", "dot_leader", "quotes", "doc_number",
         "punct_space", "signature")

INVISIBLE = re.compile(r"[­​-‏⁠﻿  \x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
FOOTNOTE = re.compile(r"\{\{FN:(\d+)\}\}")
NOTE_MARKER = re.compile(r"\s?\[\d{1,3}\]")
SEPARATOR_LINE = re.compile(r"^[\s_\-–—=*.…·]{3,}$")
INLINE_RULE = re.compile(r"\s*(?:_{3,}|-{3,}|={3,})\s*")
DOT_LEADER = re.compile(r"(?:[.…]\s*){4,}")
DOC_NUMBER = re.compile(r"\b(\d{1,4})\s*/\s*(\d{4})\s*/\s*(NĐ|NQ|TT|QĐ|VBHN)\s*-\s*([A-ZĐ][A-ZĐ&]*)")
LAW_NUMBER = re.compile(r"\b(\d{1,4})\s*/\s*(\d{4})\s*/\s*QH\s*(\d{2})\b")
TTHC_CODE = re.compile(r"\b(\d\.\d{6})\s*\.\s*(H\d{2})\b")
PUNCT_SPACE = re.compile(r"(?<=[^\s|…]) +([,;:.!?])(?=\s|$)")
# Quốc hiệu, tiêu ngữ ở đầu văn bản / mẫu biểu (in hoa); câu "...Quốc hội nước Cộng hòa xã hội chủ nghĩa Việt Nam
# khóa XV ... thông qua" viết thường nên không bị bỏ.
MOTTO = re.compile(r"CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM|CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM|"
                   r"Độc lập\s*[-–]\s*Tự do\s*[-–]\s*Hạnh phúc")


# Khối nơi nhận / chữ ký: bảng có "Nơi nhận" hoặc thẩm quyền ký "TM./KT./TL./Q." + chức danh in hoa;
# dòng chữ ký đứng riêng bắt đầu bằng chức danh in hoa ("CHỦ TỊCH QUỐC HỘI (Đã ký) ...").
# Chức danh viết thường trong nội dung ("Chủ tịch UBND cấp xã") không bị bắt.
SIGN_TABLE = re.compile(r"Nơi nhận|\b(?:TM|KT|TL|Q)\.\s*[A-ZĐ]{2,}")
SIGN_LINE = re.compile(r"^(?:(?:TM|KT|TL|Q)\.\s*)*(?:[A-ZĐ]+\.?\s+)*?(?:PHÓ\s+)?(?:CHỦ TỊCH|BỘ TRƯỞNG|THỦ TƯỚNG|"
                       r"CHỦ NHIỆM)\b")
SIGN_LINE_MAX = 200

HEADING = re.compile(r"^(Điều\s+\d+[a-z]?\s*[.:]|Chương\s+[IVXLC\d]+\b|Mục\s+\d+\b|Phụ lục\b|PHỤ LỤC\b)",
                     re.IGNORECASE)


def is_heading(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    upper = sum(c.isupper() for c in letters) / len(letters) if letters else 0
    return bool(HEADING.match(line)) or (len(line) <= 250 and upper > 0.8)


class Cleaner:
    def __init__(self, footnotes: dict[str, str], consolidated: bool):
        self.footnotes = footnotes
        self.label = "Ghi chú hợp nhất" if consolidated else "Chú thích"
        self.counts: Counter = Counter()
        self.removed: list[str] = []  # khối nơi nhận / chữ ký đã bỏ, để rà soát

    def sub(self, rule: str, pattern: re.Pattern, repl, text: str) -> str:
        text, n = pattern.subn(repl, text)
        self.counts[rule] += n
        return text

    def footnote_notes(self, line: str) -> tuple[str, list[str]]:
        notes = []

        def take(match: re.Match) -> str:
            note = self.footnotes.get(match.group(1))
            if note:
                # ngoặc kép trong chú thích đổi thành nháy đơn để không làm lệch việc theo dõi đoạn trích dẫn
                notes.append(note.replace("“", "'").replace("”", "'").replace('"', "'"))
            return ""

        return self.sub("footnote", FOOTNOTE, take, line), notes

    def quotes(self, line: str) -> str:
        if '"' not in line:
            return line
        out = []
        for i, char in enumerate(line):
            if char == '"':
                opening = i == 0 or line[i - 1] in " \t([{–-:"
                out.append("“" if opening else "”")
                self.counts["quotes"] += 1
            else:
                out.append(char)
        return "".join(out)

    def line(self, line: str) -> str | None:
        if line in (TABLE_START, TABLE_END):
            return line
        original = line
        line = unicodedata.normalize("NFC", line).replace(" ", " ").replace("Ð", "Đ").replace("ð", "đ")
        line = INVISIBLE.sub("", line)
        if line != original:
            self.counts["unicode"] += 1
        line, notes = self.footnote_notes(line)
        if not line.startswith("["):  # dòng nội dung chú thích "[n] Ghi ..." giữ nguyên ký hiệu
            line = self.sub("note_marker", NOTE_MARKER, "", line)
        if notes:
            # tiêu đề (Chương/Mục/Điều, dòng in hoa) giữ ngắn: chú thích xuống dòng riêng ngay sau
            note = f"({self.label}: {' '.join(notes)})"
            line = f"{line.rstrip()}\n{note}" if is_heading(line.strip()) else f"{line.rstrip()} {note}"
        return "\n".join(filter(None, (self.normalize(part) for part in line.split("\n")))) or None

    def normalize(self, line: str) -> str | None:
        if SEPARATOR_LINE.match(line):
            self.counts["separator"] += 1
            line = ""
        line = self.sub("separator", INLINE_RULE, " ", line)
        line = self.sub("motto", MOTTO, " ", line)
        line = re.sub(r"(\s*\|\s*)+$|^(\s*\|\s*)+", "", re.sub(r"\|\s*\|", "|", line))  # ô bảng rỗng sau khi bỏ
        line = self.sub("dot_leader", DOT_LEADER, "… ", line)
        line = self.quotes(line)
        line = self.sub("doc_number", DOC_NUMBER, r"\1/\2/\3-\4", line)
        line = self.sub("doc_number", LAW_NUMBER, r"\1/\2/QH\3", line)
        line = self.sub("doc_number", TTHC_CODE, r"\1.\2", line)
        line = self.sub("punct_space", PUNCT_SPACE, r"\1", line)
        line = re.sub(r"\s+", " ", line).strip(" |").strip()
        return line or None

    def drop_signatures(self, lines: list[str]) -> list[str]:
        """Bỏ bảng nơi nhận / chữ ký và dòng chữ ký đứng riêng."""
        out: list[str] = []
        table: list[str] | None = None
        for index, line in enumerate(lines):
            following = lines[index + 1] if index + 1 < len(lines) else ""
            if line == TABLE_START:
                table = []
            elif line == TABLE_END and table is not None:
                block = "\n".join(table)
                if SIGN_TABLE.search(block) or (table and all(SIGN_LINE.match(r) for r in table)
                                               and len(block) <= SIGN_LINE_MAX * len(table)):
                    self.counts["signature"] += 1
                    self.removed.append(block)
                elif table:
                    out.extend([TABLE_START, *table, TABLE_END])
                table = None
            elif table is not None:
                table.append(line)
            elif len(line) <= SIGN_LINE_MAX and SIGN_LINE.match(line) and not following.startswith("Căn cứ"):
                # dòng chức danh ngay trước "Căn cứ ..." là thẩm quyền ban hành ("CHỦ TỊCH UBND THÀNH PHỐ ĐÀ NẴNG"), giữ
                self.counts["signature"] += 1
                self.removed.append(line)
            else:
                out.append(line)
        return out

    def text(self, text: str) -> str:
        lines = [self.line(raw) for raw in text.splitlines()]
        kept = self.drop_signatures([part for l in lines if l is not None for part in l.split("\n")])
        return "\n".join(kept) + "\n"


def run(src: Path, out: Path, force: bool) -> None:
    reset_output(out, "clean_report.csv", force)
    rows, removed = [], []
    for group in GROUPS:
        for folder in sorted(p for p in (src / group).iterdir() if p.is_dir()):
            target = out / group / folder.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / "metadata.json", target / "metadata.json")
            for txt in sorted(folder.glob("*.txt")):
                role = txt.stem
                notes_path = folder / f"{role}.footnotes.json"
                footnotes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}
                cleaner = Cleaner(footnotes, consolidated=role == "consolidated")
                raw = txt.read_text(encoding="utf-8")
                cleaned = cleaner.text(raw)
                (target / txt.name).write_text(cleaned, encoding="utf-8")
                removed += [{"group": group, "folder": folder.name, "role": role, "chars": len(block),
                             "text": block[:300].replace("\n", " ⏎ ")} for block in cleaner.removed]
                unresolved = len(FOOTNOTE.findall(raw)) - cleaner.counts["footnote"]
                rows.append({"group": group, "folder": folder.name, "role": role,
                             "chars_before": len(raw), "chars_after": len(cleaned),
                             **{rule: cleaner.counts[rule] for rule in RULES},
                             "footnotes_without_text": unresolved + sum(1 for k in FOOTNOTE.findall(raw)
                                                                        if k not in footnotes)})
    if removed:
        with (out / "removed_signatures.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(removed[0]))
            writer.writeheader()
            writer.writerows(removed)
    with (out / "clean_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    totals = {rule: sum(r[rule] for r in rows) for rule in RULES}
    before, after = sum(r["chars_before"] for r in rows), sum(r["chars_after"] for r in rows)
    print(f"Đã làm sạch {len(rows)} file vào {out}")
    print(f"Ký tự: {before:,} -> {after:,} ({(after - before) / before:+.1%}); số lần áp dụng: {totals}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Làm sạch text đã trích xuất.")
    parser.add_argument("--src", type=Path, default=ROOT / "data/data/processed_kb/extracted")
    parser.add_argument("--out", type=Path, default=ROOT / "data/data/processed_kb/clean_extracted")
    parser.add_argument("--force", action="store_true", help="xóa và làm sạch lại thư mục đích")
    args = parser.parse_args()
    run(args.src, args.out, args.force)


if __name__ == "__main__":
    main()
