"""Chia text đã trích xuất thành chunk theo Chương / Mục / Điều / Khoản, xuất chunks.jsonl cho lc_app.

Nguồn: data/data/processed_kb/clean_extracted/<nhóm>/<văn bản>/{main,consolidated}.txt + metadata.json
       (đầu ra của kb_pipeline.clean).
Đích:  data/chunk/chunk_v1/chunks.jsonl, manifest.json, chunk_report.csv, amendments.csv.

Quy tắc chính:
- Văn bản có bản hợp nhất thì chỉ chia bản hợp nhất (doc_id thêm "@hn").
- Điều ngắn (<= WHOLE ký tự) giữ nguyên một chunk; Điều dài gộp các Khoản liền nhau tới ~TARGET ký tự,
  Khoản quá MAX ký tự tách tiếp theo Điểm (lặp lại câu dẫn của Khoản).
- Đoạn trích dẫn trong ngoặc kép của văn bản sửa đổi ("Điều 34. ..." nằm trong “...”) không được nhận là Điều
  mới; nội dung đó thuộc chunk của Điều sửa đổi chứa nó.
- Phần "Quy định ban hành kèm theo" của Quyết định/Nghị quyết đánh số Điều lại từ đầu: tách thành phần riêng (QD).
- Phụ lục được chia theo độ dài; mẫu biểu (Mẫu số ...) giữ lại nhưng không cho truy hồi.
- Ngày hiệu lực lấy từ metadata, ghi vào header (để nhúng) và vào ghi chú (để LLM đọc trong ngữ cảnh).
- Điều bị văn bản khác trong KB sửa đổi được ghi chú "đã được sửa đổi, bổ sung bởi ..."; lệnh sửa đổi đã nằm
  trong bản hợp nhất của văn bản đích thì không cho truy hồi (tránh trùng nội dung).

Định dạng mỗi dòng chunks.jsonl khớp lc_app/loader.py: chunk_id, level (0 thẻ văn bản, 2 Điều/Khoản/phụ lục),
retrievable, active, header, embed_text (= header + ghi chú + text), text và các trường metadata trong KEPT.

    python -m kb_pipeline.chunk [--src DIR] [--out DIR] [--as-of YYYY-MM-DD] [--force]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from kb_pipeline.build_metadata import DOC_NUMBER, doc_number
from kb_pipeline.extract import GROUPS, TABLE_END, TABLE_START

ROOT = Path(__file__).resolve().parents[1]
TARGET, MAX, WHOLE, TAIL, LEAD_MAX = 1400, 2400, 1800, 300, 600
HEADER_TITLE = 120
OUTPUTS = ("chunks.jsonl", "manifest.json", "chunk_report.csv", "amendments.csv")

ARTICLE = re.compile(r"^Điều\s+(\d+)([a-z]?)\s*(?:[.:]\s*(.*))?$")
CHAPTER = re.compile(r"^Chương\s+([IVXLC]+|\d+)\b\.?\s*(.*)$", re.IGNORECASE)
SECTION = re.compile(r"^Mục\s+(\d+)\b\.?\s*(.*)$", re.IGNORECASE)
KHOAN = re.compile(r"^(\d{1,3}[a-z]?)\.\s+\S")
DIEM = re.compile(r"^([a-zđ]\d?)\)\s")
ATTACHED = re.compile(r"\(\s*(?:được\s+)?(?:ban hành\s+)?kèm theo\b", re.IGNORECASE)
REG_HEADING = re.compile(r"^(QUY ĐỊNH|QUY CHẾ|ĐIỀU LỆ|QUY TRÌNH)\b")
ANNEX_HEADING = re.compile(r"^(PHỤ LỤC|Phụ lục)(\s+[IVXLC\d]+[a-z]?)?\b")
FORM_HEADING = re.compile(r"^(Mẫu số|MẪU SỐ)\s+[\w./-]+(\[\d+\])?$")
# Phụ lục là văn bản mẫu (hợp đồng, đơn, báo cáo, quyết định mẫu...): giữ lại nhưng không cho truy hồi.
FORM_TITLE = re.compile(r"\b(HỢP ĐỒNG|MẪU|ĐƠN ĐỀ NGHỊ|ĐƠN ĐĂNG KÝ|BÁO CÁO|QUYẾT ĐỊNH|TỜ KHAI|BIÊN BẢN|BẢNG KÊ|"
                        r"PHIẾU|VĂN BẢN ĐỀ NGHỊ|GIẤY ĐỀ NGHỊ|THÔNG BÁO)\b|^Phụ lục (hợp đồng|[A-Z]\b)")
SIGNATURE = re.compile(r"Nơi nhận|\b(?:TM|KT|TL|Q)\.\s|CHỦ TỊCH|BỘ TRƯỞNG|THỦ TƯỚNG|CHỦ NHIỆM")
AMEND_WORDS = re.compile(r"\b(sửa đổi|bổ sung|bãi bỏ|thay thế|bỏ cụm từ)\b", re.IGNORECASE)
DIEU_REF = re.compile(r"\bĐiều\s+(\d+[a-z]?)\b")
QUOTED_SPAN = re.compile(r"“[^”]*”")


# ---------------------------------------------------------------- cấu trúc
@dataclass
class Item:
    """Một dòng văn bản hoặc một bảng (nhiều dòng)."""
    text: str
    table: bool = False
    quoted: bool = False  # nằm trong (hoặc mở đầu) đoạn trích dẫn của văn bản sửa đổi


@dataclass
class Article:
    part: str
    chuong: str | None
    muc: str | None
    num: str
    title: str
    items: list[Item] = field(default_factory=list)
    targets: dict[str, set[str]] = field(default_factory=dict)  # văn bản đích -> các Điều bị sửa
    redundant: bool = False  # mọi lệnh sửa đổi đã có trong bản hợp nhất của văn bản đích

    @property
    def heading(self) -> str:
        return f"Điều {self.num}." + (f" {self.title}" if self.title else "")


@dataclass
class Segment:
    kind: str  # body | regulation | annex | form
    title: str
    items: list[Item]


@dataclass
class Doc:
    group: str
    folder: str
    meta: dict
    role: str
    preamble: list[Item] = field(default_factory=list)
    articles: list[Article] = field(default_factory=list)
    annexes: list[Segment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return self.meta["doc_key"] + ("@hn" if self.role == "consolidated" else "")


# ---------------------------------------------------------------- tiện ích
def vn_date(value: str | None) -> str | None:
    return dt.date.fromisoformat(value).strftime("%d/%m/%Y") if value else None


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if len(text) > 1 and text[1:2].islower() else text


def doc_title(meta: dict) -> str:
    """Tên dùng trong trích dẫn: 'Luật Đất đai', 'Nghị định quy định về giá đất'."""
    title, kind = meta["title"], meta["document_type"]
    return title if title.lower().startswith(kind.lower()) else f"{kind} {lower_first(title)}"


def doc_label(meta: dict) -> str:
    """Tên ngắn kèm số hiệu cho header: 'Luật Đất đai 31/2024/QH15', 'Nghị định 71/2024/NĐ-CP quy định về giá đất'."""
    title, kind, number = meta["title"], meta["document_type"], meta["document_number"]
    if title.lower().startswith(kind.lower()):
        label = f"{title} {number}"
    else:
        label = f"{kind} {number} {lower_first(title)}"
    return label if len(label) <= HEADER_TITLE else label[:HEADER_TITLE].rsplit(" ", 1)[0] + "…"


def short_ref(meta: dict) -> str:
    return f"{meta['document_type']} {meta['document_number']}"


def clip(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


# ---------------------------------------------------------------- đọc và tách phần
def read_items(text: str) -> list[Item]:
    """Gom dòng thành Item; theo dõi độ sâu ngoặc kép “ ” (chỉ ngoài bảng).

    Ngoặc kép quên đóng (lỗi soạn thảo) sẽ kéo mọi dòng sau thành trích dẫn; khi gặp tiêu đề Điều không mở bằng
    ngoặc kép và đúng số thứ tự tiếp theo của Điều thật gần nhất thì coi như ngoặc đã đóng."""
    items, depth, table, last = [], 0, None, None
    for line in text.splitlines():
        if line == TABLE_START:
            table = []
            continue
        if line == TABLE_END:
            if table:
                items.append(Item("\n".join(table), table=True, quoted=depth > 0))
            table = None
            continue
        if table is not None:
            table.append(line)
            continue
        opens_quote = line.startswith(("“", '"'))
        match = ARTICLE.match(line)
        if match and depth > 0 and not opens_quote and last is not None and int(match.group(1)) == last + 1:
            depth = 0
        quoted = depth > 0 or opens_quote
        if match and not quoted:
            last = int(match.group(1))
        items.append(Item(line, quoted=quoted))
        depth = max(0, depth + line.count("“") - line.count("”"))
    return items


def split_segments(items: list[Item], own_numbers: set[str]) -> list[Segment]:
    """Tách thân văn bản, các 'Quy định ban hành kèm theo' và phụ lục / mẫu biểu."""
    first_article = next((i for i, it in enumerate(items) if not it.table and it.quoted is False
                          and ARTICLE.match(it.text)), len(items))
    bounds: list[tuple[int, str]] = []
    for i, it in enumerate(items):
        if i <= first_article or it.table or it.quoted:
            continue
        text = it.text
        if ATTACHED.search(text):
            start = i
            for k in range(i - 1, max(first_article, i - 5), -1):
                prev = items[k]
                if prev.table or len(prev.text) > 500:
                    break
                if REG_HEADING.match(prev.text) or ANNEX_HEADING.match(prev.text) or FORM_HEADING.match(prev.text):
                    start = k
                    break
            head = items[start].text
            # bảng tiêu đề ngắn ngay trước (tên cơ quan, số, ngày) thuộc về phần kèm theo, không thuộc Điều trước
            while start - 1 > first_article and items[start - 1].table and len(items[start - 1].text) < 400 \
                    and not SIGNATURE.search(items[start - 1].text):
                start -= 1
            if ANNEX_HEADING.match(head):
                bounds.append((start, "annex"))
            elif FORM_HEADING.match(head) or head.upper().startswith("MẪU"):
                bounds.append((start, "form"))
            elif REG_HEADING.match(head) and any(n in text for n in own_numbers):
                bounds.append((start, "regulation"))
        elif len(text) <= 80 and ANNEX_HEADING.match(text) and ":" not in text:
            bounds.append((i, "annex"))
        elif FORM_HEADING.match(text):
            bounds.append((i, "form"))
    # Ranh giới phụ lục mà sau đó vẫn còn Điều đúng số thứ tự tiếp theo thì thực ra nằm trong thân văn bản
    # (vd dòng 'Phụ lục II' liệt kê trong một Điều); phần 'Quy định kèm theo' đánh số lại nên không xét.
    real = [(i, int(m.group(1))) for i, it in enumerate(items)
            if not it.table and not it.quoted and (m := ARTICLE.match(it.text))]

    def inside_body(index: int) -> bool:
        before = [n for i, n in real if i < index]
        return bool(before) and any(i > index and n == before[-1] + 1 for i, n in real)

    kept, in_annex = [], False  # chỉ kiểm tra ranh giới phụ lục đầu tiên sau mỗi phần thân / quy định kèm theo
    for index, kind in sorted(set(bounds)):
        if kind == "regulation":
            in_annex = False
        elif not in_annex:
            if inside_body(index):
                continue
            in_annex = True
        kept.append((index, kind))
    bounds = kept
    # bỏ ranh giới trùng (cùng một tiêu đề phụ lục bị bắt hai lần)
    merged: list[tuple[int, str]] = []
    for index, kind in sorted(set(bounds)):
        if merged and index - merged[-1][0] <= 4 and kind != "regulation" and merged[-1][1] != "regulation":
            continue
        merged.append((index, kind))
    segments, edges = [], [(0, "body")] + merged + [(len(items), "end")]
    for (start, kind), (stop, _) in zip(edges, edges[1:]):
        chunk = items[start:stop]
        if not chunk:
            continue
        lines = [it.text for it in chunk if not it.table][:2]  # bỏ qua bảng tiêu đề (tên cơ quan, số, ngày)
        title = " ".join(t for t in lines if len(t) <= 250)
        if kind == "annex" and FORM_TITLE.search(re.sub(r"^(PHỤ LỤC|Phụ lục)\s+[IVXLC\d]+[a-z]?\b", "", title)):
            kind = "form"
        segments.append(Segment(kind, clip(title, 200), chunk))
    # phụ lục chỉ có tiêu đề (vd 'Phụ lục I' ngay trước 'Mẫu số 01'): gộp tiêu đề vào phần sau
    result: list[Segment] = []
    for seg in segments:
        if result and result[-1].kind in ("annex", "form") and sum(len(i.text) for i in result[-1].items) < 200:
            prev = result.pop()
            seg = Segment(seg.kind, clip(prev.title + " – " + seg.title, 200), prev.items + seg.items)
        result.append(seg)
    return result


def is_title_line(it: Item | None) -> bool:
    """Dòng có thể là (phần tiếp theo của) tên Chương/Mục: ngắn, không phải tiêu đề khác, chủ yếu in hoa."""
    if it is None or it.table or it.quoted or len(it.text) > 250:
        return False
    if ARTICLE.match(it.text) or SECTION.match(it.text) or CHAPTER.match(it.text):
        return False
    letters = [c for c in it.text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8


def heading_title(items: list[Item], index: int, rest: str) -> tuple[str, int]:
    """Tên Chương/Mục: phần còn lại của dòng tiêu đề (hoặc dòng kế tiếp nếu dòng chỉ có số), nối thêm các dòng
    in hoa ngay sau khi tên dài bị xuống dòng ("SỞ HỮU NHÀ Ở TẠI VIỆT NAM" / "CỦA TỔ CHỨC, CÁ NHÂN NƯỚC NGOÀI")."""
    parts = [rest] if rest else []
    if not rest:
        nxt = items[index + 1] if index + 1 < len(items) else None
        if nxt and not nxt.table and len(nxt.text) <= 250 and not ARTICLE.match(nxt.text) \
                and not SECTION.match(nxt.text) and not CHAPTER.match(nxt.text):
            parts.append(nxt.text)
            index += 1
    while parts and is_title_line(items[index + 1] if index + 1 < len(items) else None):
        parts.append(items[index + 1].text)
        index += 1
    return " ".join(parts), index


def parse_body(doc: Doc, seg: Segment, part: str) -> None:
    items, chuong, muc, current, last = seg.items, None, None, None, None
    skip_to = -1
    for i, it in enumerate(items):
        if i <= skip_to:
            continue
        if it.table:
            if current and SIGNATURE.search(it.text) and len(it.text) < 2000:
                continue  # bảng chữ ký / nơi nhận
            (current.items if current else doc.preamble).append(it)
            continue
        match = ARTICLE.match(it.text)
        if match:
            number = int(match.group(1))
            real = not it.quoted and (last is None or last <= number <= last + 3)
            if not it.quoted and not real:
                doc.warnings.append(f"bỏ qua tiêu đề 'Điều {match.group(1)}' sai thứ tự (sau Điều {last})")
            if real:
                current = Article(part, chuong, muc, match.group(1) + match.group(2), (match.group(3) or "").strip())
                current.items.append(it)
                doc.articles.append(current)
                last = number
                continue
        if not it.quoted:
            chapter, section = CHAPTER.match(it.text), SECTION.match(it.text)
            # "Chương V của Nghị định này ..." là câu nội dung, không phải tiêu đề
            chapter = chapter if chapter and not chapter.group(2)[:1].islower() else None
            section = section if section and not section.group(2)[:1].islower() else None
            if chapter:
                title, skip_to = heading_title(items, i, chapter.group(2).strip())
                chuong, muc, current = f"Chương {chapter.group(1).upper()}" + (f". {title}" if title else ""), None, None
                continue
            if section:
                title, skip_to = heading_title(items, i, section.group(2).strip())
                muc, current = f"Mục {section.group(1)}" + (f". {title}" if title else ""), None
                continue
        (current.items if current else doc.preamble).append(it)


def parse_doc(group: str, folder: Path) -> Doc:
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    role = "consolidated" if (folder / "consolidated.txt").exists() else "main"
    doc = Doc(group, folder.name, meta, role)
    number = meta["document_number"]
    own = {number, re.sub(r"^0+(?=\d)", "", number)}
    items = read_items((folder / f"{role}.txt").read_text(encoding="utf-8"))
    regulation = 0
    for seg in split_segments(items, own):
        if seg.kind == "body":
            parse_body(doc, seg, "")
        elif seg.kind == "regulation":
            regulation += 1
            parse_body(doc, seg, "QD" if regulation == 1 else f"QD{regulation}")
        else:
            doc.annexes.append(seg)
    if not doc.articles:
        doc.warnings.append("không tìm thấy Điều nào")
    return doc


# ---------------------------------------------------------------- quan hệ sửa đổi
def analyse_amendments(docs: list[Doc]) -> list[dict]:
    """Từ lệnh sửa đổi của văn bản có 'amends' trong metadata, ghi nhận Điều nào của văn bản đích bị sửa."""
    by_number = {d.meta["document_number"]: d for d in docs}
    records = []
    for doc in docs:
        targets = [n for n in doc.meta.get("amends", []) if n in by_number]
        if not targets:
            continue
        for art in (a for a in doc.articles if a.part == ""):
            context = next((n for n in targets for t in (art.chuong or "", art.muc or "", art.title) if n in t), None)
            current = context or (targets[0] if len(targets) == 1 else None)
            lines = [art.title] + [it.text for it in art.items[1:] if not it.table and not it.quoted]
            for line in lines:
                visible = QUOTED_SPAN.sub(" ", line).split("“")[0]
                numbers = [(m.start(), doc_number(m)) for m in DOC_NUMBER.finditer(visible)
                           if doc_number(m) in targets]
                if numbers and (line is art.title or re.match(r"^(\d+|[a-zđ])[.)]\s", line) is None):
                    current = numbers[-1][1]
                if not AMEND_WORDS.search(visible):
                    continue
                for ref in DIEU_REF.finditer(visible):
                    after = [n for pos, n in numbers if pos > ref.start()]
                    target = after[0] if after else (numbers[-1][1] if numbers else current)
                    if not target:
                        continue
                    base = by_number[target]
                    if any(a.num == ref.group(1) and a.part == "" for a in base.articles):
                        art.targets.setdefault(target, set()).add(ref.group(1))
                if current and not art.targets:
                    art.targets.setdefault(current, set())
            for target, dieus in art.targets.items():
                base = by_number[target]
                hn = base.meta.get("consolidated") or {}
                covered = base.role == "consolidated" and doc.meta["document_number"] in hn.get("consolidates", [])
                for dieu in sorted(dieus):
                    records.append({"amender": doc.meta["document_number"], "amender_article": art.num,
                                    "target": target, "target_article": dieu,
                                    "status": "da_hop_nhat" if covered else "ghi_chu"})
            art.redundant = bool(art.targets) and all(
                by_number[t].role == "consolidated"
                and doc.meta["document_number"] in (by_number[t].meta.get("consolidated") or {}).get("consolidates", [])
                for t in art.targets)
    return records


# ---------------------------------------------------------------- chia chunk
def split_long(text: str) -> list[str]:
    """Cắt đoạn quá MAX theo dòng, rồi theo câu, cuối cùng cắt cứng."""
    pieces: list[str] = []
    for line in text.split("\n"):
        if len(line) <= MAX:
            pieces.append(line)
            continue
        sentences = re.split(r"(?<=[.;:])\s+", line)
        for sentence in sentences:
            pieces.extend(sentence[i:i + MAX] for i in range(0, len(sentence), MAX))
    out, buf = [], ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 1 > TARGET:
            out.append(buf)
            buf = piece
        else:
            buf = f"{buf}\n{piece}" if buf else piece
    if buf:
        out.append(buf)
    return out


def pack(pieces: list[tuple[str | None, str]]) -> list[tuple[list[str], str]]:
    """Gộp các mảnh liền nhau tới ~TARGET ký tự; mảnh quá MAX tự tách; mảnh cuối quá ngắn gộp vào trước."""
    out: list[tuple[list[str], str]] = []
    labels: list[str] = []
    buf = ""
    for label, text in pieces:
        if len(text) > MAX:
            if buf:
                out.append((labels, buf))
            out.extend(([label] if label else [], part) for part in split_long(text))
            labels, buf = [], ""
            continue
        if buf and len(buf) + len(text) + 1 > TARGET:
            out.append((labels, buf))
            labels, buf = [], ""
        buf = f"{buf}\n{text}" if buf else text
        if label:
            labels.append(label)
    if buf:
        out.append((labels, buf))
    if len(out) >= 2 and len(out[-1][1]) < TAIL and len(out[-2][1]) + len(out[-1][1]) + 1 <= MAX:
        tail = out.pop()
        out[-1] = (out[-1][0] + tail[0], out[-1][1] + "\n" + tail[1])
    return out


def units_of(items: list[Item], pattern: re.Pattern) -> list[tuple[str | None, list[Item]]]:
    """Tách danh sách Item theo dòng mở đầu Khoản/Điểm (bỏ qua dòng trong trích dẫn và bảng)."""
    units: list[tuple[str | None, list[Item]]] = [(None, [])]
    for it in items:
        match = None if it.table or it.quoted else pattern.match(it.text)
        if match:
            units.append((match.group(1), [it]))
        else:
            units[-1][1].append(it)
    return [u for u in units if u[1]]


def text_of(items: list[Item]) -> str:
    return "\n".join(it.text for it in items)


def article_chunks(art: Article) -> list[dict]:
    """[{khoan, diem, text}] cho một Điều."""
    full = text_of(art.items)
    khoan_units = units_of(art.items[1:], KHOAN)
    all_khoan = [label for label, _ in khoan_units if label]
    if len(full) <= WHOLE:
        return [{"khoan": all_khoan, "diem": [], "text": full}]
    chunks, pending = [], []

    def flush() -> None:
        for labels, text in pack(pending):
            chunks.append({"khoan": labels, "diem": [], "text": text})
        pending.clear()

    for label, items in khoan_units:
        body = text_of(items)
        if len(body) <= MAX:
            pending.append((label, body))
            continue
        flush()
        khoan = [label] if label else []
        diem_units = units_of(items, DIEM)
        lead_items, rest = (diem_units[0][1], diem_units[1:]) if diem_units[0][0] is None else ([], diem_units)
        if not rest:  # Khoản dài không có Điểm: cắt theo dòng / câu
            chunks.extend({"khoan": khoan, "diem": [], "text": t} for t in split_long(body))
            continue
        lead = text_of(lead_items)
        if len(lead) > LEAD_MAX:  # câu dẫn quá dài để lặp lại: đứng thành chunk riêng
            chunks.extend({"khoan": khoan, "diem": [], "text": t} for t in split_long(lead))
            lead = ""
        for diem, text in pack([(d, text_of(its)) for d, its in rest]):
            chunks.append({"khoan": khoan, "diem": diem, "text": f"{lead}\n{text}" if lead else text})
    flush()
    for chunk in chunks:  # mỗi chunk đều mở đầu bằng tên Điều để LLM biết đang đọc Điều nào
        if not chunk["text"].startswith(art.heading.split(".")[0] + "."):
            chunk["text"] = f"{art.heading}\n{chunk['text']}"
    return chunks


def range_label(labels: list[str]) -> str:
    return labels[0] if len(labels) == 1 else f"{labels[0]}-{labels[-1]}"


# ---------------------------------------------------------------- ghép bản ghi
class Writer:
    def __init__(self, docs: list[Doc], as_of: str):
        self.docs, self.as_of = docs, as_of
        self.by_number = {d.meta["document_number"]: d for d in docs}
        self.amended_by: dict[tuple[str, str], list[tuple[Doc, str]]] = defaultdict(list)
        self.amended_docs: dict[str, list[str]] = defaultdict(list)
        for doc in docs:
            for target in doc.meta.get("amends", []):
                if target in self.by_number:
                    self.amended_docs[target].append(doc.meta["document_number"])
            for art in doc.articles:
                if art.redundant:
                    continue
                for target, dieus in art.targets.items():
                    for dieu in dieus:
                        self.amended_by[(target, dieu)].append((doc, art.num))
        self.seen_ids: Counter = Counter()

    def validity(self, meta: dict) -> tuple[bool, str]:
        start, end = meta.get("effective_date"), meta.get("expiry_date")
        if start and start > self.as_of:
            return False, "chua_co_hieu_luc"
        if end and end <= self.as_of:
            return False, "het_hieu_luc"
        return True, "con_hieu_luc" if start else "khong_ro_ngay_hieu_luc"

    def effective_note(self, meta: dict) -> str:
        note = f"Hiệu lực thi hành từ {vn_date(meta['effective_date'])}" if meta.get("effective_date") \
            else "Chưa rõ ngày hiệu lực"
        if meta.get("expiry_date"):
            note += f", thực hiện đến trước ngày {vn_date(meta['expiry_date'])}"
        hn = meta.get("consolidated")
        if hn and hn.get("number"):
            note += (f". Nội dung theo văn bản hợp nhất {hn['number']}" + (f" ngày {vn_date(hn['date'])}" if hn.get("date") else "")
                     + (f" (đã cập nhật sửa đổi của {', '.join(hn['consolidates'])})" if hn.get("consolidates") else ""))
        return note + "."

    def unique(self, chunk_id: str) -> str:
        self.seen_ids[chunk_id] += 1
        return chunk_id if self.seen_ids[chunk_id] == 1 else f"{chunk_id}#{self.seen_ids[chunk_id]}"

    def record(self, doc: Doc, chunk_id: str, level: int, chunk_type: str, path: list[str], text: str,
               notes: list[str], retrievable: bool, **extra) -> dict:
        meta = doc.meta
        active, status = self.validity(meta)
        header = "[" + " > ".join([f"{doc_label(meta)} (hiệu lực từ {vn_date(meta['effective_date']) or '?'})"]
                                  + [clip(p) for p in path if p]) + "]"
        note_text = "\n".join(n for n in notes if n)
        return {
            "chunk_id": self.unique(chunk_id), "doc_key": meta["doc_key"], "level": level, "chunk_type": chunk_type,
            "retrievable": retrievable, "active": active, "validity_status": status,
            "document_number": meta["document_number"], "document_type": meta["document_type"],
            "doc_title": doc_title(meta), "issuing_authority": meta["issuing_authority"],
            "issue_date": meta["issue_date"], "effective_date": meta["effective_date"],
            "expiry_date": meta.get("expiry_date"), "scope": meta["scope"],
            "part": extra.get("part"), "chuong": extra.get("chuong"), "muc": extra.get("muc"),
            "dieu": extra.get("dieu"), "dieu_title": extra.get("dieu_title"),
            "khoan": extra.get("khoan") or [], "diem": extra.get("diem") or [],
            "header": header, "notes": note_text or None, "text": text,
            "embed_text": "\n".join(x for x in (header, note_text, text) if x),
            "source_file": f"{doc.group}/{doc.folder}/{doc.role}.txt",
        }

    def doc_card(self, doc: Doc) -> dict:
        meta = doc.meta
        lines = [f"{doc_title(meta)} số {meta['document_number']}",
                 f"Cơ quan ban hành: {meta['issuing_authority']}.",
                 f"Ngày ban hành: {vn_date(meta['issue_date'])}.",
                 f"Ngày có hiệu lực: {vn_date(meta['effective_date']) or 'chưa rõ'}."]
        if meta.get("expiry_date"):
            lines.append(f"Thực hiện đến trước ngày: {vn_date(meta['expiry_date'])}.")
        lines.append("Phạm vi áp dụng: " + ("thành phố Đà Nẵng." if meta["scope"] == "da_nang" else "toàn quốc."))
        if meta.get("amends"):
            lines.append("Sửa đổi, bổ sung: " + "; ".join(
                short_ref(self.by_number[n].meta) if n in self.by_number else n for n in meta["amends"]) + ".")
        if self.amended_docs.get(meta["document_number"]):
            lines.append("Được sửa đổi, bổ sung bởi: " + "; ".join(
                short_ref(self.by_number[n].meta) for n in self.amended_docs[meta["document_number"]]) + ".")
        ref = lambda n: short_ref(self.by_number[n].meta) if n in self.by_number else n  # noqa: E731
        if meta.get("guides"):
            lines.append("Quy định chi tiết / hướng dẫn thi hành: " + "; ".join(map(ref, meta["guides"])) + ".")
        guided_by = [d.meta["document_number"] for d in self.docs if meta["document_number"] in d.meta.get("guides", [])]
        if guided_by:
            lines.append("Được quy định chi tiết / hướng dẫn bởi: " + "; ".join(map(ref, guided_by)) + ".")
        # based_on (căn cứ ban hành) chỉ giữ trong metadata để liên kết văn bản: hầu như văn bản nào cũng căn cứ
        # Luật Đất đai, Luật Nhà ở... nên trong thẻ chỉ là danh sách lặp lại, không giúp trả lời câu hỏi
        if meta.get("replaces"):
            lines.append("Thay thế / bãi bỏ (văn bản đã hết hiệu lực): " + "; ".join(meta["replaces"]) + ".")
        for part in meta.get("partial_effective", []):
            when = (f"có hiệu lực từ {vn_date(part['from'])}" if part.get("from")
                    else f"thực hiện đến hết {vn_date(part['until'])}" if part.get("until") else None)
            lines.append(f"Hiệu lực riêng: {clip(part['provisions'], 160)}" + (f" {when}." if when
                                                                               else f" ({clip(part['source'], 200)})."))
        hn = meta.get("consolidated")
        if hn and hn.get("number"):
            lines.append(f"Văn bản hợp nhất: {hn['number']}" + (f" ngày {vn_date(hn['date'])}." if hn.get("date") else "."))
        chapters = list(dict.fromkeys(a.chuong for a in doc.articles if a.chuong and a.part == ""))
        if chapters:
            lines.append("Bố cục: " + "; ".join(clip(c, 90) for c in chapters) + ".")
        return self.record(doc, f"{meta['doc_key']}::CARD", 0, "doc_card", ["Thông tin văn bản"],
                           "\n".join(lines), [], True)

    def article_records(self, doc: Doc, art: Article) -> list[dict]:
        meta = doc.meta
        notes = [self.effective_note(meta)]
        if art.part:
            notes.append(f"Điều này thuộc Quy định ban hành kèm theo {short_ref(meta)}.")
        if art.targets:
            notes.append("Điều này sửa đổi, bổ sung " + "; ".join(
                short_ref(self.by_number[t].meta) + (f" (Điều {', '.join(sorted(d, key=natural))})" if d else "")
                for t, d in art.targets.items()) + ".")
        if art.part == "":
            amenders = self.amended_by.get((meta["document_number"], art.num), [])
            if amenders:
                notes.append("Lưu ý: Điều này đã được sửa đổi, bổ sung bởi " + "; ".join(
                    f"Điều {num} {short_ref(a.meta)} (hiệu lực {vn_date(a.meta['effective_date'])})"
                    for a, num in amenders) + "; cần đối chiếu nội dung sửa đổi.")
        retrievable = not art.redundant
        if art.redundant:
            notes.append("Nội dung sửa đổi này đã được cập nhật trong văn bản hợp nhất của văn bản đích.")
        base_id = f"{doc.doc_id}::" + (f"{art.part}::" if art.part else "") + f"D{art.num}"
        pieces = article_chunks(art)
        suffixes = [(f"K{range_label(p['khoan'])}" if p["khoan"] else f"P{n}")
                    + (f"::{range_label(p['diem'])}" if p["diem"] else "") for n, p in enumerate(pieces, 1)]
        repeats, seen = Counter(suffixes), Counter()
        records = []
        for n, piece in enumerate(pieces, 1):
            if len(pieces) == 1:
                chunk_id, chunk_type = base_id, "article"
            else:
                suffix = suffixes[n - 1]
                seen[suffix] += 1
                if repeats[suffix] > 1:  # cùng một Khoản bị cắt thành nhiều phần
                    suffix += f"::p{seen[suffix]}"
                chunk_id, chunk_type = f"{base_id}::{suffix}", "clause"
            path = ["Quy định kèm theo" if art.part else None, art.chuong, art.muc, art.heading,
                    (f"Khoản {range_label(piece['khoan'])}" if piece["khoan"] and len(pieces) > 1 else None)]
            records.append(self.record(doc, chunk_id, 2, chunk_type, path, piece["text"], notes, retrievable,
                                       part=art.part or None, chuong=art.chuong, muc=art.muc, dieu=art.num,
                                       dieu_title=art.title or None, khoan=piece["khoan"], diem=piece["diem"]))
        return records

    def annex_records(self, doc: Doc, index: int, seg: Segment) -> list[dict]:
        text = text_of(seg.items)
        notes = [self.effective_note(doc.meta), f"Nội dung phụ lục/mẫu biểu kèm theo {short_ref(doc.meta)}."]
        chunk_type, retrievable = ("mau_bieu", False) if seg.kind == "form" else ("phu_luc", True)
        records = []
        for n, part in enumerate(split_long(text), 1):
            if n > 1 and len(seg.title) <= 200:
                part = f"{seg.title}\n{part}"
            records.append(self.record(doc, f"{doc.doc_id}::PL{index}::{n}", 2, chunk_type, [clip(seg.title, 100)],
                                       part, notes, retrievable))
        return records

    def preamble_records(self, doc: Doc) -> list[dict]:
        """Phần mở đầu (quốc hiệu, căn cứ): không truy hồi, trừ khi văn bản không có Điều nào."""
        text = text_of(doc.preamble)
        parts = split_long(text) if text else []
        return [self.record(doc, f"{doc.doc_id}::PREAMBLE" + (f"::{n}" if len(parts) > 1 else ""), 2, "preamble",
                            ["Phần mở đầu"], part, [self.effective_note(doc.meta)], not doc.articles)
                for n, part in enumerate(parts, 1)]


def natural(value: str) -> tuple[int, str]:
    match = re.match(r"(\d+)(.*)", value)
    return (int(match.group(1)), match.group(2)) if match else (0, value)


# ---------------------------------------------------------------- chạy
def run(src: Path, out: Path, as_of: str, force: bool) -> None:
    # Chỉ ghi đè các file do bước này tạo; file khác trong thư mục (chunks.xlsx, eval/ ...) giữ nguyên.
    if (out / "chunks.jsonl").exists() and not force:
        sys.exit(f"{out} đã có chunks.jsonl; thêm --force để chia lại")
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        (out / name).unlink(missing_ok=True)
    docs = [parse_doc(group, folder) for group in GROUPS
            for folder in sorted(p for p in (src / group).iterdir() if p.is_dir())]
    amendments = analyse_amendments(docs)
    writer = Writer(docs, as_of)
    records, report = [], []
    for doc in docs:
        doc_records = [writer.doc_card(doc)] + writer.preamble_records(doc)
        for art in doc.articles:
            doc_records.extend(writer.article_records(doc, art))
        for index, seg in enumerate(doc.annexes, 1):
            doc_records.extend(writer.annex_records(doc, index, seg))
        records.extend(doc_records)
        sizes = [len(r["text"]) for r in doc_records if r["retrievable"] and r["level"] == 2]
        report.append({"group": doc.group, "folder": doc.folder, "doc_key": doc.meta["doc_key"], "role": doc.role,
                       "articles": len(doc.articles), "regulation_articles": sum(1 for a in doc.articles if a.part),
                       "annexes": len(doc.annexes), "chunks": len(doc_records),
                       "retrievable": sum(r["retrievable"] for r in doc_records),
                       "max_chars": max(sizes, default=0), "validity": doc_records[0]["validity_status"],
                       "warnings": "; ".join(dict.fromkeys(doc.warnings))})
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    for name, rows in (("chunk_report.csv", report), ("amendments.csv", amendments)):
        if rows:
            with (out / name).open("w", encoding="utf-8-sig", newline="") as handle:
                writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer_csv.writeheader()
                writer_csv.writerows(rows)
    sizes = sorted(len(r["text"]) for r in records if r["retrievable"] and r["level"] == 2)
    manifest = {
        "as_of": as_of, "source": str(src), "documents": len(docs), "chunks": len(records),
        "retrievable": sum(r["retrievable"] for r in records),
        "by_type": dict(Counter(r["chunk_type"] for r in records)),
        "retrievable_chars": {"p50": sizes[len(sizes) // 2], "p90": sizes[int(len(sizes) * .9)], "max": sizes[-1]},
        "amendment_links": len(amendments),
        "settings": {"target": TARGET, "max": MAX, "whole_article": WHOLE, "tail": TAIL, "lead_max": LEAD_MAX},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in report:
        if row["warnings"]:
            print(f"[cảnh báo] {row['group']}/{row['folder']}: {row['warnings']}")
    print(json.dumps(manifest, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chia chunk văn bản pháp luật cho lc_app.")
    parser.add_argument("--src", type=Path, default=ROOT / "data/data/processed_kb/clean_extracted")
    parser.add_argument("--out", type=Path, default=ROOT / "data/chunk/chunk_v1")
    parser.add_argument("--as-of", default=dt.date.today().isoformat(), help="ngày chốt KB để xét hiệu lực")
    parser.add_argument("--force", action="store_true", help="xóa và chia lại thư mục đích")
    args = parser.parse_args()
    run(args.src, args.out, args.as_of, args.force)


if __name__ == "__main__":
    main()
