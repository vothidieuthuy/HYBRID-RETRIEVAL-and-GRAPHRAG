"""Dựng processed_raw_kb: mỗi văn bản một thư mục gồm file Word và metadata.json rút gọn.

Nguồn: data/data/raw_kb/<nhóm>/<văn bản>/ (file Word + metadata.json đầy đủ).
Đích:  data/data/processed_kb/processed_raw_kb/<nhóm>/<văn bản>/ (file Word được chép nguyên, metadata.json theo SCHEMA dưới đây).

KB chỉ chứa văn bản còn hiệu lực nên metadata không cần các trường suy luận hiệu lực; chỉ giữ những trường
đi vào header chunk, dòng trích dẫn, bộ lọc phạm vi và quan hệ giữa các văn bản:
  amends             văn bản trong KB mà văn bản này sửa đổi, bổ sung
  guides             văn bản mà văn bản này quy định chi tiết / hướng dẫn thi hành (theo tiêu đề và metadata gốc)
  based_on           văn bản trong KB được nêu ở phần "Căn cứ ..." (vd QĐ Đà Nẵng căn cứ Luật Đất đai)
  replaces           văn bản cũ (ngoài KB) bị thay thế / bãi bỏ / hết hiệu lực theo điều khoản thi hành
  partial_effective  một số Điều/Khoản có ngày hiệu lực hoặc thời hạn riêng (trích từ điều khoản thi hành)
Quan hệ lấy từ trường relations của metadata gốc và từ chính file Word; verified = đã đối chiếu thủ công chưa.

    python -m kb_pipeline.build_metadata [--src DIR] [--out DIR] [--force]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("01_luat", "02_nghi_dinh", "03_thong_tu", "04_da_nang")
ROLES = ("main", "consolidated")

# Thứ tự trường trong metadata.json mới.
SCHEMA = ("doc_key", "document_number", "document_type", "title", "issuing_authority", "issue_date",
          "effective_date", "expiry_date", "scope", "is_normative", "amends", "guides", "based_on", "replaces",
          "partial_effective", "consolidated", "files", "source_url", "note", "verified")
REQUIRED = ("doc_key", "document_number", "document_type", "title", "issuing_authority", "issue_date",
            "effective_date", "scope", "files")

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Số hiệu văn bản; cho phép khoảng trắng lỗi soạn thảo quanh dấu gạch ("TT- BTNMT", "NĐ -CP").
DOC_NUMBER = re.compile(r"\b(\d{1,4})/(\d{4})/(QH\d{2}|(?:NĐ|NQ|TT|QĐ)\s*-\s*[A-ZĐ]+)")
# Tiêu đề Điều của văn bản sửa đổi: "Điều 1. Sửa đổi, bổ sung một số điều của Nghị định số 71/2024/NĐ-CP ..."
AMEND_WORDS = re.compile(r"\b(sửa đổi|bổ sung|bãi bỏ|thay thế)\b", re.IGNORECASE)
ARTICLE_HEADING = re.compile(r"^Điều\s+\d+[a-z]?\.\s")
# "Chương I" / "Mục 2": tên chương, mục thường nằm ở đoạn văn ngay sau.
PART_HEADING = re.compile(r"^(Chương|Mục)\s+[IVXLC\d]+\b", re.IGNORECASE)
W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
# Thứ bậc hiệu lực theo ký hiệu số hiệu (số nhỏ = cao hơn); văn bản chỉ sửa được văn bản cùng hoặc thấp hơn.
RANKS = (("/QH", 1), ("NĐ-CP", 3), ("/TT-", 5), ("NQ-HĐND", 6), ("QĐ-UBND", 7))


# Số hiệu có hoặc không có năm (văn bản hành chính "2932/QĐ-UBND").
ANY_NUMBER = re.compile(r"\b(\d{1,4})/(?:(\d{4})/)?(QH\d{2}|(?:NĐ|NQ|TT|QĐ)\s*-\s*[A-ZĐ]+)")
# Luật nhắc theo tên, không kèm số hiệu/năm, trong văn bản ban hành từ 2024: là luật hiện hành trong KB.
LAW_NAMES = {"Đất đai": "31/2024/QH15", "Nhà ở": "27/2023/QH15", "Kinh doanh bất động sản": "29/2023/QH15"}
LAW_NAME = re.compile(r"\bLuật (Đất đai|Nhà ở|Kinh doanh bất động sản)\b(?!\s*(?:năm\s*\d|số\s*\d|\d))")
GUIDE_TITLE = re.compile(r"quy định chi tiết|hướng dẫn|thi hành|biện pháp thi hành", re.IGNORECASE)
EFFECT_HEADING = re.compile(r"hiệu lực|thi hành|chuyển tiếp", re.IGNORECASE)
REPLACE_WORDS = re.compile(r"thay thế|hết hiệu lực|bãi bỏ", re.IGNORECASE)
PHRASE_ONLY = re.compile(r"thay thế (?:các |một số )?(?:từ|cụm từ)|bãi bỏ (?:các |một số )?(?:từ|cụm từ)", re.IGNORECASE)
VN_DATE = re.compile(r"ngày (\d{1,2}) tháng (\d{1,2}) năm (\d{4})|ngày (\d{1,2})/(\d{1,2})/(\d{4})")
THIS_DOC = r"(?:Luật|Nghị quyết|Nghị định|Thông tư|Quyết định|Quy định) này"
PARTIAL_FROM = re.compile(rf"^(?:[a-zđ]\)\s*|\d+\.\s*)?(?P<subj>.{{3,400}}?)\s(?:của|tại)\s{THIS_DOC}\s"
                          rf"có hiệu lực(?: thi hành)?(?: kể)? từ (?P<date>ngày .{{4,25}})")
PARTIAL_UNTIL = re.compile(rf"^(?:[a-zđ]\)\s*|\d+\.\s*)?(?P<subj>.{{3,400}}?)\s(?:của|tại)\s{THIS_DOC}\s"
                           r"hết hiệu lực(?P<rest>.{0,200})")
PARTIAL_RUN = re.compile(r"^(?:[a-zđ]\)\s*|\d+\.\s*)?(?P<subj>Các quy định.{3,400}?)\s(?:được )?thực hiện đến "
                         r"(?:hết |trước )?(?P<date>ngày .{4,25})")


def doc_number(match: re.Match) -> str:
    return f"{match.group(1)}/{match.group(2)}/{re.sub(r'\s+', '', match.group(3))}"


def any_number(match: re.Match) -> str:
    suffix = re.sub(r"\s+", "", match.group(3))
    return f"{match.group(1)}/{match.group(2)}/{suffix}" if match.group(2) else f"{match.group(1)}/{suffix}"


def numbers_in(text: str) -> list[str]:
    """Số hiệu văn bản nêu trong đoạn: số hiệu đầy đủ và luật nhắc theo tên."""
    found = [any_number(m) for m in ANY_NUMBER.finditer(text)]
    found += [LAW_NAMES[m.group(1)] for m in LAW_NAME.finditer(text)]
    return list(dict.fromkeys(found))


def iso_date(text: str) -> str | None:
    match = VN_DATE.search(text or "")
    if not match:
        return None
    day, month, year = (match.group(1), match.group(2), match.group(3)) if match.group(1) \
        else (match.group(4), match.group(5), match.group(6))
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def rank(number: str) -> int | None:
    return next((value for marker, value in RANKS if marker in number), None)


def docx_paragraphs(path: Path) -> list[str]:
    """Đoạn văn trong word/document.xml (đủ để đọc tiêu đề văn bản và tiêu đề Điều)."""
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for para in root.iter(f"{W_NS}p"):
        text = "".join(node.text or "" for node in para.iter(f"{W_NS}t")).strip()
        if text:
            paragraphs.append(re.sub(r"\s+", " ", text))
    return paragraphs


def amended_numbers(own_number: str, title: str, paragraphs: list[str]) -> list[str]:
    """Văn bản bị sửa đổi/bãi bỏ: số hiệu trong tiêu đề văn bản (khi tiêu đề là 'Sửa đổi...'), tiêu đề Điều
    và tiêu đề Chương/Mục (kèm đoạn tên ngay sau) có từ sửa đổi, bổ sung, bãi bỏ, thay thế."""
    found: list[str] = []
    sources = [title] if re.match(r"^(Luật )?sửa đổi|^bãi bỏ", title, re.IGNORECASE) else []
    for index, para in enumerate(paragraphs):
        if ARTICLE_HEADING.match(para):
            sources.append(para)
        elif PART_HEADING.match(para):
            sources.append(" ".join(paragraphs[index:index + 2]))
    for text in (s for s in sources if AMEND_WORDS.search(s)):
        for match in DOC_NUMBER.finditer(text):
            number = doc_number(match)
            if number != own_number and number not in found:
                found.append(number)
    return found


def body_paragraphs(paragraphs: list[str]) -> list[str]:
    """Phần thân văn bản: tới bảng 'Nơi nhận' đầu tiên (bỏ phụ lục, mẫu biểu có điều khoản mẫu)."""
    end = next((i for i, p in enumerate(paragraphs) if p.startswith("Nơi nhận")), len(paragraphs))
    return paragraphs[:end]


def guided_numbers(title: str, relations: list[dict]) -> list[str]:
    found = [r["target_number"] for r in relations if r["type"] == "GUIDES"]
    if GUIDE_TITLE.search(title):
        found += numbers_in(title)
    return list(dict.fromkeys(found))


def basis_numbers(paragraphs: list[str]) -> list[str]:
    """Số hiệu ở các dòng 'Căn cứ ...' trước Điều đầu tiên."""
    first = next((i for i, p in enumerate(paragraphs) if ARTICLE_HEADING.match(p)), len(paragraphs))
    found: list[str] = []
    for para in paragraphs[:first]:
        if para.startswith("Căn cứ"):
            found += numbers_in(para)
    return list(dict.fromkeys(found))


def effect_paragraphs(paragraphs: list[str]) -> list[str]:
    """Đoạn thuộc Điều hiệu lực / thi hành / chuyển tiếp, hoặc chứa 'có hiệu lực' (Quyết định không có tên Điều)."""
    out, in_effect = [], False
    for para in body_paragraphs(paragraphs):
        if ARTICLE_HEADING.match(para):
            in_effect = bool(EFFECT_HEADING.search(para))
        if (in_effect or "có hiệu lực" in para) and not para.startswith(("“", '"')):
            out.append(para)
    return out


def replaced_numbers(paragraphs: list[str], relations: list[dict]) -> list[str]:
    found = [r["target_number"] for r in relations if r["type"] == "REPLACES"]
    in_list = False
    for para in effect_paragraphs(paragraphs):
        # "Các Thông tư sau đây hết hiệu lực:" rồi liệt kê "a) Thông tư số ...; b) ...": lấy số hiệu đầu tiên
        # của mỗi dòng liệt kê (số sau đó thường là văn bản mà văn bản bị thay thế từng sửa đổi)
        if in_list and re.match(r"^([a-zđ]\)|[-–+])\s", para):
            match = ANY_NUMBER.search(para)
            if match:
                found.append(any_number(match))
            continue
        in_list = bool(REPLACE_WORDS.search(para)) and para.rstrip().endswith(":")
        if not REPLACE_WORDS.search(para) or PHRASE_ONLY.search(para):
            continue
        for match in ANY_NUMBER.finditer(para):
            # "Luật số 65/2014/QH13 đã được sửa đổi, bổ sung bởi Luật số 40/2019/QH14 ... hết hiệu lực":
            # số đứng sau "sửa đổi, bổ sung" trong cùng vế câu (tới dấu ";") là văn bản sửa đổi, không bị thay thế
            clause = para[max(0, match.start() - 150):match.start()].rsplit(";", 1)[-1]
            if not re.search(r"sửa đổi|bổ sung", clause):
                found.append(any_number(match))
    return list(dict.fromkeys(found))


def partial_effective(paragraphs: list[str]) -> list[dict]:
    """Điều/Khoản có ngày hiệu lực hoặc thời hạn riêng, vd 'Điều 190 và Điều 248 của Luật này có hiệu lực ...'."""
    found = []
    for para in effect_paragraphs(paragraphs):
        for pattern, key in ((PARTIAL_FROM, "from"), (PARTIAL_RUN, "until"), (PARTIAL_UNTIL, "until")):
            match = pattern.match(para)
            if not match or not re.search(r"\b(Điều|điều|khoản|Chương|Mục|quy định)\b", match.group("subj")):
                continue
            subject = match.group("subj").strip(" ,;:")
            date = iso_date(match.groupdict().get("date") or match.groupdict().get("rest") or "")
            found.append({"provisions": subject, key: date, "source": para[:300]})
            break
    return found


def consolidated_info(file_entry: dict) -> dict | None:
    if not file_entry.get("document_number"):
        return {"number": None, "date": None, "consolidates": [], "official": False}
    return {"number": file_entry["document_number"], "date": file_entry.get("issue_date"),
            "consolidates": file_entry.get("consolidates", []), "official": True}


def convert(src_meta: dict, group: str, folder: Path) -> tuple[dict, list[str]]:
    """metadata.json đầy đủ -> metadata rút gọn; trả kèm danh sách cảnh báo."""
    warnings: list[str] = []
    files = [{"file": f["file"], "role": f["role"]} for f in src_meta["files"]]
    for entry in files:
        if entry["role"] not in ROLES:
            warnings.append(f"role lạ: {entry['role']}")
        if not (folder / entry["file"]).exists():
            warnings.append(f"thiếu file {entry['file']}")
    hn = next((f for f in src_meta["files"] if f["role"] == "consolidated"), None)

    number = src_meta.get("document_number_display") or src_meta["document_number"]
    title = src_meta["title"].strip()
    paragraphs = docx_paragraphs(folder / files[0]["file"]) if files and (folder / files[0]["file"]).exists() else []
    relations = src_meta.get("relations", [])
    amends = [r["target_number"] for r in relations if r["type"] == "AMENDS"]
    amends += [n for n in amended_numbers(number, title, paragraphs) if n not in amends]
    own = {number, src_meta["document_number"]}
    guides = [n for n in guided_numbers(title, relations) if n not in own]
    based_on = [n for n in basis_numbers(paragraphs) if n not in own]
    # văn bản đang được sửa đổi (kể cả ngoài KB, vd NĐ 71/2024) chỉ bị bãi bỏ một phần, không bị thay thế
    replaces = [n for n in replaced_numbers(paragraphs, relations) if n not in own and n not in amends]

    source = (src_meta.get("source_page_url") or next(iter(src_meta.get("official_file_urls") or []), None)
              or next((f.get("source_url") for f in src_meta["files"] if f.get("source_url")), None))
    evidence = src_meta.get("date_evidence") or {}
    note = "; ".join(f"{key}: {value}" for key, value in evidence.items()) or None

    meta = {
        "doc_key": src_meta["doc_key"],
        "document_number": number,
        "document_type": src_meta["document_type"],
        "title": title,
        "issuing_authority": src_meta["issuing_authority"],
        "issue_date": src_meta.get("issue_date"),
        "effective_date": src_meta.get("effective_date"),
        "expiry_date": src_meta.get("expiry_date"),
        "scope": "da_nang" if group == "04_da_nang" else "toan_quoc",
        "is_normative": src_meta.get("is_normative", True),
        "amends": amends,
        "guides": guides,
        "based_on": based_on,
        "replaces": replaces,
        "partial_effective": partial_effective(paragraphs),
        "consolidated": consolidated_info(hn) if hn else None,
        "files": files,
        "source_url": source,
        "note": note,
        "verified": bool(src_meta.get("verified", False)),
    }
    for key in REQUIRED:
        if meta[key] in (None, "", []):
            warnings.append(f"thiếu {key}")
    for key in ("issue_date", "effective_date", "expiry_date"):
        if meta[key] and not DATE.match(meta[key]):
            warnings.append(f"{key} sai định dạng: {meta[key]}")
    if meta["consolidated"] and not meta["consolidated"]["official"]:
        warnings.append("bản hợp nhất không phải VBHN chính thức")
    return {key: meta[key] for key in SCHEMA}, warnings


def build(src: Path, out: Path, force: bool) -> int:
    if out.exists() and any(out.iterdir()):
        if not force:
            sys.exit(f"{out} đã có dữ liệu; thêm --force để dựng lại")
        if not (out / "catalog.csv").exists():
            sys.exit(f"{out} không phải thư mục do pipeline tạo (thiếu catalog.csv); không xóa")
        shutil.rmtree(out)
    docs, keys = [], {}
    for group in GROUPS:
        for folder in sorted(p for p in (src / group).iterdir() if p.is_dir()):
            src_meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            meta, warnings = convert(src_meta, group, folder)
            if meta["doc_key"] in keys:
                warnings.append(f"trùng doc_key với {keys[meta['doc_key']]}")
            keys[meta["doc_key"]] = f"{group}/{folder.name}"
            docs.append((group, folder, meta, warnings))
    # Chỉ giữ văn bản bị sửa đổi có trong KB và không cao hơn văn bản sửa đổi
    # (bỏ "sửa đổi các luật có liên quan" ngoài KB và số hiệu căn cứ bị bắt nhầm như NQ Quốc hội trong nghị định).
    in_kb = {meta["document_number"] for _, _, meta, _ in docs}
    for _, _, meta, _ in docs:
        own = rank(meta["document_number"]) or 99
        meta["amends"] = [n for n in meta["amends"] if n in in_kb and (rank(n) or 0) >= own]
        # văn bản được hướng dẫn phải cao hơn văn bản hướng dẫn (NĐ hướng dẫn Luật, không ngược lại)
        meta["guides"] = [n for n in meta["guides"] if (rank(n) or 99) < own]
        # căn cứ: chỉ giữ văn bản có trong KB (bỏ Luật Tổ chức Chính phủ, Luật Ban hành VBQPPL ...)
        meta["based_on"] = [n for n in meta["based_on"] if n in in_kb and n not in meta["guides"]]
        # văn bản bị thay thế đã hết hiệu lực nên không thể nằm trong KB; nếu có thì đó là bãi bỏ một phần (sửa đổi)
        # và không thể cao hơn văn bản thay thế (nghị định không thay thế luật)
        meta["replaces"] = [n for n in meta["replaces"] if n not in in_kb and (rank(n) or 99) >= own]
    rows = []
    for group, folder, meta, warnings in docs:
        target = out / group / folder.name
        target.mkdir(parents=True)
        for entry in meta["files"]:
            if (folder / entry["file"]).exists():
                shutil.copy2(folder / entry["file"], target / entry["file"])
        (target / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
        rows.append({"group": group, "folder": folder.name, **{k: meta[k] for k in (
            "doc_key", "document_number", "document_type", "issuing_authority", "issue_date",
            "effective_date", "expiry_date", "scope", "is_normative")},
            "amends": "; ".join(meta["amends"]), "guides": "; ".join(meta["guides"]),
            "based_on": "; ".join(meta["based_on"]), "replaces": "; ".join(meta["replaces"]),
            "partial_effective": " | ".join(f"{p['provisions'][:80]} -> {p.get('from') or p.get('until')}"
                                            for p in meta["partial_effective"]),
            "files": "; ".join(f"{f['file']}({f['role']})" for f in meta["files"]),
            "title": meta["title"], "warnings": "; ".join(warnings)})
    # Danh mục để rà soát bằng Excel (utf-8-sig để Excel hiển thị đúng tiếng Việt).
    with (out / "catalog.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["warnings"]:
            print(f"[cảnh báo] {row['group']}/{row['folder']}: {row['warnings']}")
    print(f"Đã dựng {len(rows)} văn bản vào {out}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dựng processed_raw_kb với metadata rút gọn.")
    parser.add_argument("--src", type=Path, default=ROOT / "data/data/raw_kb")
    parser.add_argument("--out", type=Path, default=ROOT / "data/data/processed_kb/processed_raw_kb")
    parser.add_argument("--force", action="store_true", help="xóa và dựng lại thư mục đích")
    args = parser.parse_args()
    build(args.src, args.out, args.force)


if __name__ == "__main__":
    main()
