"""Bước 2: gộp kết quả trích xuất thành một đồ thị (như bước merge/summarize của Microsoft GraphRAG).

- Chuẩn hóa tên (chỉ đổi TÊN thực thể, không tạo quan hệ): dấu thanh kiểu mới ("Uỷ" -> "Ủy"); bỏ chữ "số" sau loại
  văn bản; tên Điều bỏ phần khoản/điểm;
  tên Điều chép theo dòng tiêu đề ("Điều 220. Tách thửa đất, ...") quy về "Điều 220 <tên văn bản>" trích từ cùng chunk;
  tên văn bản không số hiệu được gộp vào tên có số hiệu khi trong đồ thị chỉ có ĐÚNG MỘT ứng viên cùng tên gọi, hoặc
  nhiều ứng viên nhưng chỉ một là văn bản của KB (nguồn cạnh CHUA) ("Luật Đất đai" -> "Luật Đất đai 31/2024/QH15", không
  về 45/2013/QH13), áp dụng cả cho tên Điều chứa tên văn bản đó. Chỉ dựa vào các tên LLM đã trích, không dùng metadata.
- Gộp nút theo tên (không phân biệt hoa/thường), loại = loại được trích nhiều nhất; gộp cạnh theo (nguồn, đích, loại),
  giữ hướng; độ mạnh cộng dồn. Mô tả trùng hoặc nằm trọn trong mô tả khác bị bỏ; còn từ hai mô tả trở lên thì LLM gộp
  thành một đoạn (như summarize_descriptions của Microsoft), không thêm thông tin ngoài.
- Tên Điều trong đoạn văn thuộc "Quy định kèm theo" quy về "Điều N Quy định kèm theo <văn bản>" theo dòng tiêu đề LLM
  đã đọc (gpt-4o-mini hay bỏ cụm này).
- Bỏ và đếm: nút loại lạ, nút chung chung, cạnh loại lạ, cạnh có đầu mút bị bỏ, cạnh tự nối, cạnh sai miền/đích của
  ontology (DOMAIN_RANGE).

    .venv-lc/Scripts/python -m lc_app.graph.build
Đầu ra: <graph_dir>/graph.json (nodes, edges, stats).
"""
from __future__ import annotations

import json
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI

from ..config import Config
from .extract import full_header, unit_text
from .ontology import DOC_KINDS, DOCUMENT_NAME, DOMAIN_RANGE, ENTITY_TYPES, GENERIC_NAMES, RELATION_TYPES, UNIT_NAME

SO =re.compile(rf"\b{DOC_KINDS}\s+số\s+")
NUMBER = re.compile(r"\d+/\d{4}/[\wĐđ-]+")
PART = r"(?:[Kk]hoản|[Đđ]iểm)\s+\S+"
PART_BEFORE = re.compile(rf"^(?:{PART}\s+)+(?=Điều\b)")                      # "Khoản 3 Điều 18 X" -> "Điều 18 X"
PART_AFTER = re.compile(rf"^(Điều\s+\d+[a-z]?)(?:\s+{PART}|\.\d+(?=\s))+")  # "Điều 220 Khoản 3 X", "Điều 220.1 X"
DIEU = re.compile(r"^Điều\s+\d+[a-z]?\s+")
TITLED = re.compile(r"^Điều\s+(\d+[a-z]?)\.\s")
SUMMARY_PROMPT = ("Bạn là trợ lý tóm tắt văn bản pháp luật. Gộp các mô tả sau về cùng một {kind} \"{name}\" thành MỘT "
                  "đoạn mô tả mạch lạc, viết ở ngôi thứ ba, không lặp ý. Chỉ dùng thông tin có trong các mô tả; không "
                  "thêm hiểu biết bên ngoài, không tự định nghĩa. Giữ nguyên số hiệu văn bản, số Điều, khoản, số liệu. "
                  "Chỉ trả về đoạn mô tả.\n\nCác mô tả:\n{descs}")
NON_WORD = re.compile(r"\W+")
OLD_TONE = re.compile(r"(?<![qQ])([oOuU])([àáảãạèéẻẽẹỳýỷỹỵÀÁẢÃẠÈÉẺẼẸỲÝỶỸỴ])(?!\w)")
SUMMARY_INPUT_CHARS = 12000   # ~3.000 token mỗi lần gộp (Microsoft: max_input_tokens); dài hơn thì gộp theo đợt


def strip_number(name: str) -> str:
    return re.sub(rf"\s+(?:số\s+)?{NUMBER.pattern}", "", name).strip()


def cite(name: str, text: str) -> str | None:
    """Đối chiếu tên LLM ghi (trước khi gộp bí danh) với đoạn văn nó đọc. Số Điều không có trong đoạn văn (kể cả dạng
    liệt kê "các Điều 17, 18 và 19") -> None, bỏ ("khoản 3 Điều này" bị ghi "Điều 3"). Số hiệu văn bản không có trong
    đoạn văn -> bỏ số hiệu khỏi tên: "Luật Đất đai 31/2024/QH15" (đoạn văn chỉ ghi "Luật Đất đai") gộp lại qua bí danh,
    số hiệu bịa ("Nghị định 104/2024/NĐ-CP" khi đoạn văn ghi 103) còn "Nghị định", bị lọc vì không nêu đích danh."""
    article = re.match(r"Điều (\d+[a-z]?)\b", name)
    if article and not re.search(rf"\bĐiều\s+(?:\d+[a-z]?\s*(?:,|và|hoặc)\s*)*{article.group(1)}\b", text):
        return None
    number = NUMBER.search(name)
    return name if not number or number.group(0) in text else strip_number(name)


def grounded(name: str, description: str, texts: list[str]) -> bool:
    """Mô tả "quy định về <chủ đề>" của một Điều chỉ được nhắc tới có căn cứ khi chủ đề nằm cùng mệnh đề (tách theo
    ; : xuống dòng) với số Điều trong đoạn văn nguồn. "..., gồm: khoản 4 Điều 50, khoản 1 Điều 55" -> không căn cứ."""
    topic = re.search(r"quy định (?:về|chi tiết về)?\s*([^,.;]+)", description or "")
    number = re.match(r"Điều (\d+[a-z]?)", name)
    if not topic or not number:
        return True
    words = " ".join(topic.group(1).casefold().split()[:3])
    clauses = [" ".join(c.casefold().split()) for t in texts for c in re.split(r"[;:\n]", t)
               if re.search(rf"\bĐiều {number.group(1)}\b", c)]
    return any(words in c for c in clauses)


def distinct(descriptions: list[str]) -> list[str]:
    """Bỏ mô tả trùng hoặc nằm trọn trong mô tả khác (sau khi bỏ dấu câu, chữ hoa)."""
    keys = [NON_WORD.sub(" ", d.casefold()).strip() for d in descriptions]
    return [d for i, d in enumerate(descriptions)
            if keys[i] and not any(j != i and keys[i] in keys[j] and (keys[i] != keys[j] or j < i)
                                   for j in range(len(keys)))]


def new_tone(name: str) -> str:
    """Dấu thanh kiểu cũ ở vần mở về kiểu mới mà KB dùng chủ yếu ("Uỷ ban" -> "Ủy ban", "hoá" -> "hóa"); "quý", "hoàn"
    không đổi (sau q, hoặc vần có âm cuối)."""
    def move(match: re.Match) -> str:
        second = unicodedata.normalize("NFD", match.group(2))
        return unicodedata.normalize("NFC", match.group(1) + second[1:]) + second[0]
    return OLD_TONE.sub(move, unicodedata.normalize("NFC", name))


def tidy(name: str) -> str:
    name = SO.sub(lambda m: m.group(1) + " ", re.sub(r"\s+", " ", new_tone(name)).strip())
    name = re.sub(rf"\s+số\s+(?={NUMBER.pattern})", " ", name)   # "Luật Đất đai số 31/2024/QH15"
    name = re.sub(r"^Phụ lục số\s+", "Phụ lục ", name)            # "Phụ lục số 05 Thông tư ..."
    return PART_AFTER.sub(r"\1", PART_BEFORE.sub("", name)).strip()


def alias_map(names: set[str], owners: set[str]) -> dict[str, str]:
    """Tên văn bản không số hiệu -> tên có số hiệu: ứng viên duy nhất; có nhiều ứng viên (Luật Đất đai 31/2024 và
    45/2013 hết hiệu lực được nhắc tới) thì chọn ứng viên duy nhất là văn bản của KB (nguồn của cạnh CHUA)."""
    documents = {DIEU.sub("", n).removeprefix("Quy định kèm theo ").strip() for n in names}
    numbered = [d for d in documents if NUMBER.search(d)]
    aliases = {}
    for doc in documents - set(numbered):
        # "Luật Đất đai năm 2024", "Luật Đất đai 2024": chỉ ứng viên có năm trong số hiệu trùng năm đó
        dated = re.fullmatch(r"(.+?)\s+(?:năm\s+)?(\d{4})", doc)
        base, year = dated.groups() if dated else (doc, None)
        candidates = [d for d in numbered if d.startswith(base + " ") and NUMBER.fullmatch(d[len(base) + 1:])
                      and (year is None or d[len(base) + 1:].split("/")[1] == year)]
        in_kb = [d for d in candidates if d in owners]
        if len(candidates) == 1 or len(in_kb) == 1:
            aliases[doc] = candidates[0] if len(candidates) == 1 else in_kb[0]
    return aliases


def resolve(name: str, aliases: dict[str, str]) -> str:
    for short, full in aliases.items():   # chỉ tên văn bản và tên Điều, không đụng tên khái niệm
        if name == short or (DIEU.match(name) and name.endswith(" " + short)):
            return name[: len(name) - len(short)] + full
    return name


def titled_articles(record: dict) -> dict[str, str]:
    """"Điều 220. Tách thửa đất, ..." (LLM chép dòng tiêu đề) -> "Điều 220 <tên văn bản>" trích từ cùng chunk."""
    names = {tidy(e["name"]) for e in record["entities"]} | {tidy(x[k]) for x in record["relations"]
                                                             for k in ("source", "target")}
    fixes = {}
    for name in names:
        match = TITLED.match(name)
        if match:
            proper = [n for n in names if n != name and n.startswith(f"Điều {match.group(1)} ")
                      and not TITLED.match(n)]
            if len(proper) == 1:
                fixes[name] = proper[0]
    return fixes


AMEND = r"(?:đã\s+)?(?:quy định\s+)?(?:việc\s+)?(?:sửa đổi|bổ sung|bãi bỏ|thay thế)\b"


def forms(name: str) -> set[str]:
    """Cách mô tả gọi một thực thể: tên đầy đủ, hoặc "Điều N" (mô tả hay viết tắt tên Điều)."""
    name = " ".join(name.split()).casefold()
    short = re.match(r"điều \d+[a-z]?\b", name)
    return {name, short.group(0)} if short else {name}


def amends(description: str, subject: str) -> bool:
    """Mô tả mở đầu bằng "<subject> (quy định) sửa đổi/bổ sung/bãi bỏ/thay thế ..." (chủ thể đi sửa là subject)."""
    return any(re.match(rf"{re.escape(f)}\s+{AMEND}", description) for f in forms(subject))


def passive(relation: dict) -> bool:
    """Mô tả viết "<nguồn> (đã) được sửa đổi, bổ sung ... bởi/tại <đích>": nguồn mới là văn bản bị sửa."""
    desc = " ".join(relation["description"].split()).casefold()
    source, target = (" ".join(relation[k].split()).casefold() for k in ("source", "target"))
    return bool(re.search(rf"{re.escape(source)}\s*(?:\([^)]*\)\s*)?(?:đã\s+)?được\s+(?:sửa đổi|bổ sung|thay thế|bãi bỏ)"
                          rf"[^.;]*?\b(?:bởi|tại|theo)\s+(?:các\s+)?{re.escape(target)}", desc))


def align(relation: dict) -> str | None:
    """Cho cạnh khớp với câu mô tả chính LLM viết cho nó (gpt-4o-mini ghi mô tả đúng nhưng hay đặt ngược nguồn/đích hoặc
    gán THAM_CHIEU cho câu sửa đổi). Chỉ so với tên LLM ghi ở nguồn, đích; không đọc lại đoạn văn, không tạo cạnh mới.
    Trả về lý do khi đã sửa."""
    desc = " ".join(relation["description"].split()).casefold()
    if relation["type"] == "SUA_DOI" and (passive(relation) or (amends(desc, relation["target"])
                                                                and not amends(desc, relation["source"]))):
        relation["source"], relation["target"] = relation["target"], relation["source"]
        return "đổi chiều SUA_DOI theo mô tả"
    if relation["type"] == "THAM_CHIEU" and amends(desc, relation["source"]):
        relation["type"] = "SUA_DOI"
        return "THAM_CHIEU có mô tả sửa đổi -> SUA_DOI"
    return None


def annexed_articles(record: dict, header: str) -> dict[str, str]:
    """Đoạn văn thuộc Quy định kèm theo một văn bản (dòng tiêu đề LLM đọc có "> Quy định kèm theo >"): Điều của chính văn
    bản đó ghi "Điều N Quy định kèm theo <tên văn bản>" (quy tắc đặt tên 3), kể cả khi LLM bỏ cụm này ("Điều 18 Quyết định
    135/2026/QĐ-UBND") hoặc viết tắt ("Điều 6 Quy định 75/2026/QĐ-UBND"). Tên văn bản lấy từ đầu dòng tiêu đề."""
    own = re.match(rf"\[{DOC_KINDS}\s+({NUMBER.pattern})", header)
    if "> Quy định kèm theo >" not in header or not own:
        return {}
    kind, number = own.groups()
    pattern = re.compile(rf"^Điều\s+(\d+[a-z]?)\s+(?:Quy định\s+(?:kèm theo\s+)?)?(?:{kind}\s+)?{re.escape(number)}$")
    names = {tidy(e["name"]) for e in record["entities"]} | {tidy(x[k]) for x in record["relations"]
                                                             for k in ("source", "target")}
    return {name: f"Điều {m.group(1)} Quy định kèm theo {kind} {number}" for name in names
            if (m := pattern.match(name)) and "Quy định kèm theo" not in name}


def build(config: Config) -> dict:
    extractions = [json.loads(line) for line in (config.graph_dir / "extractions.jsonl").open(encoding="utf-8")]
    records = {}
    with config.chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[record["chunk_id"]] = record
    stats = Counter()
    for r in extractions:   # sửa tên Điều dạng tiêu đề, tên Điều của Quy định kèm theo trước khi gộp
        titled, annexed = titled_articles(r), annexed_articles(r, full_header(records[r["chunk_id"]]))
        stats["tên Điều dạng tiêu đề được sửa"] += len(titled)
        stats["tên Điều Quy định kèm theo được sửa"] += len(annexed)
        fixes = {**titled, **{k: annexed.get(tidy(v), v) for k, v in titled.items()}, **annexed}
        for e in r["entities"]:
            e["name"] = fixes.get(tidy(e["name"]), e["name"])
        for x in r["relations"]:
            x["source"], x["target"] = (fixes.get(tidy(x["source"]), x["source"]),
                                        fixes.get(tidy(x["target"]), x["target"]))
    names = {tidy(e["name"]) for r in extractions for e in r["entities"]}
    names |= {tidy(x[k]) for r in extractions for x in r["relations"] for k in ("source", "target")}
    names |= {strip_number(n) for n in names   # tên văn bản, tên Điều bị bỏ số hiệu ở cite() cũng gộp được
              if NUMBER.search(n) and re.match(rf"(?:Điều \S+ (?:Quy định kèm theo )?)?{DOC_KINDS}\b", n)}
    owners = {tidy(x["source"]) for r in extractions for x in r["relations"] if x["type"] == "CHUA"}
    aliases = alias_map(names, owners)
    texts = {r["chunk_id"]: unit_text(records[r["chunk_id"]]) for r in extractions}   # đúng đoạn văn LLM đã đọc

    def key(raw: str, chunk_id: str) -> str | None:
        name = cite(tidy(raw), texts[chunk_id])
        if name is not None and name != tidy(raw):
            stats["bỏ số hiệu không có trong đoạn văn khỏi tên"] += 1
        return None if name is None else resolve(name, aliases)

    nodes: dict[str, dict] = {}
    for r in extractions:
        for e in r["entities"]:
            name = key(e["name"], r["chunk_id"])
            if name is None:
                stats["bỏ thực thể có số Điều không có trong đoạn văn"] += 1
                continue
            if e["type"] not in ENTITY_TYPES:
                stats["bỏ nút loại lạ"] += 1
                continue
            if name.casefold() in GENERIC_NAMES:
                stats["bỏ nút chung chung"] += 1
                continue
            if e["type"] == "DIEU_KHOAN" and not UNIT_NAME.match(name):   # Chương, Mục, "Cụm từ ...", "Điều 69"
                stats["bỏ Điều/Phụ lục tên sai dạng"] += 1
                continue
            if e["type"] == "VAN_BAN" and not DOCUMENT_NAME.search(name):   # "nghị định của Chính phủ"
                stats["bỏ văn bản không nêu đích danh"] += 1
                continue
            if e["type"] == "VAN_BAN" and re.match(r"(Điều|Phụ lục)\b", name):   # Phụ lục bị gán nhầm loại văn bản
                stats["bỏ Điều/Phụ lục gán loại VAN_BAN"] += 1
                continue
            node = nodes.setdefault(name.casefold(), {"name": name, "types": Counter(), "descriptions": [],
                                                      "chunk_ids": []})
            node["types"][e["type"]] += 1
            if e["description"] and e["description"] not in node["descriptions"]:
                node["descriptions"].append(e["description"])
            if r["chunk_id"] not in node["chunk_ids"]:
                node["chunk_ids"].append(r["chunk_id"])

    kind_of = lambda key_: nodes[key_]["types"].most_common(1)[0][0]   # noqa: E731
    edges: dict[tuple, dict] = {}
    for r in extractions:
        for x in r["relations"]:
            reason = align(x)   # chính mô tả LLM nói khác cặp nguồn, đích hoặc loại nó ghi
            if reason:
                stats[reason] += 1
            source, target = key(x["source"], r["chunk_id"]), key(x["target"], r["chunk_id"])
            if x["type"] not in RELATION_TYPES:
                stats["bỏ cạnh loại lạ"] += 1
            elif source is None or target is None:
                stats["bỏ cạnh có số Điều không có trong đoạn văn"] += 1   # kể cả khi nút có thật ở chunk khác
            elif (source := source.casefold()) not in nodes or (target := target.casefold()) not in nodes:
                stats["bỏ cạnh thiếu đầu mút"] += 1
            elif source == target:
                stats["bỏ cạnh tự nối"] += 1
            elif not (kind_of(source) in DOMAIN_RANGE[x["type"]][0] and kind_of(target) in DOMAIN_RANGE[x["type"]][1]):
                stats["bỏ cạnh sai miền/đích"] += 1   # ontology cố định: vd THAM_CHIEU Điều -> khái niệm
            else:
                edge = edges.setdefault((source, target, x["type"]), {"descriptions": [], "weight": 0.0,
                                                                      "chunk_ids": []})
                edge["weight"] += x["weight"]
                if x["description"] and x["description"] not in edge["descriptions"]:
                    edge["descriptions"].append(x["description"])
                if r["chunk_id"] not in edge["chunk_ids"]:
                    edge["chunk_ids"].append(r["chunk_id"])

    local = threading.local()   # mỗi luồng một ChatOpenAI: dùng chung một đối tượng giữa các luồng lỗi ở lần gọi đầu

    def llm() -> ChatOpenAI:
        if not hasattr(local, "llm"):
            local.llm = ChatOpenAI(model=config.graph_llm_model, temperature=0, max_retries=8)
        return local.llm

    def summarize(job: tuple[str, str, list[str]]) -> str:
        # như summarize_descriptions của Microsoft: một mô tả thì giữ, nhiều mô tả thì LLM gộp; quá dài thì gộp theo
        # đợt (mỗi đợt <= SUMMARY_INPUT_CHARS) rồi gộp tiếp các bản tóm tắt
        kind, name, descriptions = job
        descriptions = distinct(descriptions)
        while len(descriptions) > 1:
            groups, size = [[]], 0
            for d in descriptions:
                if len(groups[-1]) >= 2 and size + len(d) > SUMMARY_INPUT_CHARS:   # >= 2: mỗi vòng giảm số mô tả
                    groups.append([])
                    size = 0
                groups[-1].append(d)
                size += len(d)
            descriptions = [g[0] if len(g) == 1 else llm().invoke(SUMMARY_PROMPT.format(
                kind=kind, name=name, descs="\n".join(f"- {d}" for d in g))).content.strip() for g in groups]
        return descriptions[0] if descriptions else ""

    # Điều chỉ được nhắc tới (không là đích của CHUA): bỏ mô tả đoán nội dung, thay bằng câu trung tính
    contained = {t for (_, t, kind) in edges if kind == "CHUA"}
    citers = defaultdict(list)
    for (s, t, kind) in edges:
        if kind != "CHUA" and nodes[s]["name"] not in citers[t]:
            citers[t].append(nodes[s]["name"])
    for key_, node in nodes.items():
        if node["types"].most_common(1)[0][0] != "DIEU_KHOAN" or key_ in contained:
            continue
        sources = [texts[c] for c in node["chunk_ids"] if c in texts]
        kept = [d for d in node["descriptions"] if grounded(node["name"], d, sources)]
        if len(kept) < len(node["descriptions"]):
            stats["mô tả đoán nội dung bị thay"] += 1
            node["descriptions"] = kept or [f"Được {', '.join(citers[key_]) or 'đoạn văn'} dẫn chiếu"]

    jobs = [("thực thể", n["name"], n["descriptions"]) for n in nodes.values()]
    jobs += [("quan hệ", f"{nodes[s]['name']} -> {nodes[t]['name']}", e["descriptions"]) for (s, t, _), e in edges.items()]
    stats["mô tả được LLM gộp"] = sum(len(distinct(d)) > 1 for _, _, d in jobs)
    with ThreadPoolExecutor(max_workers=8) as pool:
        summaries = list(pool.map(summarize, jobs))
    graph = {
        "nodes": [{"id": n["name"], "type": n["types"].most_common(1)[0][0], "description": desc,
                   "chunk_ids": n["chunk_ids"]} for n, desc in zip(nodes.values(), summaries)],
        "edges": [{"source": nodes[s]["name"], "target": nodes[t]["name"], "type": kind, "description": desc,
                   "weight": e["weight"], "chunk_ids": e["chunk_ids"]}
                  for ((s, t, kind), e), desc in zip(edges.items(), summaries[len(nodes):])],
        "aliases": aliases,
    }
    graph["stats"] = {"text_units": len(extractions), "nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
                      "nodes_by_type": dict(Counter(n["type"] for n in graph["nodes"])),
                      "edges_by_type": dict(Counter(e["type"] for e in graph["edges"])), **stats}
    return graph


def main() -> None:
    config = Config.from_env()
    set_llm_cache(SQLiteCache(database_path=str(config.graph_cache)))
    graph = build(config)
    out = config.graph_dir / "graph.json"
    out.write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"aliases": graph["aliases"], **graph["stats"]}, ensure_ascii=False, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
