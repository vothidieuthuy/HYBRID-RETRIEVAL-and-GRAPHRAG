"""Phần xử lý của giao diện: nạp retriever của lc_app, sinh câu trả lời theo chế độ và đối tượng, dựng đồ thị con để vẽ.

Không viết lại logic truy hồi: dùng nguyên lc_app.retrievers, lc_app.chain (prompt chốt, ngân sách 8.000 ký tự) và
lc_app.graph.global_search; chỉ thêm một đoạn hướng dẫn trình bày theo đối tượng ở cuối prompt.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))   # thêm cuối: thư mục gốc có thư mục "streamlit" trùng tên gói

import streamlit as st  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from lc_app.chain import INSTRUCTIONS, NO_CONTEXT, cut_to_budget, format_context  # noqa: E402
from lc_app.config import Config  # noqa: E402
from lc_app.loader import citation  # noqa: E402
from lc_app.retrievers import Retrievers  # noqa: E402

MODES = {
    "graph-augmented-hybrid": {"label": "Lai tăng cường đồ thị", "badge": "Khuyến nghị", "icon": ":material/hub:",
                               "desc": "Truy hồi lai (hybrid) được đồ thị tri thức tăng cường: dùng quan hệ SỬA ĐỔI để "
                                       "chèn nội dung sửa đổi ngay sau Điều bị sửa. Phù hợp hầu hết câu hỏi cụ thể."},
    "hybrid-rrf": {"label": "Kết hợp (Hybrid)", "badge": "", "icon": ":material/join_inner:",
                   "desc": "Trộn tìm theo ý nghĩa và theo từ khóa bằng Reciprocal Rank Fusion."},
    "dense": {"label": "Ngữ nghĩa (Dense)", "badge": "", "icon": ":material/psychology:",
              "desc": "Tìm đoạn luật gần nghĩa với câu hỏi (Vietnamese_Embedding + FAISS)."},
    "bm25": {"label": "Từ khóa (BM25)", "badge": "", "icon": ":material/key:",
             "desc": "Khớp từ khóa; mạnh khi câu hỏi có số hiệu văn bản, số Điều."},
    "graph-global": {"label": "Tổng hợp toàn cục", "badge": "Câu hỏi tổng quát", "icon": ":material/travel_explore:",
                     "desc": "GraphRAG Global: tổng hợp từ báo cáo các cộng đồng trong đồ thị. Dành cho câu hỏi "
                             "tổng quát (liệt kê, so sánh nhiều nhóm). Chậm hơn, khoảng 30–90 giây."},
}

PERSONAS = {
    "nguoi_dan": {
        "label": "Người dân", "icon": ":material/family_restroom:", "accent": "#0F766E", "soft": "#E6F4F1",
        "title": "Hỏi đáp pháp luật đất đai, nhà ở cho người dân",
        "tagline": "Hỏi bằng lời thường. Câu trả lời ngắn gọn, dễ hiểu, kèm căn cứ pháp lý.",
        "style": "ĐỐI TƯỢNG ĐỌC: người dân không có chuyên môn pháp luật. Viết câu ngắn, từ ngữ phổ thông; giải thích "
                 "thuật ngữ pháp lý khi dùng lần đầu; nếu là thủ tục thì trình bày thành các bước đánh số (nộp ở đâu, "
                 "cần giấy tờ gì) theo NGỮ CẢNH.",
        "examples": ["Điều kiện để được tách thửa đất ở tại Đà Nẵng là gì?",
                     "Hồ sơ đăng ký cấp Giấy chứng nhận quyền sử dụng đất lần đầu gồm những gì?",
                     "Ai được mua nhà ở xã hội và cần điều kiện gì?",
                     "Khi Nhà nước thu hồi đất, tôi được bồi thường những khoản nào?"],
        "details": False,
    },
    "doanh_nghiep": {
        "label": "Doanh nghiệp", "icon": ":material/business_center:", "accent": "#9A3412", "soft": "#FBEFE6",
        "title": "Tra cứu điều kiện, thủ tục, nghĩa vụ cho doanh nghiệp",
        "tagline": "Điều kiện kinh doanh, giao đất, thuê đất, dự án bất động sản, nghĩa vụ tài chính.",
        "style": "ĐỐI TƯỢNG ĐỌC: doanh nghiệp, nhà đầu tư. Trình bày gọn theo các mục: điều kiện; hồ sơ, thủ tục; "
                 "nghĩa vụ tài chính; thời hạn; chỉ đưa mục nào có trong NGỮ CẢNH; dùng gạch đầu dòng ngắn.",
        "examples": ["Tổ chức kinh doanh bất động sản phải đáp ứng điều kiện gì?",
                     "Chủ đầu tư được chuyển nhượng dự án bất động sản khi đáp ứng điều kiện nào?",
                     "Những trường hợp nào được miễn, giảm tiền thuê đất?",
                     "Khi nào giao đất, cho thuê đất phải đấu giá quyền sử dụng đất?"],
        "details": False,
    },
    "can_bo": {
        "label": "Cán bộ", "icon": ":material/account_balance:", "accent": "#1E3A8A", "soft": "#E8EEFB",
        "title": "Tra cứu nghiệp vụ cho cán bộ, công chức",
        "tagline": "Căn cứ Điều, Khoản chính xác; thẩm quyền; quy định đã sửa đổi; xem chi tiết truy hồi và đồ thị.",
        "style": "ĐỐI TƯỢNG ĐỌC: cán bộ, công chức chuyên môn. Dùng thuật ngữ pháp lý chính xác; nêu Điều, Khoản, Điểm "
                 "và số hiệu văn bản cho từng nhận định; nêu rõ thẩm quyền và quy định đã được sửa đổi, bổ sung nếu "
                 "NGỮ CẢNH có ghi chú sửa đổi.",
        "examples": ["Những nghị định nào đã sửa đổi, bổ sung Nghị định 103/2024/NĐ-CP?",
                     "Thẩm quyền thu hồi đất được phân định thế nào khi tổ chức chính quyền địa phương 2 cấp?",
                     "Trình tự, thủ tục thu hồi đất từ thông báo thu hồi đến quyết định thu hồi gồm những bước nào?",
                     "Hình thức xử phạt và biện pháp khắc phục hậu quả đối với hành vi lấn, chiếm đất?"],
        "details": True,
    },
}

ENTITY_STYLE = {"VAN_BAN": ("#1E3A8A", "Văn bản"), "DIEU_KHOAN": ("#0F766E", "Điều/Khoản"),
                "CO_QUAN": ("#9A3412", "Cơ quan"), "KHAI_NIEM": ("#6B7280", "Khái niệm")}
RELATION_STYLE = {"SUA_DOI": ("#DC2626", "sửa đổi"), "HUONG_DAN": ("#16A34A", "hướng dẫn"),
                  "THAM_CHIEU": ("#2563EB", "dẫn chiếu"), "CHUA": ("#94A3B8", "chứa"),
                  "AP_DUNG_CHO": ("#CBD5E1", "áp dụng cho")}


@st.cache_resource(show_spinner="Đang nạp mô hình nhúng và chỉ mục (lần đầu khoảng 30 giây)…")
def load_engine() -> tuple[Config, Retrievers]:
    config = Config.from_env()
    return config, Retrievers(config)


@st.cache_resource(show_spinner="Đang nạp báo cáo cộng đồng…")
def load_global(level: int = 1):
    from lc_app.graph.global_search import GlobalSearch
    return GlobalSearch(Config.from_env(), level=level)


def global_ready() -> bool:
    graph_dir = Config.from_env().graph_dir
    return (graph_dir / "reports.json").exists() and (graph_dir / "communities.json").exists()


@st.cache_resource(show_spinner="Đang nạp đồ thị tri thức…")
def load_graph() -> dict:
    graph_dir = Config.from_env().graph_dir
    graph = json.loads((graph_dir / "graph.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    by_chunk = defaultdict(list)
    for n in graph["nodes"]:
        for cid in n["chunk_ids"]:
            by_chunk[cid].append(n["id"])
    adjacency = defaultdict(list)
    for e in graph["edges"]:
        adjacency[e["source"]].append(e)
        adjacency[e["target"]].append(e)
    degree = Counter({k: len(v) for k, v in adjacency.items()})
    communities = []
    if (graph_dir / "communities.json").exists():
        communities = json.loads((graph_dir / "communities.json").read_text(encoding="utf-8"))
    return {"nodes": nodes, "edges": graph["edges"], "by_chunk": by_chunk, "adjacency": adjacency, "degree": degree,
            "communities": {c["id"]: c for c in communities}, "stats": graph.get("stats", {})}


def kb_stats(retrievers: Retrievers) -> dict:
    docs = retrievers.bm25.docs
    return {"documents": len({d.metadata.get("doc_key") for d in docs}), "chunks": len(docs),
            "danang": len({d.metadata.get("doc_key") for d in docs if str(d.metadata.get("doc_key", "")).startswith("DN.")})}


def source_rows(documents) -> list[dict]:
    return [{"n": i, "chunk_id": d.id, "citation": citation(d), "text": d.metadata["text"],
             "notes": d.metadata.get("notes"), "header": d.metadata.get("header"),
             "graph_rel": d.metadata.get("graph_rel"), "graph_via": d.metadata.get("graph_via"),
             "effective_date": d.metadata.get("effective_date")}
            for i, d in enumerate(documents, 1)]


def retrieve(mode: str, question: str) -> tuple[list[dict], list[dict]]:
    """Trả về (các chunk đưa vào ngữ cảnh, danh sách xếp hạng top-20 để xem chi tiết)."""
    config, retrievers = load_engine()
    ranked = retrievers.get(mode).invoke(question)
    used = cut_to_budget(ranked, config.depth, config.context_chars)
    used_ids = {d.id for d in used}
    ranking = [{"hạng": i, "chunk_id": d.id, "trích dẫn": citation(d), "vào ngữ cảnh": d.id in used_ids,
                "qua đồ thị": d.metadata.get("graph_rel") or ""} for i, d in enumerate(ranked[:20], 1)]
    return used, ranking


def stream_answer(question: str, documents, persona: str):
    """Sinh câu trả lời (stream) theo prompt của lc_app.chain, cộng hướng dẫn trình bày theo đối tượng."""
    config, _ = load_engine()
    if not documents:
        yield NO_CONTEXT
        return
    text = INSTRUCTIONS + "\n" + PERSONAS[persona]["style"]
    prompt = ChatPromptTemplate.from_messages([("human", text + "\n\nNGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}")])
    llm = ChatOpenAI(model=config.llm_model, temperature=0, streaming=True)
    for piece in (prompt | llm).stream({"context": format_context(documents), "question": question}):
        yield piece.content


def global_answer(question: str, persona: str) -> dict:
    search = load_global()
    result = search.invoke(question, style=PERSONAS[persona]["style"].replace("NGỮ CẢNH", "các ý"))
    result["report_rows"] = [{**{k: search.by_id[i][k] for k in ("community", "title", "summary", "rating", "level")},
                              "findings": search.by_id[i]["findings"], "chunk_ids": search.by_id[i]["chunk_ids"]}
                             for i in result["reports"]]
    result["n_reports"] = len(search.reports)
    return result


# ---------------------------------------------------------------- đồ thị con để vẽ

def _node(graph: dict, name: str, extra: dict | None = None) -> dict:
    n = graph["nodes"][name]
    color, kind = ENTITY_STYLE.get(n["type"], ("#6B7280", n["type"]))
    label = name if len(name) <= 38 else name[:36] + "…"
    return {"id": name, "label": label, "title": f"{kind}: {name}\n\n{n['description'][:500]}", "color": color,
            "group": n["type"], "value": 1 + min(graph["degree"][name], 60), **(extra or {})}


def _edges_among(graph: dict, names: set[str], skip: tuple[str, ...] = ()) -> list[dict]:
    out, seen = [], set()
    for name in names:
        for e in graph["adjacency"][name]:
            key = (e["source"], e["target"], e["type"])
            if e["type"] in skip or key in seen or e["source"] not in names or e["target"] not in names:
                continue
            seen.add(key)
            color, label = RELATION_STYLE.get(e["type"], ("#94A3B8", e["type"]))
            out.append({"from": e["source"], "to": e["target"], "color": color, "label": label,
                        "title": e["description"][:400], "arrows": "to", "width": 2.5 if e["type"] == "SUA_DOI" else 1.2})
    return out


def answer_subgraph(sources: list[dict], max_entities: int = 40) -> dict:
    """Thực thể được các chunk ngữ cảnh nhắc tới + nút nguồn [n]; thêm Điều sửa đổi 1 bước (SUA_DOI)."""
    graph = load_graph()
    score = Counter()
    for s in sources:
        for name in graph["by_chunk"].get(s["chunk_id"], []):
            score[name] += 1
    chosen = [n for n, _ in sorted(score.items(), key=lambda kv: (-kv[1], -graph["degree"][kv[0]]))[:max_entities]]
    names = set(chosen)
    for name in list(names):   # nội dung sửa đổi liên quan (đúng loại quan hệ chế độ đồ thị dùng)
        if graph["nodes"][name]["type"] == "DIEU_KHOAN":
            for e in graph["adjacency"][name]:
                if e["type"] == "SUA_DOI" and len(names) < max_entities + 12:
                    names.update((e["source"], e["target"]))
    nodes = [_node(graph, n) for n in names]
    edges = _edges_among(graph, names)
    for s in sources:
        sid = f"[{s['n']}]"
        nodes.append({"id": sid, "label": sid, "shape": "box", "color": "#111827", "font": {"color": "#FFFFFF"},
                      "title": s["citation"], "group": "SOURCE"})
        for name in graph["by_chunk"].get(s["chunk_id"], []):
            if name in names:
                edges.append({"from": sid, "to": name, "color": "#D1D5DB", "dashes": True, "width": 0.8})
    return {"nodes": nodes, "edges": edges}


def global_subgraph(report_rows: list[dict], per_community: int = 10, max_communities: int = 6) -> dict:
    graph = load_graph()
    palette = ["#1E3A8A", "#0F766E", "#9A3412", "#7C3AED", "#BE123C", "#0369A1"]
    names, color_of = set(), {}
    for i, row in enumerate(report_rows[:max_communities]):
        members = graph["communities"].get(row["community"], {}).get("entities", [])
        for name in sorted(members, key=lambda n: -graph["degree"][n])[:per_community]:
            names.add(name)
            color_of.setdefault(name, (palette[i % len(palette)], row["title"]))
    nodes = [_node(graph, n, {"color": color_of[n][0], "title": f"Cộng đồng: {color_of[n][1]}\n\n{n}"}) for n in names]
    return {"nodes": nodes, "edges": _edges_among(graph, names)}


def search_entities(query: str, limit: int = 30) -> list[str]:
    graph = load_graph()
    q = query.casefold().strip()
    if not q:
        return []
    hits = [n for n in graph["nodes"] if q in n.casefold()]
    return sorted(hits, key=lambda n: (-graph["degree"][n], len(n)))[:limit]


def ego_subgraph(name: str, relations: list[str], limit: int = 60) -> dict:
    graph = load_graph()
    edges = [e for e in graph["adjacency"][name] if e["type"] in relations]
    edges.sort(key=lambda e: -e["weight"])
    names = {name} | {x for e in edges[:limit] for x in (e["source"], e["target"])}
    nodes = [_node(graph, n, {"borderWidth": 4} if n == name else None) for n in names]
    return {"nodes": nodes, "edges": _edges_among(graph, names, skip=tuple(set(RELATION_STYLE) - set(relations)))}
