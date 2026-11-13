"""Bước 4–5 của Microsoft GraphRAG: chia đồ thị thành cộng đồng phân cấp rồi để LLM viết báo cáo cho từng cộng đồng.

Cộng đồng: Microsoft dùng hierarchical Leiden (graspologic). Ở đây dùng Louvain của networkx (không cài thêm gói), chia
đệ quy cộng đồng lớn hơn `max_size` như max_cluster_size của Microsoft, rồi tách mỗi cộng đồng thành các thành phần
liên thông: đây là điểm Leiden sửa so với Louvain (Louvain có thể sinh cộng đồng không liên thông). Đồ thị vô hướng,
trọng số = độ mạnh cạnh đã cộng dồn; bỏ cạnh CHUA (văn bản -> Điều của chính nó): giữ thì cộng đồng gần như trùng với
từng văn bản, mất các nhóm chủ đề xuyên văn bản. Thực thể không có cạnh nào khác không thuộc cộng đồng nào.
Tái lập được: nút đánh số nguyên theo thứ tự tên, cạnh thêm theo thứ tự cố định, seed cố định (cùng kết quả với mọi
PYTHONHASHSEED).
Quy ước như Microsoft: level 0 là cấp thô nhất, level càng lớn càng mịn.

Báo cáo: từ cấp mịn lên cấp thô; ngữ cảnh = thực thể (tên, loại, mô tả) và quan hệ (nguồn, đích, loại, mô tả) của cộng
đồng, xếp theo bậc; vượt ngân sách thì báo cáo của cộng đồng con thay phần chi tiết (build_mixed_context của Microsoft).
Báo cáo giữ chunk_ids nguồn (hợp các chunk của thực thể) để truy về đoạn luật. LLM cache SQLite dùng chung với bước dựng
đồ thị (config.graph_cache): chạy lại không tốn tiền.

    .venv-lc/Scripts/python -m lc_app.graph.community --plan   # chỉ chia cộng đồng, đếm, ước chi phí
    .venv-lc/Scripts/python -m lc_app.graph.community          # chia cộng đồng + viết báo cáo
Đầu ra: <graph_dir>/communities.json, <graph_dir>/reports.json.
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI

from ..config import Config

DESC_CHARS = 400
REPORT_PROMPT = """Bạn là chuyên gia pháp luật đất đai, nhà ở Việt Nam, viết báo cáo về một cộng đồng trong đồ thị tri thức pháp luật để phục vụ người tra cứu.

Dữ liệu gồm các thực thể, quan hệ (và có thể báo cáo của cộng đồng con) của cộng đồng. Hãy viết báo cáo về: các văn bản, Điều và khái niệm chính; quy định, điều kiện, thủ tục, nghĩa vụ, thẩm quyền liên quan; mối liên hệ giữa các văn bản (sửa đổi, hướng dẫn, dẫn chiếu); những điểm người dân hoặc cán bộ cần lưu ý.

Trả về duy nhất một JSON:
{{"title": "tên ngắn, cụ thể, nêu chủ đề và văn bản tiêu biểu",
  "summary": "tóm tắt về cấu trúc và nội dung cộng đồng",
  "rating": <số thực 0-10: mức quan trọng của cộng đồng đối với việc tra cứu pháp luật đất đai, nhà ở>,
  "rating_explanation": "một câu giải thích điểm",
  "findings": [{{"summary": "ý chính ngắn", "explanation": "giải thích nhiều câu, nêu rõ Điều, văn bản căn cứ"}}]}}

QUY TẮC
- findings gồm 3–8 ý, bám sát dữ liệu; không đưa thông tin không có trong dữ liệu.
- Ghi tên Điều, văn bản đúng như trong dữ liệu; giữ nguyên nghĩa các từ tình thái (phải, được, có thể, không được).
- Tổng độ dài không quá 800 từ.

DỮ LIỆU
{context}"""


def cut(text: str, limit: int = DESC_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def detect(graph: dict, max_size: int, seed: int) -> list[dict]:
    # nút là số nguyên (thứ tự tên), cạnh thêm theo thứ tự cố định: Louvain của networkx duyệt tập nút, mà hash của
    # chuỗi đổi theo mỗi lần chạy Python (PYTHONHASHSEED) -> dùng tên làm nút thì cùng seed vẫn ra cộng đồng khác
    names = sorted(n["id"] for n in graph["nodes"])
    index = {name: i for i, name in enumerate(names)}
    weights: dict[tuple[int, int], float] = defaultdict(float)
    for e in graph["edges"]:
        if e["type"] == "CHUA" or e["source"] == e["target"]:
            continue
        a, b = sorted((index[e["source"]], index[e["target"]]))
        weights[(a, b)] += e["weight"]
    g = nx.Graph()
    g.add_nodes_from(sorted({x for pair in weights for x in pair}))
    g.add_weighted_edges_from((a, b, w) for (a, b), w in sorted(weights.items()))
    communities: list[dict] = []

    def add(nodes: set[str], level: int, parent: int | None) -> None:
        sub = g.subgraph(sorted(nodes))
        groups = [set(c) for c in nx.community.louvain_communities(sub, weight="weight", seed=seed)]
        if len(groups) == 1 and parent is not None:   # không chia thêm được
            return
        parts = [set(c) for group in groups for c in nx.connected_components(sub.subgraph(group))]
        for part in sorted(parts, key=lambda p: (-len(p), min(p))):
            community = {"id": len(communities), "level": level, "parent": parent, "children": [],
                         "entities": sorted(names[i] for i in part)}
            communities.append(community)
            if parent is not None:
                communities[parent]["children"].append(community["id"])
            if len(part) > max_size:
                add(part, level + 1, community["id"])

    for component in sorted(nx.connected_components(g), key=lambda c: (-len(c), min(c))):
        if len(component) > max_size:
            add(set(component), 0, None)
        else:
            communities.append({"id": len(communities), "level": 0, "parent": None, "children": [],
                                "entities": sorted(names[i] for i in component)})
    return communities


def context(community: dict, nodes: dict, edges: list[dict], degree: Counter, reports: dict, budget: int) -> str:
    members = set(community["entities"])
    entity_lines = [f"{n}|{nodes[n]['type']}|{cut(nodes[n]['description'])}"
                    for n in sorted(members, key=lambda n: -degree[n])]
    inner = sorted((e for e in edges if e["source"] in members and e["target"] in members),
                   key=lambda e: -(degree[e["source"]] + degree[e["target"]]))
    rel_lines = [f"{e['source']}|{e['target']}|{e['type']}|{cut(e['description'])}" for e in inner]
    head_e, head_r = "Thực thể\ntên|loại|mô tả\n", "\n\nQuan hệ\nnguồn|đích|loại|mô tả\n"
    full = head_e + "\n".join(entity_lines) + head_r + "\n".join(rel_lines)
    if len(full) <= budget:
        return full
    children = sorted((reports[c] for c in community["children"] if c in reports), key=lambda r: -r["size"])
    parts, used = [], 0
    for report in children:   # báo cáo con (cộng đồng lớn trước) chiếm tối đa nửa ngân sách
        line = f"{report['title']}: {cut(report['summary'], 700)}"
        if used + len(line) > budget // 2:
            break
        parts.append(line)
        used += len(line) + 1
    head = ("Báo cáo cộng đồng con\n" + "\n".join(parts) + "\n\n") if parts else ""
    remaining = (budget - len(head) - 100) // 2

    def take(lines: list[str]) -> list[str]:
        out, size = [], 0
        for line in lines:
            if size + len(line) > remaining:
                break
            out.append(line)
            size += len(line) + 1
        return out

    return head + head_e + "\n".join(take(entity_lines)) + head_r + "\n".join(take(rel_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", action="store_true", help="chỉ chia cộng đồng và ước chi phí")
    parser.add_argument("--max-size", type=int, default=30, help="cộng đồng lớn hơn thì chia tiếp (max_cluster_size)")
    parser.add_argument("--min-size", type=int, default=2, help="cộng đồng nhỏ hơn không viết báo cáo")
    parser.add_argument("--budget", type=int, default=16000, help="ngân sách ngữ cảnh mỗi báo cáo (ký tự)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = Config.from_env()
    graph = json.loads((config.graph_dir / "graph.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = [e for e in graph["edges"] if e["type"] != "CHUA"]
    degree = Counter(x for e in edges for x in (e["source"], e["target"]))
    communities = detect(graph, args.max_size, args.seed)
    for c in communities:
        c["chunk_ids"] = sorted({cid for n in c["entities"] for cid in nodes[n]["chunk_ids"]})
    (config.graph_dir / "communities.json").write_text(json.dumps(communities, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
    eligible = [c for c in communities if len(c["entities"]) >= args.min_size]
    by_level = Counter(c["level"] for c in eligible)
    covered = len({n for c in communities for n in c["entities"]})
    chars = sum(min(len(context(c, nodes, edges, degree, {}, args.budget)), args.budget) for c in eligible)
    print(f"{len(communities)} cộng đồng ({len(eligible)} cộng đồng >= {args.min_size} thực thể, theo cấp "
          f"{dict(sorted(by_level.items()))}); {covered}/{len(nodes)} thực thể thuộc cộng đồng; "
          f"kích thước cấp 0: {sorted((len(c['entities']) for c in communities if c['level'] == 0), reverse=True)[:10]}; "
          f"ước ~${(chars / 3 + 1500 * len(eligible)) / 1e6 * 0.15 + 900 * len(eligible) / 1e6 * 0.6:.2f}", flush=True)
    if args.plan:
        return

    set_llm_cache(SQLiteCache(database_path=str(config.graph_cache)))
    local = threading.local()

    def llm() -> ChatOpenAI:
        if not hasattr(local, "llm"):
            local.llm = ChatOpenAI(model=config.graph_llm_model, temperature=0, max_retries=8,
                                   model_kwargs={"response_format": {"type": "json_object"}})
        return local.llm

    reports: dict[int, dict] = {}

    def write(community: dict) -> dict | None:
        reply = llm().invoke(REPORT_PROMPT.format(
            context=context(community, nodes, edges, degree, reports, args.budget)))
        try:
            data = json.loads(reply.content)
            rating = float(data.get("rating", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not data.get("title") or not data.get("summary"):
            return None
        return {"community": community["id"], "level": community["level"], "size": len(community["entities"]),
                "title": str(data["title"]), "summary": str(data["summary"]), "rating": rating,
                "rating_explanation": str(data.get("rating_explanation", "")),
                "findings": [{"summary": str(f.get("summary", "")), "explanation": str(f.get("explanation", ""))}
                             for f in data.get("findings") or [] if isinstance(f, dict)],
                "chunk_ids": community["chunk_ids"], "tokens": reply.usage_metadata}

    groups = defaultdict(list)
    for c in eligible:
        groups[c["level"]].append(c)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for level in sorted(groups, reverse=True):   # cấp mịn trước: cấp cha dùng báo cáo của cấp con
            for report in pool.map(write, groups[level]):
                if report:
                    reports[report["community"]] = report
            print(f"cấp {level}: {len(groups[level])} cộng đồng, tổng {len(reports)} báo cáo", flush=True)
    ordered = sorted(reports.values(), key=lambda r: (r["level"], r["community"]))
    (config.graph_dir / "reports.json").write_text(json.dumps(ordered, ensure_ascii=False, indent=1), encoding="utf-8")
    tokens = [r["tokens"] for r in ordered if r.get("tokens")]   # lần gọi lấy từ cache không có token
    print(f"{len(ordered)} báo cáo ({len(eligible) - len(ordered)} lỗi/bỏ); chi phí lần này "
          f"${sum(t['input_tokens'] for t in tokens) / 1e6 * 0.15 + sum(t['output_tokens'] for t in tokens) / 1e6 * 0.6:.2f}"
          f" -> {config.graph_dir / 'reports.json'}")


if __name__ == "__main__":
    main()
