"""Vẽ đồ thị trong Neo4j thành trang HTML dạng sơ đồ mẫu (LEGAL_ONTOLOGY.md mục 7).

- Văn bản chính (--main) ở giữa, màu đậm; văn bản khác màu nhạt; Điều, khái niệm, cơ quan mỗi loại một màu; có chú
  thích "Loại nút"; nhãn cạnh tiếng Việt.
- Gộp lên cấp văn bản: Điều chỉ được dẫn chiếu (không có đoạn văn nào trong đồ thị chứa nó, tức không có cạnh CHUA đi
  vào) được vẽ gộp vào văn bản của nó, cạnh ghi số Điều được dẫn chiếu.
- Mỗi Điều hiện tối đa MAX_APPLIES cạnh "áp dụng cho" có độ mạnh cao nhất.

    .venv-lc/Scripts/python -m lc_app.graph.visualize [--main "Luật Đất đai 31/2024/QH15"] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from langchain_neo4j import Neo4jGraph

from ..config import Config
from .ontology import RELATION_LABELS

MAX_APPLIES = 5
COLORS = {"MAIN": "#1F4E79", "VAN_BAN": "#4A6F96", "DIEU_KHOAN": "#7A7A7A", "KHAI_NIEM": "#A5A5A5",
          "CO_QUAN": "#3C8D7E"}
LEGEND = (("MAIN", "Văn bản chính"), ("VAN_BAN", "Văn bản liên quan"), ("DIEU_KHOAN", "Điều khoản"),
          ("KHAI_NIEM", "Khái niệm"), ("CO_QUAN", "Cơ quan"))
DIEU = re.compile(r"^Điều\s+\d+[a-z]?\s+(?:Quy định kèm theo\s+)?")

PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Đồ thị pháp luật demo</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/standalone/umd/vis-network.min.js"></script>
<style>
:root {{ --bg:#ffffff; --fg:#1f2933; --muted:#52606d; --line:#d9dee3; }}
body {{ margin:0; font-family: Arial, sans-serif; background:var(--bg); color:var(--fg); }}
header {{ padding:16px 20px 4px; }} h1 {{ font-size:20px; margin:0 0 4px; }} p {{ margin:0; color:var(--muted); font-size:13px; }}
#graph {{ height: calc(100vh - 150px); min-height:480px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
.legend {{ display:flex; flex-wrap:wrap; gap:18px; padding:12px 20px; font-size:14px; align-items:center; }}
.legend b {{ margin-right:4px; }} .sw {{ display:inline-block; width:28px; height:16px; border-radius:3px; vertical-align:middle; margin-right:6px; }}
#info {{ padding:0 20px 16px; font-size:13px; color:var(--muted); white-space:pre-wrap; }}
</style></head><body>
<header><h1>{title}</h1><p>{subtitle}</p></header>
<div id="graph"></div>
<div class="legend"><b>Loại nút:</b>{legend}</div>
<div id="info">Bấm vào nút hoặc cạnh để xem mô tả.</div>
<script>
const nodes = new vis.DataSet({nodes});
const edges = new vis.DataSet({edges});
const network = new vis.Network(document.getElementById("graph"), {{nodes, edges}}, {{
  nodes: {{ shape: "box", margin: 10, font: {{ color: "#ffffff", face: "Arial", size: 14, bold: {{ color: "#ffffff" }} }},
           borderWidth: 1, widthConstraint: {{ maximum: 220 }} }},
  edges: {{ arrows: {{ to: {{ enabled: true, scaleFactor: 0.6 }} }}, color: {{ color: "#5f6b76" }},
           font: {{ face: "Arial", size: 12, color: "#2f3a45", strokeWidth: 3, strokeColor: "#ffffff", italic: true }},
           smooth: {{ type: "dynamic" }} }},
  physics: {{ barnesHut: {{ gravitationalConstant: -9000, springLength: 190 }}, stabilization: {{ iterations: 300 }} }}
}});
network.on("click", (p) => {{
  const item = p.nodes.length ? nodes.get(p.nodes[0]) : p.edges.length ? edges.get(p.edges[0]) : null;
  document.getElementById("info").textContent = item ? item.title : "Bấm vào nút hoặc cạnh để xem mô tả.";
}});
</script></body></html>"""


def document_of(name: str) -> str | None:
    return DIEU.sub("", name) if DIEU.match(name) else None


def build_view(neo4j: Neo4jGraph, main: str) -> tuple[list[dict], list[dict]]:
    entities = {r["id"]: r for r in neo4j.query(
        "MATCH (e:__Entity__) RETURN e.id AS id, [l IN labels(e) WHERE l <> '__Entity__'][0] AS type, "
        "e.description AS description")}
    relations = neo4j.query("MATCH (s:__Entity__)-[r]->(t:__Entity__) RETURN s.id AS source, t.id AS target, "
                            "type(r) AS type, r.description AS description, r.weight AS weight")
    contained = {r["target"] for r in relations if r["type"] == "CHUA"}
    # Điều chỉ được dẫn chiếu -> gộp vào văn bản của nó (nếu văn bản đó là một nút)
    collapse = {n: document_of(n) for n, e in entities.items()
                if e["type"] == "DIEU_KHOAN" and n not in contained and document_of(n) in entities}
    applies = defaultdict(list)
    merged: dict[tuple, dict] = {}
    for r in relations:
        if r["type"] == "AP_DUNG_CHO":
            applies[r["source"]].append(r)
            continue
        source, target = collapse.get(r["source"], r["source"]), collapse.get(r["target"], r["target"])
        if source == target:
            continue
        edge = merged.setdefault((source, target, r["type"]), {"count": 0, "details": []})
        edge["count"] += 1
        edge["details"].append(f"{r['source']} → {r['target']}: {r['description']}")
    for source, items in applies.items():
        for r in sorted(items, key=lambda x: -(x["weight"] or 0))[:MAX_APPLIES]:
            merged[(r["source"], r["target"], r["type"])] = {"count": 1, "details": [r["description"]]}
    shown = {s for s, _, _ in merged} | {t for _, t, _ in merged}
    nodes = [{"id": n, "label": n, "color": COLORS["MAIN" if n == main else entities[n]["type"]],
              "title": f"{n} [{entities[n]['type']}]\n{entities[n]['description']}",
              **({"font": {"size": 17}, "mass": 3} if n == main else {})}
             for n in sorted(shown)]
    edges = [{"from": s, "to": t, "title": "\n".join(e["details"]),
              "label": RELATION_LABELS[kind] + (f" ({e['count']} điều)" if e["count"] > 1 else "")}
             for (s, t, kind), e in merged.items()]
    return nodes, edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Vẽ đồ thị Neo4j thành HTML dạng sơ đồ mẫu.")
    parser.add_argument("--main", default="Luật Đất đai 31/2024/QH15", help="văn bản chính (đặt ở giữa)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    config = Config.from_env()
    neo4j = Neo4jGraph(**config.neo4j, refresh_schema=False)
    nodes, edges = build_view(neo4j, args.main)
    chunks = neo4j.query("MATCH (c:Chunk) RETURN count(c) AS n")[0]["n"]
    legend = "".join(f'<span><span class="sw" style="background:{COLORS[k]}"></span>{label}</span>' for k, label in LEGEND)
    html = PAGE.format(title="Đồ thị pháp luật: tách thửa, hợp thửa đất (demo)",
                       subtitle=f"{chunks} chunk · {len(nodes)} nút · {len(edges)} cạnh hiển thị · trích bằng "
                                f"{config.graph_llm_model} theo LEGAL_ONTOLOGY.md v2.0 · văn bản chính: {args.main}",
                       legend=legend, nodes=json.dumps(nodes, ensure_ascii=False),
                       edges=json.dumps(edges, ensure_ascii=False))
    out = args.out or config.graph_dir / "demo_graph.html"
    out.write_text(html, encoding="utf-8")
    print(f"{len(nodes)} nút, {len(edges)} cạnh -> {out}")


if __name__ == "__main__":
    main()
