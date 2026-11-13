"""Dựng bảng quan hệ giữa các văn bản trong KB từ metadata.

Nguồn: data/data/processed_kb/processed_raw_kb/*/*/metadata.json (amends, guides, based_on, replaces, consolidated)
       + data/chunk/chunk_v1/amendments.csv (sửa đổi ở mức Điều, nếu có).
Đích:  data/data/processed_kb/relations/
         relations.csv / relations.jsonl   mỗi dòng một quan hệ: nguồn -> đích
         relations.xlsx                    Văn bản | Quan hệ | Sửa đổi theo Điều | Tổng hợp
         relations_graph.html              sơ đồ quan hệ, mở bằng trình duyệt

Loại quan hệ (chiều nguồn -> đích):
  SUA_DOI      nguồn sửa đổi, bổ sung đích (cả hai trong KB)
  HUONG_DAN    nguồn quy định chi tiết / hướng dẫn thi hành đích
  CAN_CU       nguồn ban hành căn cứ đích (chỉ văn bản trong KB)
  THAY_THE     nguồn thay thế / bãi bỏ đích (đích đã hết hiệu lực, ngoài KB)
  HOP_NHAT     văn bản hợp nhất của nguồn đã gộp nội dung sửa đổi của đích

    python -m kb_pipeline.build_relations [--meta DIR] [--amendments CSV] [--out DIR]
(cần openpyxl để xuất Excel; thiếu thì bỏ qua file .xlsx)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPES = {
    "SUA_DOI": ("amends", "sửa đổi, bổ sung"),
    "HUONG_DAN": ("guides", "quy định chi tiết / hướng dẫn"),
    "CAN_CU": ("based_on", "căn cứ ban hành"),
    "THAY_THE": ("replaces", "thay thế / bãi bỏ"),
}
LABELS = {**{k: v[1] for k, v in TYPES.items()}, "HOP_NHAT": "hợp nhất nội dung sửa đổi của"}
GROUP_NAMES = {"01_luat": "Luật, Nghị quyết QH", "02_nghi_dinh": "Nghị định", "03_thong_tu": "Thông tư",
               "04_da_nang": "Văn bản Đà Nẵng"}


def load_documents(meta_dir: Path) -> dict[str, dict]:
    docs = {}
    for path in sorted(meta_dir.glob("*/*/metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["group"] = path.parent.parent.name
        docs[meta["document_number"]] = meta
    return docs


def build_edges(docs: dict[str, dict]) -> list[dict]:
    edges = []
    for number, meta in docs.items():
        for kind, (field, _) in TYPES.items():
            for target in meta.get(field) or []:
                edges.append({"source": number, "type": kind, "target": target})
        hn = meta.get("consolidated") or {}
        for target in hn.get("consolidates") or []:
            edges.append({"source": number, "type": "HOP_NHAT", "target": target,
                          "evidence": f"Văn bản hợp nhất {hn.get('number')}"})
    for edge in edges:
        target = docs.get(edge["target"])
        edge.update(source_title=docs[edge["source"]]["title"], label=LABELS[edge["type"]],
                    target_in_kb=target is not None, target_title=target["title"] if target else None)
        edge.setdefault("evidence", None)
    return edges


def load_article_amendments(path: Path, docs: dict[str, dict]) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["amender_title"] = docs.get(row["amender"], {}).get("title")
        row["target_title"] = docs.get(row["target"], {}).get("title")
    return rows


def short(meta: dict | None, number: str) -> str:
    return f"{meta['document_type']} {number}" if meta else number


# ---------------------------------------------------------------- Excel
def write_excel(path: Path, docs: dict[str, dict], edges: list[dict], articles: list[dict]) -> bool:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        return False
    font, bold = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True, color="FFFFFF")
    head = PatternFill("solid", fgColor="1F4E78")
    wb = Workbook()

    def sheet(title: str, header: list[str], rows: list[list], widths: dict[int, int]):
        ws = wb.active if wb.active.title == "Sheet" else wb.create_sheet()
        ws.title = title
        ws.append(header)
        for row in rows:
            ws.append(row)
        for row in ws.iter_rows():
            for cell in row:
                cell.font = font
        for cell in ws[1]:
            cell.font, cell.fill = bold, head
        for i in range(1, len(header) + 1):
            ws.column_dimensions[get_column_letter(i)].width = widths.get(i, 16)
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(header))}{len(rows) + 1}"
        return ws

    n = len(edges) + 1
    rel = lambda col: f"'Quan hệ'!${col}$2:${col}${n}"  # noqa: E731
    doc_rows = []
    for i, (number, meta) in enumerate(docs.items(), start=2):
        doc_rows.append([number, meta["document_type"], meta["title"], GROUP_NAMES.get(meta["group"]),
                         meta["issuing_authority"], meta["effective_date"], meta.get("expiry_date"),
                         f"=COUNTIF({rel('A')},A{i})", f"=COUNTIF({rel('E')},A{i})",
                         f"=COUNTIFS({rel('E')},A{i},{rel('C')},\"SUA_DOI\")",
                         f"=COUNTIFS({rel('E')},A{i},{rel('C')},\"HUONG_DAN\")"])
    sheet("Văn bản", ["Số hiệu", "Loại", "Tên", "Nhóm", "Cơ quan ban hành", "Hiệu lực từ", "Hết hiệu lực",
                      "Số quan hệ đi", "Số quan hệ đến", "Bị sửa đổi bởi (số VB)", "Được hướng dẫn bởi (số VB)"],
          doc_rows, {1: 20, 3: 70, 4: 20, 5: 26})
    sheet("Quan hệ", ["Văn bản nguồn", "Tên văn bản nguồn", "Loại quan hệ", "Diễn giải", "Văn bản đích",
                      "Tên văn bản đích", "Đích có trong KB", "Ghi chú"],
          [[e["source"], e["source_title"], e["type"], e["label"], e["target"], e["target_title"],
            "có" if e["target_in_kb"] else "không (ngoài KB)", e["evidence"]] for e in edges],
          {1: 20, 2: 60, 3: 12, 4: 28, 5: 20, 6: 60, 7: 16, 8: 30})
    if articles:
        sheet("Sửa đổi theo Điều", ["Văn bản sửa đổi", "Điều (văn bản sửa đổi)", "Văn bản bị sửa", "Điều bị sửa",
                                    "Trạng thái", "Tên văn bản sửa đổi", "Tên văn bản bị sửa"],
              [[a["amender"], a["amender_article"], a["target"], a["target_article"],
                "đã có trong VBHN" if a["status"] == "da_hop_nhat" else "ghi chú vào Điều bị sửa",
                a["amender_title"], a["target_title"]] for a in articles],
              {1: 20, 3: 20, 5: 22, 6: 60, 7: 60})
    summary = [[kind, LABELS[kind], f"=COUNTIF({rel('C')},A{i})",
                f"=COUNTIFS({rel('C')},A{i},{rel('G')},\"có\")"] for i, kind in enumerate(LABELS, start=2)]
    total = len(summary) + 2
    summary.append(["Tổng", "", f"=SUM(C2:C{total - 1})", f"=SUM(D2:D{total - 1})"])
    ws = sheet("Tổng hợp", ["Loại quan hệ", "Diễn giải", "Số quan hệ", "Trong đó đích có trong KB"], summary,
               {1: 14, 2: 32, 3: 14, 4: 26})
    for cell in ws[total]:
        cell.font = Font(name="Arial", size=10, bold=True)
    ws.append([])
    ws.append(["Ghi chú: quan hệ lấy từ metadata (amends, guides, based_on, replaces, consolidated) do "
               "kb_pipeline.build_metadata trích từ metadata gốc và nội dung văn bản; chưa đối chiếu thủ công."])
    wb.move_sheet("Tổng hợp", offset=-3)
    wb.calculation.fullCalcOnLoad = True
    wb.save(path)
    return True


# ---------------------------------------------------------------- HTML
GRAPH_HTML = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quan hệ văn bản KB</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/standalone/umd/vis-network.min.js"></script>
<style>
 :root { --bg:#fff; --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; }
 @media (prefers-color-scheme: dark) { :root { --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --line:#3d444d; } }
 body { margin:0; font-family: Arial, sans-serif; background:var(--bg); color:var(--fg); }
 header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex; flex-wrap:wrap; gap:12px; align-items:center; }
 header h1 { font-size:16px; margin:0 12px 0 0; }
 label { font-size:13px; color:var(--muted); cursor:pointer; white-space:nowrap; }
 .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:middle; }
 #net { height: calc(100vh - 110px); }
 #info { padding:6px 16px; font-size:13px; color:var(--muted); border-top:1px solid var(--line); min-height:20px; }
</style></head><body>
<header><h1>Quan hệ giữa các văn bản trong KB</h1><span id="filters"></span>
<label><input type="checkbox" id="ext" checked> văn bản ngoài KB</label></header>
<div id="net"></div><div id="info">Bấm vào một văn bản để xem quan hệ.</div>
<script>
const DATA = __DATA__;
const EDGE = {SUA_DOI:"#d1242f", HUONG_DAN:"#1a7f37", CAN_CU:"#8c959f", THAY_THE:"#9a6700", HOP_NHAT:"#8250df"};
const NODE = {"01_luat":"#0969da","02_nghi_dinh":"#1a7f37","03_thong_tu":"#bf8700","04_da_nang":"#cf222e","ngoai_kb":"#afb8c1"};
const show = Object.fromEntries(Object.keys(EDGE).map(k => [k, k !== "CAN_CU"]));
const f = document.getElementById("filters");
for (const [k, color] of Object.entries(EDGE)) {
  const l = document.createElement("label");
  l.innerHTML = `<input type="checkbox" ${show[k] ? "checked" : ""}> <span class="sw" style="background:${color}"></span>${DATA.labels[k]}`;
  l.querySelector("input").onchange = e => { show[k] = e.target.checked; draw(); };
  f.appendChild(l);
}
document.getElementById("ext").onchange = draw;
const net = new vis.Network(document.getElementById("net"), {}, {
  physics:{solver:"forceAtlas2Based", stabilization:{iterations:200}},
  edges:{arrows:{to:{enabled:true, scaleFactor:.5}}, smooth:false, width:1.2},
  nodes:{shape:"dot", font:{size:12, color:getComputedStyle(document.body).color}}, interaction:{hover:true}});
function draw() {
  const ext = document.getElementById("ext").checked;
  const edges = DATA.edges.filter(e => show[e.type] && (ext || e.target_in_kb));
  const used = new Set(edges.flatMap(e => [e.source, e.target]));
  const nodes = DATA.nodes.filter(n => n.in_kb || used.has(n.id)).filter(n => ext || n.in_kb).map(n => ({
    id:n.id, label:n.short, title:n.title, color:NODE[n.group], size:n.in_kb ? 10 + Math.min(20, n.degree) : 6}));
  net.setData({nodes, edges: edges.map((e, i) => ({id:i, from:e.source, to:e.target, color:EDGE[e.type], title:DATA.labels[e.type]}))});
}
net.on("click", p => {
  if (!p.nodes.length) return;
  const id = p.nodes[0], n = DATA.nodes.find(x => x.id === id);
  const out = DATA.edges.filter(e => e.source === id).map(e => `${DATA.labels[e.type]} ${e.target}`);
  const inc = DATA.edges.filter(e => e.target === id).map(e => `${e.source} ${DATA.labels[e.type]}`);
  document.getElementById("info").textContent = `${n.title} — đi: ${out.join("; ") || "không"} | đến: ${inc.join("; ") || "không"}`;
});
draw();
</script></body></html>
"""


def write_graph(path: Path, docs: dict[str, dict], edges: list[dict]) -> None:
    degree = Counter(x for e in edges for x in (e["source"], e["target"]))
    nodes = [{"id": n, "short": n, "title": f"{m['document_type']} {n} – {m['title']}", "group": m["group"],
              "in_kb": True, "degree": degree[n]} for n, m in docs.items()]
    nodes += [{"id": t, "short": t, "title": f"{t} (ngoài KB)", "group": "ngoai_kb", "in_kb": False, "degree": 0}
              for t in sorted({e["target"] for e in edges if not e["target_in_kb"]})]
    data = {"nodes": nodes, "edges": [{k: e[k] for k in ("source", "target", "type", "target_in_kb")} for e in edges],
            "labels": LABELS}
    path.write_text(GRAPH_HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dựng bảng quan hệ giữa các văn bản trong KB.")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/data/processed_kb/processed_raw_kb")
    parser.add_argument("--amendments", type=Path, default=ROOT / "data/chunk/chunk_v1/amendments.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data/data/processed_kb/relations")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    docs = load_documents(args.meta)
    edges = build_edges(docs)
    articles = load_article_amendments(args.amendments, docs)

    fields = ["source", "source_title", "type", "label", "target", "target_title", "target_in_kb", "evidence"]
    with (args.out / "relations.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: e[k] for k in fields} for e in edges)
    with (args.out / "relations.jsonl").open("w", encoding="utf-8") as handle:
        for edge in edges:
            handle.write(json.dumps({k: edge[k] for k in fields}, ensure_ascii=False) + "\n")
    excel = write_excel(args.out / "relations.xlsx", docs, edges, articles)
    write_graph(args.out / "relations_graph.html", docs, edges)

    counts = Counter(e["type"] for e in edges)
    print(f"{len(docs)} văn bản, {len(edges)} quan hệ văn bản ({dict(counts)}), "
          f"{sum(not e['target_in_kb'] for e in edges)} trỏ ra ngoài KB; {len(articles)} quan hệ sửa đổi mức Điều")
    print(f"-> {args.out}" + ("" if excel else " (không có openpyxl: bỏ qua relations.xlsx)"))


if __name__ == "__main__":
    main()
