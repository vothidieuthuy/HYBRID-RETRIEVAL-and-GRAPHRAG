"""Vẽ đồ thị bằng vis-network (nạp từ CDN, không cần cài gói Python) trong một component HTML của Streamlit."""
from __future__ import annotations

import json

import streamlit.components.v1 as components

from backend import ENTITY_STYLE, RELATION_STYLE

TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  body{margin:0;font-family:'Be Vietnam Pro',system-ui,sans-serif;background:#fff}
  #wrap{position:relative;border:1px solid #E2E8F0;border-radius:14px;overflow:hidden}
  #net{height:__H__px;background:radial-gradient(circle at 20% 20%,#F8FAFC,#FFFFFF)}
  .legend{display:flex;flex-wrap:wrap;gap:6px 14px;padding:10px 14px;border-top:1px solid #E2E8F0;font-size:12.5px;color:#334155}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .dot{width:11px;height:11px;border-radius:50%;display:inline-block}
  .bar{width:18px;height:3px;display:inline-block;border-radius:2px}
  button{position:absolute;top:10px;right:10px;z-index:5;border:1px solid #CBD5E1;background:#fff;border-radius:8px;
         padding:6px 10px;font-size:12.5px;cursor:pointer;color:#1E293B}
  button:focus-visible{outline:3px solid #2563EB;outline-offset:2px}
  .empty{display:flex;align-items:center;justify-content:center;height:__H__px;color:#64748B;font-size:14px}
</style></head><body><div id="wrap">
<button onclick="net && net.fit({animation:{duration:400}})" aria-label="Thu vừa khung">Vừa khung</button>
<div id="net"></div><div class="legend">__LEGEND__</div></div>
<script>
const nodes = __NODES__, edges = __EDGES__;
let net = null;
if (!nodes.length) { document.getElementById('net').outerHTML = '<div class="empty">Không có thực thể nào để hiển thị.</div>'; }
else {
  net = new vis.Network(document.getElementById('net'),
    {nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges.map((e, i) => ({id: i, ...e})))},
    {nodes: {shape: 'dot', scaling: {min: 8, max: 26}, font: {size: 13, face: 'Be Vietnam Pro, system-ui', strokeWidth: 3,
             strokeColor: '#ffffff'}, borderWidth: 1.5},
     edges: {smooth: {type: 'continuous'}, font: {size: 10, align: 'middle', color: '#475569', strokeWidth: 3},
             arrows: {to: {scaleFactor: 0.5}}},
     physics: {solver: 'forceAtlas2Based', forceAtlas2Based: {gravitationalConstant: -60, springLength: 110},
               stabilization: {iterations: 220}},
     interaction: {hover: true, tooltipDelay: 120, navigationButtons: false, keyboard: true}});
  net.once('stabilizationIterationsDone', () => net.setOptions({physics: false}));
}
</script></body></html>"""


def legend(include_sources: bool) -> str:
    items = [f'<span><i class="dot" style="background:{c}"></i>{label}</span>' for c, label in ENTITY_STYLE.values()]
    items += [f'<span><i class="bar" style="background:{c}"></i>{label}</span>'
              for c, label in RELATION_STYLE.values()]
    if include_sources:
        items.append('<span><i class="dot" style="background:#111827;border-radius:3px"></i>đoạn nguồn [n]</span>')
    return "".join(items)


def render(data: dict, height: int = 520, include_sources: bool = False) -> None:
    html = (TEMPLATE.replace("__H__", str(height)).replace("__LEGEND__", legend(include_sources))
            .replace("__NODES__", json.dumps(data["nodes"], ensure_ascii=False))
            .replace("__EDGES__", json.dumps(data["edges"], ensure_ascii=False)))
    components.html(html, height=height + 70, scrolling=False)
