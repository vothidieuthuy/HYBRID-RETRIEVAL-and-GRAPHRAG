"""Giao diện hỏi đáp pháp luật đất đai, nhà ở Đà Nẵng: 3 đối tượng (người dân, doanh nghiệp, cán bộ), 5 chế độ tìm
kiếm (dense, BM25, hybrid, hybrid + đồ thị, GraphRAG global) và khám phá đồ thị tri thức.

    cd streamlit
    ..\\.venv-lc\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

import time

import streamlit as st

st.set_page_config(page_title="Trợ lý pháp luật Đất đai – Nhà ở Đà Nẵng", page_icon=":material/gavel:",
                   layout="wide", initial_sidebar_state="expanded")

import backend as B  # noqa: E402
import graph_view as G  # noqa: E402
from style import LOGO, css  # noqa: E402

DEFAULTS = {"persona": "nguoi_dan", "mode": "graph-augmented-hybrid", "messages": [], "big_font": False, "pending": None,
            "view": "Hỏi đáp"}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    persona = st.session_state.persona
    st.markdown(f'<div class="brand"><div class="logo">{LOGO}</div><div><b>Trợ lý pháp luật<br>Đất đai – Nhà ở</b>'
                f'<small>Thành phố Đà Nẵng</small></div></div>', unsafe_allow_html=True)
    st.radio("Bạn là", list(B.PERSONAS), key="persona",
             format_func=lambda k: f"{B.PERSONAS[k]['icon']} **{B.PERSONAS[k]['label']}**",
             captions=[p["tagline"] for p in B.PERSONAS.values()])
    persona = st.session_state.persona
    with st.expander(":material/tune: Chế độ tìm kiếm", expanded=B.PERSONAS[persona]["details"]):
        st.radio("Chế độ", list(B.MODES), key="mode", label_visibility="collapsed",
                 format_func=lambda m: f"{B.MODES[m]['icon']} {B.MODES[m]['label']}"
                                       + (f" · *{B.MODES[m]['badge']}*" if B.MODES[m]["badge"] else ""),
                 captions=[m["desc"] for m in B.MODES.values()])
    st.toggle("Cỡ chữ lớn", key="big_font", help="Phóng to chữ toàn trang cho dễ đọc.")
    if st.button(":material/delete_sweep: Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    try:
        _, retrievers = B.load_engine()
        stats = B.kb_stats(retrievers)
        graph_stats = B.load_graph()["stats"]
        st.markdown(
            '<div class="section-title">Kho dữ liệu</div>'
            f'<div class="stat"><span>Văn bản còn hiệu lực</span><b>{stats["documents"]}</b></div>'
            f'<div class="stat"><span>trong đó của Đà Nẵng</span><b>{stats["danang"]}</b></div>'
            f'<div class="stat"><span>Đoạn luật tra cứu được</span><b>{stats["chunks"]:,}</b></div>'
            f'<div class="stat"><span>Thực thể / quan hệ đồ thị</span><b>{graph_stats.get("nodes", 0):,} / '
            f'{graph_stats.get("edges", 0):,}</b></div>'.replace(",", "."), unsafe_allow_html=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Chưa nạp được dữ liệu: {exc}")
    st.markdown('<div class="disclaimer">Thông tin chỉ để tham khảo, được tổng hợp tự động từ văn bản pháp luật. '
                'Không thay thế văn bản trả lời chính thức của cơ quan nhà nước có thẩm quyền.</div>',
                unsafe_allow_html=True)

profile = B.PERSONAS[persona]
mode = st.session_state.mode
st.markdown(css(profile["accent"], profile["soft"], st.session_state.big_font), unsafe_allow_html=True)

# ------------------------------------------------------------------ hero
mode_info = B.MODES[mode]
st.markdown(
    f'<div class="hero"><span class="who">{profile["label"]}</span><h1>{profile["title"]}</h1>'
    f'<p>{profile["tagline"]}</p><div class="chips">'
    f'<span class="chip on">Chế độ: {mode_info["label"]}</span>'
    '<span class="chip">Luật Đất đai 2024 · Luật Nhà ở 2023 · Luật KDBĐS 2023</span>'
    '<span class="chip">Nghị định, Thông tư hướng dẫn</span><span class="chip">Quyết định của UBND TP Đà Nẵng</span>'
    '</div></div>', unsafe_allow_html=True)

view = st.radio("Trang", ["Hỏi đáp", "Khám phá đồ thị", "Hướng dẫn"], key="view", horizontal=True,
                label_visibility="collapsed")


# ------------------------------------------------------------------ hiển thị một câu trả lời
GRAPH_REL = {"SUA_DOI": "SỬA ĐỔI", "DAN_CHIEU": "DẪN CHIẾU"}   # graph_rel do graph-augmented-hybrid gán


def render_sources(sources: list[dict], compact: bool) -> None:
    for s in sources:
        title = f"[{s['n']}] {s['citation']}"
        with st.expander(title, expanded=False):
            if s.get("graph_rel") in GRAPH_REL:
                st.markdown(f'<div class="src-graph">Đoạn này được đồ thị bổ sung qua quan hệ '
                            f'<b>{GRAPH_REL[s["graph_rel"]]}</b>'
                            f'{" (" + s["graph_via"] + ")" if s.get("graph_via") else ""}.</div>',
                            unsafe_allow_html=True)
            if s.get("notes"):
                st.markdown(f'<div class="src-note">{s["notes"]}</div>', unsafe_allow_html=True)
            text = s["text"] if not compact or len(s["text"]) < 1200 else s["text"][:1200] + "…"
            st.markdown(text.replace("\n", "  \n"))
            if not compact:
                st.caption(f"Mã đoạn: {s['chunk_id']}")


def render_answer(message: dict, latest: bool) -> None:
    info = B.MODES[message["mode"]]
    who = B.PERSONAS[message["persona"]]
    badges = [f'<span class="badge accent">{info["label"]}</span>', f'<span class="badge">{who["label"]}</span>',
              f'<span class="badge">{message["seconds"]:.1f} giây</span>']
    added = sum(s.get("graph_rel") in GRAPH_REL for s in message.get("sources", []))
    if added:
        badges.append(f'<span class="badge red">Đồ thị bổ sung {added} đoạn</span>')
    st.markdown(f'<div class="meta">{"".join(badges)}</div>', unsafe_allow_html=True)
    st.markdown(message["answer"])
    details = who["details"]
    if message["mode"] == "graph-global":
        tabs = st.tabs([f":material/hub: Báo cáo cộng đồng ({len(message['reports'])})",
                        ":material/bubble_chart: Đồ thị cộng đồng"] + ([":material/list: Các ý đã tổng hợp"] if details else []))
        with tabs[0]:
            st.caption(f"Tổng hợp từ {message['n_reports']} báo cáo cộng đồng; dưới đây là các báo cáo được dẫn.")
            for r in message["reports"]:
                st.markdown(f'<div class="report"><span class="r">cộng đồng {r["community"]} · mức quan trọng '
                            f'{r["rating"]:.0f}/10</span><b>{r["title"]}</b><br>{r["summary"]}</div>',
                            unsafe_allow_html=True)
        with tabs[1]:
            if st.toggle("Hiển thị đồ thị", value=latest, key=f"g{message['id']}"):
                G.render(B.global_subgraph(message["reports"]), height=480)
        if details:
            with tabs[2]:
                for p in message["points"]:
                    st.markdown(f"- **{p['score']}** · {p['description']}")
        return
    labels = [f":material/menu_book: Căn cứ pháp lý ({len(message['sources'])})", ":material/hub: Đồ thị liên quan"]
    if details:
        labels.append(":material/analytics: Chi tiết truy hồi")
    tabs = st.tabs(labels)
    with tabs[0]:
        render_sources(message["sources"], compact=not details)
    with tabs[1]:
        st.caption("Các văn bản, Điều, cơ quan, khái niệm mà các đoạn căn cứ nhắc tới; ô đen [n] là đoạn căn cứ. "
                   "Cạnh đỏ là quan hệ sửa đổi.")
        if st.toggle("Hiển thị đồ thị", value=latest, key=f"g{message['id']}"):
            G.render(B.answer_subgraph(message["sources"]), height=480, include_sources=True)
    if details:
        with tabs[2]:
            st.caption("Top 20 kết quả của chế độ tìm kiếm; ngân sách ngữ cảnh 8.000 ký tự.")
            st.dataframe(message["ranking"], use_container_width=True, hide_index=True)


def ask(question: str) -> None:
    started = time.time()
    message = {"id": len(st.session_state.messages), "role": "assistant", "persona": persona, "mode": mode}
    with st.chat_message("assistant", avatar=":material/gavel:"):
        try:
            if mode == "graph-global":
                if not B.global_ready():
                    st.warning("Chế độ Tổng hợp toàn cục chưa sẵn sàng: chưa có báo cáo cộng đồng "
                               "(chạy `python -m lc_app.graph.community`). Hãy chọn chế độ khác.")
                    return
                with st.status("Đang tổng hợp từ các báo cáo cộng đồng của đồ thị…", expanded=False) as status:
                    result = B.global_answer(question, persona)
                    status.update(label=f"Đã đọc {result['map_calls']} lô báo cáo, tổng hợp {len(result['points'])} ý",
                                  state="complete")
                message.update(answer=result["answer"], reports=result["report_rows"], points=result["points"],
                               n_reports=result["n_reports"])
            else:
                with st.spinner("Đang tìm căn cứ pháp lý…"):
                    documents, ranking = B.retrieve(mode, question)
                answer = st.write_stream(B.stream_answer(question, documents, persona))
                message.update(answer=answer, sources=B.source_rows(documents), ranking=ranking)
        except Exception as exc:  # noqa: BLE001
            hint = " Chế độ đồ thị cần Neo4j đang chạy (docker compose up -d)." if "graph" in mode else ""
            st.error(f"Không trả lời được: {type(exc).__name__}: {exc}.{hint}")
            return
    message["seconds"] = time.time() - started
    st.session_state.messages.append(message)
    st.rerun()


# ------------------------------------------------------------------ trang Hỏi đáp
if view == "Hỏi đáp":
    if not st.session_state.messages:
        st.markdown('<div class="section-title">Câu hỏi gợi ý</div>', unsafe_allow_html=True)
        columns = st.columns(2)
        for i, example in enumerate(profile["examples"]):
            if columns[i % 2].button(example, key=f"ex{persona}{i}", use_container_width=True,
                                     icon=":material/help:"):
                st.session_state.pending = example
                st.rerun()
        if mode == "graph-global":
            st.info("Chế độ Tổng hợp toàn cục hợp với câu hỏi tổng quát, ví dụ: *Nhà nước được thu hồi đất trong "
                    "những trường hợp nào và mỗi nhóm khác nhau ra sao?*", icon=":material/lightbulb:")
    messages = st.session_state.messages
    for i, m in enumerate(messages):
        if m["role"] == "user":
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant", avatar=":material/gavel:"):
                render_answer(m, latest=i == len(messages) - 1)
    question = st.chat_input("Nhập câu hỏi về đất đai, nhà ở, kinh doanh bất động sản…") or st.session_state.pending
    if question:
        st.session_state.pending = None
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(question)
        ask(question)

# ------------------------------------------------------------------ trang Khám phá đồ thị
elif view == "Khám phá đồ thị":
    st.markdown('<div class="section-title">Khám phá đồ thị tri thức pháp luật</div>'
                '<p class="hint">Tìm một văn bản, Điều, cơ quan hoặc khái niệm để xem các quan hệ trực tiếp của nó '
                '(sửa đổi, hướng dẫn, dẫn chiếu, chứa, áp dụng cho).</p>', unsafe_allow_html=True)
    left, right = st.columns([2, 3])
    query = left.text_input("Tìm văn bản", value="Nghị định 103/2024/NĐ-CP",
                            placeholder="Ví dụ: Luật Đất đai, Điều 79, Giấy chứng nhận…")
    hits = B.search_entities(query)
    if not hits:
        st.info("Không tìm thấy thực thể nào khớp.")
    else:
        name = left.selectbox("Thực thể", hits)
        relations = right.multiselect("Loại quan hệ", list(B.RELATION_STYLE),
                                      default=["SUA_DOI", "HUONG_DAN", "THAM_CHIEU", "CHUA"],
                                      format_func=lambda r: B.RELATION_STYLE[r][1])
        node = B.load_graph()["nodes"][name]
        kind = B.ENTITY_STYLE.get(node["type"], ("", node["type"]))[1]
        left.markdown(f'<div class="report"><b>{name}</b><br><span class="badge accent">{kind}</span> '
                      f'<span class="badge">{len(node["chunk_ids"])} đoạn nguồn</span>'
                      f'<p style="margin:.5rem 0 0">{node["description"]}</p></div>', unsafe_allow_html=True)
        with right:
            G.render(B.ego_subgraph(name, relations), height=520)

# ------------------------------------------------------------------ trang Hướng dẫn
else:
    st.markdown('<div class="section-title">Cách dùng</div>', unsafe_allow_html=True)
    st.markdown(
        "1. **Chọn bạn là ai** ở thanh bên: câu trả lời được trình bày theo nhu cầu của người dân, doanh nghiệp "
        "hoặc cán bộ. Nội dung pháp lý và căn cứ giống nhau, chỉ khác cách trình bày.\n"
        "2. **Đặt câu hỏi** bằng lời của bạn, hoặc bấm một câu hỏi gợi ý.\n"
        "3. **Kiểm tra căn cứ**: số [1], [2] trong câu trả lời trỏ tới đoạn luật ở thẻ *Căn cứ pháp lý*.\n"
        "4. **Xem đồ thị** để thấy văn bản nào sửa đổi, hướng dẫn hay dẫn chiếu văn bản nào.")
    st.markdown('<div class="section-title">Các chế độ tìm kiếm</div>', unsafe_allow_html=True)
    st.dataframe([{"Chế độ": f"{m['label']}", "Dùng khi": m["desc"]} for m in B.MODES.values()],
                 use_container_width=True, hide_index=True)
    st.markdown('<div class="section-title">Lưu ý</div>', unsafe_allow_html=True)
    st.markdown("- Kho dữ liệu chỉ gồm văn bản còn hiệu lực tại ngày chốt; câu trả lời không thay thế văn bản "
                "trả lời chính thức của cơ quan nhà nước.\n"
                "- Chế độ *Tổng hợp toàn cục* gọi mô hình ngôn ngữ nhiều lần nên chậm và tốn chi phí hơn.")
