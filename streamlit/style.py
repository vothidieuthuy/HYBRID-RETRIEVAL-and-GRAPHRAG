"""CSS của giao diện: màu nhấn theo đối tượng, chữ Be Vietnam Pro, tương phản >= 4.5:1, focus nhìn thấy được."""
from __future__ import annotations


def css(accent: str, soft: str, big_font: bool) -> str:
    base = "18px" if big_font else "16px"
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
:root {{ --accent:{accent}; --soft:{soft}; --ink:#0F172A; --muted:#475569; --line:#E2E8F0; --card:#FFFFFF; }}
html {{ font-size:{base}; }}
html, body, [class*="st-"], .stMarkdown, button, input, textarea {{ font-family:'Be Vietnam Pro', system-ui, sans-serif; }}
/* icon Material của Streamlit (span translate="no"; alert/toast đổi data-testid) phải giữ font icon */
.material-symbols-rounded, span[translate="no"], [data-testid="stIconMaterial"], [data-testid="stAlertDynamicIcon"],
[data-testid="stToastDynamicIcon"], [data-testid="stElementToolbarButtonIcon"] {{ font-family:'Material Symbols Rounded' !important; }}
[data-testid="stAppViewContainer"] > .main .block-container, .block-container {{ padding-top:1.4rem; max-width:1180px; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"] {{ background:#FFFFFF; border-right:1px solid var(--line); }}
[data-testid="stSidebar"] .block-container {{ padding-top:1rem; }}

.brand {{ display:flex; gap:.7rem; align-items:center; margin:.2rem 0 1rem; }}
.brand .logo {{ width:42px; height:42px; border-radius:12px; background:var(--accent); display:grid; place-items:center; flex:none; }}
.brand b {{ display:block; color:var(--ink); font-size:1.02rem; line-height:1.25; }}
.brand small {{ color:var(--muted); font-size:.8rem; }}

.hero {{ background:var(--card); border:1px solid var(--line); border-left:6px solid var(--accent); border-radius:18px;
        padding:1.3rem 1.5rem 1.1rem; margin-bottom:1rem; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.05); }}
.hero .who {{ display:inline-flex; align-items:center; gap:.4rem; background:var(--soft); color:var(--accent); font-weight:600;
             font-size:.82rem; padding:.22rem .7rem; border-radius:999px; margin-bottom:.55rem; }}
.hero h1 {{ font-size:1.75rem; line-height:1.25; margin:0 0 .35rem; color:var(--ink); font-weight:700; padding:0; }}
.hero p {{ margin:0; color:var(--muted); font-size:1rem; }}
.chips {{ display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.85rem; }}
.chip {{ display:inline-flex; align-items:center; gap:.35rem; border:1px solid var(--line); background:#F8FAFC; color:#1E293B;
        border-radius:999px; padding:.2rem .7rem; font-size:.8rem; }}
.chip.on {{ border-color:var(--accent); color:var(--accent); background:var(--soft); font-weight:600; }}

/* thanh chuyển trang (radio ngang) hiển thị như tab */
.main [role="radiogroup"][aria-orientation="horizontal"], section.main [data-testid="stRadio"] > div[role="radiogroup"] {{ gap:.4rem; }}
[data-testid="stMain"] [data-testid="stRadio"] div[role="radiogroup"]:has(> label:nth-child(3)):not(:has(> label:nth-child(4))) > label {{
  background:#FFFFFF; border:1px solid var(--line); border-radius:999px; padding:.35rem 1rem; margin:0; cursor:pointer; }}
[data-testid="stMain"] [data-testid="stRadio"] div[role="radiogroup"]:has(> label:nth-child(3)):not(:has(> label:nth-child(4))) > label:has(input:checked) {{
  background:var(--accent); border-color:var(--accent); }}
[data-testid="stMain"] [data-testid="stRadio"] div[role="radiogroup"]:has(> label:nth-child(3)):not(:has(> label:nth-child(4))) > label:has(input:checked) p {{
  color:#FFFFFF; font-weight:600; }}
[data-testid="stMain"] [data-testid="stRadio"] div[role="radiogroup"]:has(> label:nth-child(3)):not(:has(> label:nth-child(4))) > label > div:first-child {{ display:none; }}

.section-title {{ font-weight:700; color:var(--ink); font-size:1.02rem; margin:.4rem 0 .5rem; }}
.hint {{ color:var(--muted); font-size:.9rem; }}
div.stButton > button {{ border-radius:12px; border:1px solid var(--line); background:#FFFFFF; color:var(--ink);
                         text-align:left; justify-content:flex-start; min-height:3.1rem; white-space:normal; line-height:1.35;
                         transition:border-color .15s, box-shadow .15s; cursor:pointer; }}
div.stButton > button:hover {{ border-color:var(--accent); box-shadow:0 0 0 3px var(--soft); color:var(--ink); }}
div.stButton > button:focus-visible {{ outline:3px solid var(--accent); outline-offset:2px; }}
div.stButton > button[kind="primary"] {{ background:var(--accent); color:#FFFFFF; border-color:var(--accent); justify-content:center; }}

[data-testid="stChatMessage"] {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:.9rem 1rem; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ background:var(--soft); border-color:transparent; }}
[data-testid="stChatInput"] {{ border-radius:16px; }}
[data-testid="stChatInput"] textarea {{ font-size:1rem; }}
.meta {{ display:flex; flex-wrap:wrap; gap:.4rem; margin-bottom:.4rem; }}
.badge {{ font-size:.75rem; font-weight:600; padding:.12rem .55rem; border-radius:6px; background:#F1F5F9; color:#334155; }}
.badge.accent {{ background:var(--soft); color:var(--accent); }}
.badge.red {{ background:#FEE2E2; color:#991B1B; }}
.src-note {{ background:#FFF7ED; border-left:3px solid #EA580C; padding:.45rem .7rem; border-radius:6px; font-size:.88rem; color:#7C2D12; margin:.2rem 0 .5rem; }}
.src-graph {{ background:#FEF2F2; border-left:3px solid #DC2626; padding:.45rem .7rem; border-radius:6px; font-size:.88rem; color:#7F1D1D; margin:.2rem 0 .5rem; }}
.report {{ border:1px solid var(--line); border-radius:12px; padding:.75rem .9rem; margin-bottom:.6rem; background:#FFFFFF; }}
.report b {{ color:var(--ink); }}
.report .r {{ float:right; font-size:.75rem; color:var(--muted); }}
.disclaimer {{ font-size:.78rem; color:var(--muted); border-top:1px solid var(--line); padding-top:.7rem; margin-top:.8rem; }}
.stat {{ display:flex; justify-content:space-between; font-size:.85rem; color:#334155; padding:.18rem 0; }}
.stat b {{ color:var(--ink); }}
[data-testid="stExpander"] details {{ border-radius:12px; border-color:var(--line); }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>"""


LOGO = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v18"/><path d="M5 7h14"/>
<path d="M5 7l-3 7a4 4 0 0 0 6 0z"/><path d="M19 7l-3 7a4 4 0 0 0 6 0z"/><path d="M8 21h8"/></svg>"""
