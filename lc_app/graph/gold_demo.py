"""Chấm đồ thị demo (9 chunk tách thửa, hợp thửa) với bộ quan hệ chuẩn đọc tay từ văn bản.

GOLD: quan hệ PHẢI có (loại đúng, chiều đúng); WRONG: quan hệ sai đã biết (đảo chiều, sai loại).
Mỗi đầu mút là tập tên chấp nhận được (văn bản hoặc Điều của nó). Đọc graph.json trong LC_GRAPH_DIR.

    .venv-lc/Scripts/python -m lc_app.graph.gold_demo
"""
from __future__ import annotations

import json

from ..config import Config

LDD, ND101, ND49, NQ254 = ("Luật Đất đai 31/2024/QH15", "Nghị định 101/2024/NĐ-CP", "Nghị định 49/2026/NĐ-CP",
                           "Nghị quyết 254/2025/QH15")
QD75, QD135 = "Quyết định 75/2026/QĐ-UBND", "Quyết định 135/2026/QĐ-UBND"
D220 = f"Điều 220 {LDD}"
D1_101, D7_101, D12_49, D11_254 = f"Điều 1 {ND101}", f"Điều 7 {ND101}", f"Điều 12 {ND49}", f"Điều 11 {NQ254}"
D1_75, D5_75, D18_135 = (f"Điều 1 Quy định kèm theo {QD75}", f"Điều 5 Quy định kèm theo {QD75}",
                         f"Điều 18 Quy định kèm theo {QD135}")
UBND, VPDK = "Ủy ban nhân dân cấp tỉnh", "Văn phòng đăng ký đất đai"


def a(*names: str) -> set[str]:
    return {n.casefold() for n in names}


GOLD = [  # (nguồn chấp nhận, loại, đích chấp nhận, ghi chú)
    (a(LDD), "CHUA", a(D220), "LĐĐ chứa Điều 220"),
    (a(ND101), "CHUA", a(D1_101), "NĐ 101 chứa Điều 1"),
    (a(ND101), "CHUA", a(D7_101), "NĐ 101 chứa Điều 7"),
    (a(ND49), "CHUA", a(D12_49), "NĐ 49 chứa Điều 12"),
    (a(NQ254), "CHUA", a(D11_254), "NQ 254 chứa Điều 11"),
    (a(QD75), "CHUA", a(D1_75), "QĐ 75 chứa Điều 1 QĐKT"),
    (a(QD75), "CHUA", a(D5_75), "QĐ 75 chứa Điều 5 QĐKT"),
    (a(QD135), "CHUA", a(D18_135), "QĐ 135 chứa Điều 18 QĐKT"),
    (a(ND101, D1_101), "HUONG_DAN", a(LDD), "NĐ 101 hướng dẫn LĐĐ (Điều 1)"),
    (a(ND49, D12_49), "SUA_DOI", a(ND101, f"Điều 18 {ND101}"), "NĐ 49 sửa đổi NĐ 101"),
    (a("Nghị định 151/2025/NĐ-CP"), "SUA_DOI", a(ND101), "NĐ 151 sửa đổi NĐ 101 (đúng chiều)"),
    (a("Nghị định 226/2025/NĐ-CP"), "SUA_DOI", a(ND101), "NĐ 226 sửa đổi NĐ 101 (đúng chiều)"),
    (a(D7_101, ND101), "THAM_CHIEU", a(D220, LDD), "Điều 7 NĐ 101 dẫn Điều 220"),
    (a(D11_254, NQ254), "THAM_CHIEU", a(D220, LDD), "Điều 11 NQ 254 dẫn Điều 220"),
    (a(D1_75), "THAM_CHIEU", a(D220, LDD), "Điều 1 QĐ 75 dẫn Điều 220"),
    (a(D5_75), "THAM_CHIEU", a(D220, LDD), "Điều 5 QĐ 75 dẫn Điều 220"),
    (a(D5_75), "THAM_CHIEU", a(D11_254, NQ254), "Điều 5 QĐ 75 dẫn khoản 3 Điều 11 NQ 254"),
    (a(D5_75), "THAM_CHIEU", a("Luật Nhà ở"), "Điều 5 QĐ 75 dẫn Luật Nhà ở"),
    (a(D18_135), "THAM_CHIEU", a(D220, LDD), "Điều 18 QĐ 135 dẫn Điều 220"),
    (a(D18_135), "THAM_CHIEU", a(D11_254, NQ254), "Điều 18 QĐ 135 dẫn khoản 3 Điều 11 NQ 254"),
    (a(D220), "AP_DUNG_CHO", a("tách thửa đất", "hợp thửa đất", "tách thửa đất, hợp thửa đất"),
     "Điều 220 áp dụng cho tách/hợp thửa"),
    (a(D220), "AP_DUNG_CHO", a(UBND), "Điều 220 K4: UBND cấp tỉnh quy định chi tiết"),
    (a(D7_101), "AP_DUNG_CHO", a(VPDK, "Chi nhánh Văn phòng đăng ký đất đai"), "Điều 7 NĐ 101: VPĐKĐĐ kiểm tra"),
    (a(D18_135), "AP_DUNG_CHO", a(VPDK, "Chi nhánh Văn phòng đăng ký đất đai"), "Điều 18 QĐ 135: VPĐKĐĐ kiểm tra"),
]
WRONG = [
    (a(ND101, D1_101, D7_101), "SUA_DOI", a("Nghị định 151/2025/NĐ-CP", "Nghị định 226/2025/NĐ-CP"),
     "đảo chiều: NĐ 101 sửa đổi NĐ 151/226"),
    (a(D5_75, QD75), "HUONG_DAN", None, "Điều 5 QĐ 75 'hướng dẫn' (thực ra là tham chiếu)"),
    (a(D12_49, ND49), "THAM_CHIEU", a(ND101), "NĐ 49 sửa đổi NĐ 101 nhưng gán THAM_CHIEU"),
]


def score(graph: dict) -> dict:
    edges = {(e["source"].casefold(), e["type"], e["target"].casefold()) for e in graph["edges"]}
    hit = lambda s, t, d: any(es in s and et == t and (d is None or ed in d) for es, et, ed in edges)  # noqa: E731
    found = [note for s, t, d, note in GOLD if hit(s, t, d)]
    missed = [note for s, t, d, note in GOLD if not hit(s, t, d)]
    wrong = [note for s, t, d, note in WRONG if hit(s, t, d)]
    return {"gold": len(GOLD), "found": len(found), "missed": missed, "wrong": wrong}


def main() -> None:
    config = Config.from_env()
    graph = json.loads((config.graph_dir / "graph.json").read_text(encoding="utf-8"))
    result = score(graph)
    print(f"{config.graph_dir.name}: đúng {result['found']}/{result['gold']} quan hệ chuẩn, "
          f"{len(result['wrong'])} quan hệ sai đã biết; {len(graph['nodes'])} nút, {len(graph['edges'])} cạnh")
    for note in result["missed"]:
        print("   thiếu:", note)
    for note in result["wrong"]:
        print("   SAI  :", note)


if __name__ == "__main__":
    main()
