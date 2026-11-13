"""Ontology tối giản (LEGAL_ONTOLOGY.md v2.0) và prompt trích xuất theo khuôn của Microsoft GraphRAG.

Prompt giữ cấu trúc GRAPH_EXTRACTION_PROMPT của Microsoft (-Mục tiêu- / -Các bước- / -Ví dụ- / -Dữ liệu thật-, tuple
<|>, ## và <|COMPLETE|>, một tin nhắn user) với hai chỗ lệch có chủ đích: loại nút cố định và trường LOẠI_QUAN_HỆ.
Ví dụ few-shot viết tay từ chunk của KB nằm ngoài bộ demo (Điều 195 Luật Đất đai, Điều 1 NĐ 102/2024, Điều 3 NĐ 50/2026
cho câu bị động "được sửa đổi, bổ sung bởi").
"""
from __future__ import annotations

import re

ENTITY_TYPES = ("VAN_BAN", "DIEU_KHOAN", "CO_QUAN", "KHAI_NIEM")
RELATION_TYPES = ("CHUA", "HUONG_DAN", "THAM_CHIEU", "SUA_DOI", "AP_DUNG_CHO")
# miền -> đích của từng quan hệ (LEGAL_ONTOLOGY.md mục 3)
DOMAIN_RANGE = {
    "CHUA": ({"VAN_BAN"}, {"DIEU_KHOAN"}),
    "HUONG_DAN": ({"VAN_BAN", "DIEU_KHOAN"}, {"VAN_BAN"}),
    "THAM_CHIEU": ({"VAN_BAN", "DIEU_KHOAN"}, {"VAN_BAN", "DIEU_KHOAN"}),
    "SUA_DOI": ({"VAN_BAN", "DIEU_KHOAN"}, {"VAN_BAN", "DIEU_KHOAN"}),
    "AP_DUNG_CHO": ({"DIEU_KHOAN"}, {"KHAI_NIEM", "CO_QUAN"}),
}
RELATION_LABELS = {"CHUA": "chứa", "HUONG_DAN": "hướng dẫn", "THAM_CHIEU": "tham chiếu", "SUA_DOI": "sửa đổi",
                   "AP_DUNG_CHO": "áp dụng cho"}
# cụm chung chung không tạo nút (quy tắc 4.4); so sánh sau khi chuẩn hóa chữ thường
GENERIC_NAMES = frozenset({"nhà nước", "cơ quan có thẩm quyền", "cơ quan nhà nước", "cơ quan nhà nước có thẩm quyền",
                           "các cơ quan liên quan", "cơ quan liên quan", "pháp luật", "tổ chức, cá nhân",
                           "tổ chức và cá nhân", "trường hợp khác", "điều này", "luật này", "nghị định này"})

DOC_KINDS = r"(Bộ luật|Luật|Nghị quyết|Nghị định|Thông tư liên tịch|Thông tư|Quyết định|Chỉ thị)"
# tên hợp lệ của nút DIEU_KHOAN: "Điều <số> [Quy định kèm theo] <văn bản>" hoặc "Phụ lục [<số>] <văn bản>" (quy tắc 3)
UNIT_NAME = re.compile(rf"^(?:Điều \d+[a-z]?|Phụ lục(?: [IVXLC\d]+[a-z]?)?) (?:Quy định kèm theo )?{DOC_KINDS} \S")
# tên hợp lệ của nút VAN_BAN: có số hiệu, hoặc Luật/Bộ luật/Hiến pháp có tên gọi riêng ("Bộ luật Dân sự")
DOCUMENT_NAME = re.compile(r"^(?:Hiến pháp|Bộ luật \S|Luật \S)|\d+/(?:\d{4}/)?[\wĐđ-]+")

TUPLE, RECORD, COMPLETE = "<|>", "##", "<|COMPLETE|>"

EXTRACTION_PROMPT = """-Mục tiêu-
Cho một đoạn văn bản pháp luật và danh sách loại thực thể, hãy xác định mọi thực thể thuộc các loại đó trong đoạn văn và
mọi quan hệ giữa các thực thể đã xác định.

-Loại thực thể-
VAN_BAN: văn bản được nêu đích danh (Luật, Bộ luật, Nghị quyết, Nghị định, Thông tư, Quyết định...).
DIEU_KHOAN: một Điều cụ thể của một văn bản (khoản, điểm ghi trong mô tả, không tạo thực thể riêng).
CO_QUAN: cơ quan, tổ chức có thẩm quyền hoặc thực hiện thủ tục.
KHAI_NIEM: mọi đối tượng pháp lý khác mà Điều quy định: thủ tục, chủ thể, loại đất, tài sản, giấy tờ, quyền, nghĩa vụ,
chế độ, hạn mức, tiền.

-Loại quan hệ (có hướng NGUỒN -> ĐÍCH)-
CHUA: VAN_BAN -> DIEU_KHOAN; văn bản có Điều này (đọc từ dòng tiêu đề [ ... > Điều X] hoặc "Điều X." đầu đoạn).
HUONG_DAN: VAN_BAN hoặc DIEU_KHOAN -> VAN_BAN; quy định chi tiết, hướng dẫn thi hành. Luôn trỏ tới VĂN BẢN được hướng
dẫn; các Điều được hướng dẫn ghi trong mô tả.
THAM_CHIEU: VAN_BAN hoặc DIEU_KHOAN -> VAN_BAN hoặc DIEU_KHOAN; dẫn chiếu "theo quy định tại", "quy định tại".
"Theo quy định tại Điều 45 Luật Đất đai" là THAM_CHIEU, không phải HUONG_DAN. Đoạn văn nêu số Điều ("Điều 220 của
Luật Đất đai", "khoản 3 Điều 11 của Nghị quyết ...") thì THAM_CHIEU trỏ tới thực thể Điều đó, KHÔNG trỏ tới văn bản.
SUA_DOI: VAN_BAN hoặc DIEU_KHOAN -> VAN_BAN hoặc DIEU_KHOAN; sửa đổi, bổ sung, bãi bỏ hoặc thay thế (ghi rõ hình thức).
Câu "Sửa đổi, bổ sung (một số điều của) X" hay "bãi bỏ ... của X" là SUA_DOI tới X, KHÔNG phải THAM_CHIEU.
NGUỒN luôn là văn bản đi sửa, ĐÍCH là văn bản bị sửa. Câu bị động "X (đã) được sửa đổi, bổ sung bởi/tại Y" là SUA_DOI
từ Y tới X, không phải từ X tới Y, và cũng không phải từ văn bản của đoạn văn tới Y. "X (được sửa đổi, bổ sung tại Y,
Z)" thì tạo quan hệ SUA_DOI cho MỖI văn bản Y, Z.
AP_DUNG_CHO: DIEU_KHOAN -> KHAI_NIEM hoặc CO_QUAN; Điều quy định về đối tượng đó.
Chỉ tạo quan hệ đúng cặp loại nêu trên (KHAI_NIEM, CO_QUAN chỉ là đích của AP_DUNG_CHO). Không tạo quan hệ giữa một Điều
chỉ được nhắc tới và văn bản của nó (tên Điều đã chứa tên văn bản).

-Quy tắc đặt tên (bắt buộc, để cùng một đối tượng luôn có cùng một tên)-
1. Luật, Bộ luật: "<Loại> <tên gọi> <số hiệu>", ví dụ "Luật Đất đai 31/2024/QH15". Loại văn bản khác: "<Loại> <số
   hiệu>", KHÔNG kèm trích yếu, ví dụ "Nghị định 102/2024/NĐ-CP", "Quyết định 75/2026/QĐ-UBND". Bỏ chữ "số": "Nghị
   quyết số 254/2025/QH15" ghi "Nghị quyết 254/2025/QH15". Đoạn văn không có số hiệu thì ghi như đoạn văn viết ("Bộ
   luật Dân sự"); không tự thêm số hiệu từ hiểu biết bên ngoài.
2. "Luật này", "Nghị định này", "Quyết định này" thay bằng tên văn bản trong dòng tiêu đề.
3. Điều: "Điều <số> <tên văn bản>". Nếu dòng tiêu đề có "> Quy định kèm theo >" thì MỌI Điều của đoạn văn đó ghi
   "Điều <số> Quy định kèm theo <tên văn bản>". "Điều này" thay bằng Điều trong dòng tiêu đề. KHÔNG tạo thực thể cho
   Khoản hay Điểm ("Khoản 3 Điều 18 ..." ghi là "Điều 18 ..."; không viết "Điều 220.1"; khoản, điểm ghi trong mô tả).
4. Khái niệm, cơ quan viết đầy đủ, không viết tắt ("Ủy ban nhân dân cấp tỉnh", không "UBND tỉnh"). Dùng tên đầy đủ
   nhất mà đoạn văn dùng cho đối tượng đó ("Giấy chứng nhận quyền sử dụng đất", không rút thành "Giấy chứng nhận";
   "lối đi kết nối với đường giao thông công cộng" và "lối đi" là một đối tượng thì chỉ dùng một tên).
5. Không tạo thực thể cho cụm chung chung: "Nhà nước", "cơ quan có thẩm quyền", "cơ quan nhà nước", "tổ chức, cá nhân".
6. Chỉ dùng thông tin có trong đoạn văn; không có trong đoạn văn thì không tạo quan hệ.

-Quy tắc viết mô tả (bắt buộc)-
1. Mô tả chỉ nêu điều ĐOẠN VĂN NÀY nói về thực thể (nó được quy định thế nào, ai làm gì với nó, điều kiện, thời hạn,
   số liệu). Không dùng hiểu biết bên ngoài, không tự viết định nghĩa nếu đoạn văn không định nghĩa.
2. Điều hoặc văn bản chỉ được nhắc tới bằng số hiệu thì mô tả chỉ ghi nó được nhắc tới thế nào. Chỉ ghi nội dung khi
   CHÍNH CÂU nhắc tới nó nêu nội dung đó ("Hạn mức công nhận đất ở theo quy định tại khoản 5 Điều 141" -> "khoản 5 Điều
   141 được dẫn chiếu về hạn mức công nhận đất ở"). Khi đoạn văn chỉ liệt kê số Điều thì KHÔNG đoán nội dung:
   sai: "Điều 9 quy định về phân loại đất"; đúng: "Được Nghị định 102/2024/NĐ-CP quy định chi tiết, hướng dẫn thi hành".

-Các bước-
1. Xác định mọi thực thể. Với mỗi thực thể, lấy:
- tên_thực_thể: theo quy tắc đặt tên
- loại_thực_thể: một trong [{entity_types}]
- mô_tả_thực_thể: mô tả đầy đủ thuộc tính và vai trò của thực thể, chỉ dựa trên đoạn văn
Định dạng mỗi thực thể: ("entity"{tuple_delimiter}<tên_thực_thể>{tuple_delimiter}<loại_thực_thể>{tuple_delimiter}<mô_tả_thực_thể>)

2. Từ các thực thể ở bước 1, xác định mọi cặp (nguồn, đích) có quan hệ rõ ràng. Với mỗi cặp, lấy:
- thực_thể_nguồn, thực_thể_đích: tên như ở bước 1
- loại_quan_hệ: một trong [{relation_types}]
- mô_tả_quan_hệ: vì sao hai thực thể có quan hệ; ghi khoản, điểm liên quan
- độ_mạnh_quan_hệ: số từ 1 đến 10
Định dạng mỗi quan hệ: ("relationship"{tuple_delimiter}<thực_thể_nguồn>{tuple_delimiter}<thực_thể_đích>{tuple_delimiter}<loại_quan_hệ>{tuple_delimiter}<mô_tả_quan_hệ>{tuple_delimiter}<độ_mạnh_quan_hệ>)

3. Trả về một danh sách gồm mọi thực thể và quan hệ ở bước 1 và 2, dùng **{record_delimiter}** làm dấu ngăn cách.

4. Khi xong, ghi {completion_delimiter}

######################
-Ví dụ-
######################
Ví dụ 1:
Loại thực thể: [{entity_types}]
Văn bản:
[Luật Đất đai 31/2024/QH15 (hiệu lực từ 01/08/2024) > Chương XIII. CHẾ ĐỘ SỬ DỤNG ĐẤT > Mục 2. CHẾ ĐỘ SỬ DỤNG ĐẤT > Điều 195. Đất ở tại nông thôn]
Điều 195. Đất ở tại nông thôn
1. Đất ở tại nông thôn là đất để làm nhà ở và mục đích khác phục vụ cho đời sống trong cùng một thửa đất thuộc khu vực nông thôn.
2. Căn cứ vào quỹ đất và tình hình thực tế của địa phương, Ủy ban nhân dân cấp tỉnh quy định hạn mức giao đất ở cho cá nhân tại nông thôn.
4. Nhà nước có chính sách tạo điều kiện cho những người sống ở nông thôn có chỗ ở trên cơ sở tận dụng đất trong những khu dân cư sẵn có, hạn chế việc mở rộng khu dân cư trên đất nông nghiệp.
######################
Kết quả:
("entity"{tuple_delimiter}Luật Đất đai 31/2024/QH15{tuple_delimiter}VAN_BAN{tuple_delimiter}Luật Đất đai, có hiệu lực từ 01/08/2024, có Chương XIII về chế độ sử dụng đất)
{record_delimiter}
("entity"{tuple_delimiter}Điều 195 Luật Đất đai 31/2024/QH15{tuple_delimiter}DIEU_KHOAN{tuple_delimiter}Điều 195 quy định về đất ở tại nông thôn: khái niệm, hạn mức giao đất ở cho cá nhân và chính sách chỗ ở tại nông thôn)
{record_delimiter}
("entity"{tuple_delimiter}đất ở tại nông thôn{tuple_delimiter}KHAI_NIEM{tuple_delimiter}Đất để làm nhà ở và mục đích khác phục vụ cho đời sống trong cùng một thửa đất thuộc khu vực nông thôn)
{record_delimiter}
("entity"{tuple_delimiter}hạn mức giao đất ở cho cá nhân tại nông thôn{tuple_delimiter}KHAI_NIEM{tuple_delimiter}Hạn mức giao đất ở cho cá nhân tại nông thôn, được quy định căn cứ vào quỹ đất và tình hình thực tế của địa phương)
{record_delimiter}
("entity"{tuple_delimiter}Ủy ban nhân dân cấp tỉnh{tuple_delimiter}CO_QUAN{tuple_delimiter}Cơ quan quy định hạn mức giao đất ở cho cá nhân tại nông thôn)
{record_delimiter}
("relationship"{tuple_delimiter}Luật Đất đai 31/2024/QH15{tuple_delimiter}Điều 195 Luật Đất đai 31/2024/QH15{tuple_delimiter}CHUA{tuple_delimiter}Điều 195 thuộc Mục 2 Chương XIII của Luật Đất đai{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}Điều 195 Luật Đất đai 31/2024/QH15{tuple_delimiter}đất ở tại nông thôn{tuple_delimiter}AP_DUNG_CHO{tuple_delimiter}Khoản 1 định nghĩa đất ở tại nông thôn{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}Điều 195 Luật Đất đai 31/2024/QH15{tuple_delimiter}hạn mức giao đất ở cho cá nhân tại nông thôn{tuple_delimiter}AP_DUNG_CHO{tuple_delimiter}Khoản 2 quy định thẩm quyền và căn cứ quy định hạn mức{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}Điều 195 Luật Đất đai 31/2024/QH15{tuple_delimiter}Ủy ban nhân dân cấp tỉnh{tuple_delimiter}AP_DUNG_CHO{tuple_delimiter}Khoản 2: Ủy ban nhân dân cấp tỉnh quy định hạn mức giao đất ở cho cá nhân tại nông thôn{tuple_delimiter}8)
{completion_delimiter}
######################
Ví dụ 2:
Loại thực thể: [{entity_types}]
Văn bản:
[Nghị định 102/2024/NĐ-CP quy định chi tiết thi hành một số điều của Luật Đất đai (hiệu lực từ 01/08/2024) > Chương I. QUY ĐỊNH CHUNG > Điều 1. Phạm vi điều chỉnh]
Điều 1. Phạm vi điều chỉnh
Nghị định này quy định chi tiết, hướng dẫn thi hành khoản 6 Điều 3, Điều 9, khoản 2 Điều 10, Điều 16, ..., Điều 219, Điều 223, khoản 8 Điều 234 Luật Đất đai.
######################
Kết quả:
("entity"{tuple_delimiter}Nghị định 102/2024/NĐ-CP{tuple_delimiter}VAN_BAN{tuple_delimiter}Nghị định quy định chi tiết thi hành một số điều của Luật Đất đai, có hiệu lực từ 01/08/2024)
{record_delimiter}
("entity"{tuple_delimiter}Điều 1 Nghị định 102/2024/NĐ-CP{tuple_delimiter}DIEU_KHOAN{tuple_delimiter}Điều 1 quy định phạm vi điều chỉnh của Nghị định 102/2024/NĐ-CP)
{record_delimiter}
("entity"{tuple_delimiter}Luật Đất đai{tuple_delimiter}VAN_BAN{tuple_delimiter}Luật được Nghị định 102/2024/NĐ-CP quy định chi tiết, hướng dẫn thi hành)
{record_delimiter}
("entity"{tuple_delimiter}Điều 9 Luật Đất đai{tuple_delimiter}DIEU_KHOAN{tuple_delimiter}Được Nghị định 102/2024/NĐ-CP quy định chi tiết, hướng dẫn thi hành; đoạn văn không nêu nội dung của Điều này)
{record_delimiter}
("relationship"{tuple_delimiter}Điều 1 Nghị định 102/2024/NĐ-CP{tuple_delimiter}Điều 9 Luật Đất đai{tuple_delimiter}THAM_CHIEU{tuple_delimiter}Điều 1 liệt kê Điều 9 trong phạm vi được Nghị định quy định chi tiết, hướng dẫn thi hành{tuple_delimiter}6)
{record_delimiter}
("relationship"{tuple_delimiter}Nghị định 102/2024/NĐ-CP{tuple_delimiter}Điều 1 Nghị định 102/2024/NĐ-CP{tuple_delimiter}CHUA{tuple_delimiter}Điều 1 thuộc Chương I của Nghị định{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}Nghị định 102/2024/NĐ-CP{tuple_delimiter}Luật Đất đai{tuple_delimiter}HUONG_DAN{tuple_delimiter}Quy định chi tiết, hướng dẫn thi hành khoản 6 Điều 3, Điều 9, khoản 2 Điều 10, Điều 16, ..., Điều 223, khoản 8 Điều 234 Luật Đất đai{tuple_delimiter}10)
{completion_delimiter}
######################
Ví dụ 3:
Loại thực thể: [{entity_types}]
Văn bản:
[Nghị định 50/2026/NĐ-CP quy định chi tiết một số điều của Nghị quyết số 254/2025/QH15 về tiền sử dụng đất, tiền thuê đất (hiệu lực từ 31/01/2026) > Chương II. QUY ĐỊNH CỤ THỂ > Điều 3. Căn cứ tính tiền sử dụng đất quy định tại khoản 1 Điều 5 Nghị quyết số 254/2025/QH15 > Khoản 1-2]
Điều 3. Căn cứ tính tiền sử dụng đất quy định tại khoản 1 Điều 5 Nghị quyết số 254/2025/QH15
1. Diện tích đất tính tiền sử dụng đất theo quy định tại Điều 4 Nghị định số 103/2024/NĐ-CP ngày 30 tháng 7 năm 2024 của Chính phủ quy định về tiền sử dụng đất, tiền thuê đất (được sửa đổi, bổ sung bởi Nghị định số 291/2025/NĐ-CP).
2. Giá đất tính tiền sử dụng đất là giá đất được xác định theo quy định tại Điều 7 và Điều 8 Nghị quyết số 254/2025/QH15 và quy định tại khoản 1 Điều 13 Nghị định này.
######################
Kết quả:
("entity"{tuple_delimiter}Nghị định 50/2026/NĐ-CP{tuple_delimiter}VAN_BAN{tuple_delimiter}Nghị định quy định chi tiết một số điều của Nghị quyết 254/2025/QH15 về tiền sử dụng đất, tiền thuê đất, có hiệu lực từ 31/01/2026)
{record_delimiter}
("entity"{tuple_delimiter}Điều 3 Nghị định 50/2026/NĐ-CP{tuple_delimiter}DIEU_KHOAN{tuple_delimiter}Điều 3 quy định căn cứ tính tiền sử dụng đất theo khoản 1 Điều 5 Nghị quyết 254/2025/QH15: diện tích đất (khoản 1) và giá đất (khoản 2) tính tiền sử dụng đất)
{record_delimiter}
("entity"{tuple_delimiter}Nghị quyết 254/2025/QH15{tuple_delimiter}VAN_BAN{tuple_delimiter}Nghị quyết được Nghị định 50/2026/NĐ-CP quy định chi tiết một số điều)
{record_delimiter}
("entity"{tuple_delimiter}Điều 5 Nghị quyết 254/2025/QH15{tuple_delimiter}DIEU_KHOAN{tuple_delimiter}Khoản 1 Điều 5 được dẫn chiếu về căn cứ tính tiền sử dụng đất)
{record_delimiter}
("entity"{tuple_delimiter}Điều 7 Nghị quyết 254/2025/QH15{tuple_delimiter}DIEU_KHOAN{tuple_delimiter}Được dẫn chiếu, cùng Điều 8, về việc xác định giá đất tính tiền sử dụng đất)
{record_delimiter}
("entity"{tuple_delimiter}Điều 8 Nghị quyết 254/2025/QH15{tuple_delimiter}DIEU_KHOAN{tuple_delimiter}Được dẫn chiếu, cùng Điều 7, về việc xác định giá đất tính tiền sử dụng đất)
{record_delimiter}
("entity"{tuple_delimiter}Điều 13 Nghị định 50/2026/NĐ-CP{tuple_delimiter}DIEU_KHOAN{tuple_delimiter}Khoản 1 Điều 13 được dẫn chiếu về việc xác định giá đất tính tiền sử dụng đất)
{record_delimiter}
("entity"{tuple_delimiter}Nghị định 103/2024/NĐ-CP{tuple_delimiter}VAN_BAN{tuple_delimiter}Nghị định của Chính phủ quy định về tiền sử dụng đất, tiền thuê đất, ban hành ngày 30/7/2024, đã được Nghị định 291/2025/NĐ-CP sửa đổi, bổ sung)
{record_delimiter}
("entity"{tuple_delimiter}Điều 4 Nghị định 103/2024/NĐ-CP{tuple_delimiter}DIEU_KHOAN{tuple_delimiter}Được dẫn chiếu về diện tích đất tính tiền sử dụng đất)
{record_delimiter}
("entity"{tuple_delimiter}Nghị định 291/2025/NĐ-CP{tuple_delimiter}VAN_BAN{tuple_delimiter}Nghị định sửa đổi, bổ sung Nghị định 103/2024/NĐ-CP)
{record_delimiter}
("entity"{tuple_delimiter}diện tích đất tính tiền sử dụng đất{tuple_delimiter}KHAI_NIEM{tuple_delimiter}Xác định theo quy định tại Điều 4 Nghị định 103/2024/NĐ-CP)
{record_delimiter}
("entity"{tuple_delimiter}giá đất tính tiền sử dụng đất{tuple_delimiter}KHAI_NIEM{tuple_delimiter}Giá đất được xác định theo Điều 7, Điều 8 Nghị quyết 254/2025/QH15 và khoản 1 Điều 13 Nghị định 50/2026/NĐ-CP)
{record_delimiter}
("relationship"{tuple_delimiter}Nghị định 50/2026/NĐ-CP{tuple_delimiter}Điều 3 Nghị định 50/2026/NĐ-CP{tuple_delimiter}CHUA{tuple_delimiter}Điều 3 thuộc Chương II của Nghị định{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}Điều 3 Nghị định 50/2026/NĐ-CP{tuple_delimiter}Điều 5 Nghị quyết 254/2025/QH15{tuple_delimiter}THAM_CHIEU{tuple_delimiter}Tiêu đề Điều 3 dẫn chiếu khoản 1 Điều 5 về căn cứ tính tiền sử dụng đất{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}Điều 3 Nghị định 50/2026/NĐ-CP{tuple_delimiter}Điều 4 Nghị định 103/2024/NĐ-CP{tuple_delimiter}THAM_CHIEU{tuple_delimiter}Khoản 1: diện tích đất tính tiền sử dụng đất theo Điều 4{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}Điều 3 Nghị định 50/2026/NĐ-CP{tuple_delimiter}Điều 7 Nghị quyết 254/2025/QH15{tuple_delimiter}THAM_CHIEU{tuple_delimiter}Khoản 2: giá đất xác định theo Điều 7{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}Điều 3 Nghị định 50/2026/NĐ-CP{tuple_delimiter}Điều 8 Nghị quyết 254/2025/QH15{tuple_delimiter}THAM_CHIEU{tuple_delimiter}Khoản 2: giá đất xác định theo Điều 8{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}Điều 3 Nghị định 50/2026/NĐ-CP{tuple_delimiter}Điều 13 Nghị định 50/2026/NĐ-CP{tuple_delimiter}THAM_CHIEU{tuple_delimiter}Khoản 2: giá đất xác định theo khoản 1 Điều 13 Nghị định này{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}Nghị định 291/2025/NĐ-CP{tuple_delimiter}Nghị định 103/2024/NĐ-CP{tuple_delimiter}SUA_DOI{tuple_delimiter}Nghị định 103/2024/NĐ-CP được sửa đổi, bổ sung bởi Nghị định 291/2025/NĐ-CP (câu bị động: Nghị định 291 là văn bản đi sửa){tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}Điều 3 Nghị định 50/2026/NĐ-CP{tuple_delimiter}diện tích đất tính tiền sử dụng đất{tuple_delimiter}AP_DUNG_CHO{tuple_delimiter}Khoản 1 quy định căn cứ diện tích đất{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}Điều 3 Nghị định 50/2026/NĐ-CP{tuple_delimiter}giá đất tính tiền sử dụng đất{tuple_delimiter}AP_DUNG_CHO{tuple_delimiter}Khoản 2 quy định căn cứ giá đất{tuple_delimiter}8)
{completion_delimiter}

######################
-Dữ liệu thật-
######################
Loại thực thể: [{entity_types}]
Văn bản:
{input_text}
######################
Kết quả:"""

CONTINUE_PROMPT = ("RẤT NHIỀU thực thể và quan hệ đã bị bỏ sót ở lần trích xuất trước. Chỉ ghi thêm những thực thể thuộc các "
                   "loại đã cho và quan hệ thuộc các loại quan hệ đã cho, theo đúng định dạng:\n")


def render_prompt(text: str) -> str:
    return EXTRACTION_PROMPT.format(entity_types=",".join(ENTITY_TYPES), relation_types=",".join(RELATION_TYPES),
                                    tuple_delimiter=TUPLE, record_delimiter=RECORD, completion_delimiter=COMPLETE,
                                    input_text=text)
