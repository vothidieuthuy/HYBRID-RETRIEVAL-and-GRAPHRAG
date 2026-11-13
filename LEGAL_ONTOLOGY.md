# Ontology đồ thị pháp luật đất đai – nhà ở – kinh doanh bất động sản (v2.0 tối giản, 01/11/2026)

Đặc tả cho bước trích xuất đồ thị của GraphRAG (pipeline Microsoft): 4 loại nút, 5 loại quan hệ, 4 quy tắc đặt tên.
Bản trước: `_archive/2026-11-01/LEGAL_ONTOLOGY_v1.1.md` (6 nút, 6 quan hệ), `LEGAL_ONTOLOGY_v0.1.md` (3 lớp, có quy tắc).

Đồ thị đích có dạng sơ đồ mẫu:

```
   Nghị định 102/2024/NĐ-CP ──HUONG_DAN──►  Luật Đất đai 31/2024/QH15  ──THAM_CHIEU──► Bộ luật Dân sự
   Nghị định 151/2025/NĐ-CP ──SUA_DOI────►          │
                                                   CHUA
                                                    ▼
                         Điều 220 Luật Đất đai 31/2024/QH15 (Tách thửa đất, hợp thửa đất)
                          │ AP_DUNG_CHO            │ AP_DUNG_CHO              │ AP_DUNG_CHO
                          ▼                        ▼                          ▼
                 KHAI_NIEM: tách thửa đất   KHAI_NIEM: diện tích       CO_QUAN: Ủy ban nhân dân cấp tỉnh
                                            tối thiểu tách thửa
```

## 1. Nguyên tắc

1. **Chỉ LLM, chỉ từ văn bản chunk.** Không dùng regex, metadata hay `relations/` để tạo nút hoặc quan hệ.
2. **Text unit** = chunk có `graph_unit = true` (4.306 chunk) trong `data/chunk/chunk_v1/clean_chunk_v1/chunks.jsonl`,
   bộ chunk dùng chung với naive RAG (chung `chunk_id`). Nội dung đưa cho LLM = `header` + `text`; header cho biết đoạn
   văn thuộc văn bản và Điều nào. Không đưa `notes` (ghi chú sửa đổi do pipeline suy ra từ văn bản khác).
3. **Không có trong đoạn văn thì không có cạnh;** không dùng hiểu biết bên ngoài (không tự thêm số hiệu).
4. **Mọi nút và cạnh giữ `chunk_id` nguồn** (như `text_unit_ids` của Microsoft).

## 2. Loại nút (4)

| Loại | Gồm | Ví dụ | Không tạo nút |
|---|---|---|---|
| `VAN_BAN` | văn bản được nêu đích danh (có hoặc không có trong KB) | Luật Đất đai 31/2024/QH15; Nghị định 102/2024/NĐ-CP; Bộ luật Dân sự | "pháp luật về đất đai", "các văn bản có liên quan" |
| `DIEU_KHOAN` | một Điều cụ thể (khoản, điểm ghi trong mô tả) | Điều 220 Luật Đất đai 31/2024/QH15 | "Điều này" khi không xác định được văn bản |
| `CO_QUAN` | cơ quan, tổ chức có thẩm quyền hoặc thực hiện thủ tục | Ủy ban nhân dân cấp tỉnh; Văn phòng đăng ký đất đai | "Nhà nước", "cơ quan có thẩm quyền", "cơ quan nhà nước" |
| `KHAI_NIEM` | mọi đối tượng pháp lý khác mà Điều quy định: thủ tục, chủ thể, loại đất, tài sản, giấy tờ, quyền, nghĩa vụ, chế độ, hạn mức, tiền | tách thửa đất; người sử dụng đất; đất ở; Giấy chứng nhận quyền sử dụng đất; bồi thường về đất; hạn mức giao đất ở | "tổ chức, cá nhân", "trường hợp khác" (chung chung) |

## 3. Loại quan hệ (5)

| Quan hệ | Từ → Đến | Ý nghĩa | Dấu hiệu trong đoạn văn |
|---|---|---|---|
| `CHUA` | `VAN_BAN` → `DIEU_KHOAN` | văn bản có Điều này | header "... > Điều 220"; đoạn văn mở đầu "Điều 220. ..." |
| `HUONG_DAN` | `VAN_BAN`/`DIEU_KHOAN` → `VAN_BAN` | quy định chi tiết, hướng dẫn thi hành | "quy định chi tiết", "hướng dẫn thi hành"; các Điều được hướng dẫn ghi trong mô tả |
| `THAM_CHIEU` | `VAN_BAN`/`DIEU_KHOAN` → `VAN_BAN`/`DIEU_KHOAN` | dẫn chiếu tới quy định khác | "theo quy định tại", "quy định tại" (chỉ khi nêu được văn bản hoặc Điều) |
| `SUA_DOI` | `VAN_BAN`/`DIEU_KHOAN` → `VAN_BAN`/`DIEU_KHOAN` | sửa đổi, bổ sung, bãi bỏ hoặc thay thế | "sửa đổi, bổ sung", "bãi bỏ", "thay thế"; mô tả ghi rõ hình thức |
| `AP_DUNG_CHO` | `DIEU_KHOAN` → `KHAI_NIEM`/`CO_QUAN` | Điều quy định về đối tượng đó | nội dung chính của Điều |

- Hướng cạnh có nghĩa: Nghị định –`HUONG_DAN`→ Luật đọc là "Nghị định hướng dẫn Luật". Chiều ngược chỉ là cách đọc.
- "Theo quy định tại Điều 45 Luật Đất đai" trong thân văn bản là `THAM_CHIEU`, không phải `HUONG_DAN`. Đoạn văn nêu số
  Điều thì `THAM_CHIEU` trỏ tới Điều đó, không trỏ tới văn bản.
- `SUA_DOI`: nguồn là văn bản đi sửa. Câu bị động "X (được sửa đổi, bổ sung bởi/tại Y, Z)" cho Y → X và Z → X.
- Một Điều chỉ được nhắc tới không nối với văn bản của nó (tên Điều đã chứa tên văn bản); `CHUA` chỉ đọc từ header.

## 4. Quy tắc đặt tên

Microsoft gộp nút theo tên, nên một đối tượng phải luôn có cùng một tên.

1. **Văn bản:** Luật/Bộ luật = `<Loại> <tên gọi> <số hiệu>` ("Luật Đất đai 31/2024/QH15"); loại khác = `<Loại> <số hiệu>`
   ("Nghị định 102/2024/NĐ-CP"), không kèm trích yếu. Đoạn văn không có số hiệu thì ghi như đoạn văn viết ("Bộ luật Dân
   sự"). "Luật này", "Nghị định này" thay bằng tên trong header.
2. **Điều:** `Điều <số> <tên văn bản>`; Điều thuộc phần "Quy định kèm theo" ghi `Điều <số> Quy định kèm theo <tên văn
   bản>`. "Điều này" thay bằng Điều trong header.
3. **Khái niệm, cơ quan:** viết đầy đủ, không viết tắt ("Ủy ban nhân dân cấp tỉnh", không "UBND tỉnh"; "Giấy chứng nhận
   quyền sử dụng đất", không "sổ đỏ").
4. **Không tạo nút** cho cụm chung chung (cột cuối bảng mục 2).

## 5. Định dạng trích xuất

Tuple của Microsoft, thêm một trường loại quan hệ; gleaning 1 vòng; 3 ví dụ few-shot viết tay từ chunk của KB nằm ngoài
bộ demo (Điều 195 Luật Đất đai; Điều 1 NĐ 102/2024 liệt kê Điều được hướng dẫn; Điều 3 NĐ 50/2026 có dẫn chiếu và câu bị
động "được sửa đổi, bổ sung bởi"). Text unit = header đầy đủ (tên văn bản, Chương, Điều, tên phụ lục không bị cắt) + text.

```
("entity"<|>TÊN<|>LOẠI<|>MÔ TẢ)##
("relationship"<|>NGUỒN<|>ĐÍCH<|>LOẠI_QUAN_HỆ<|>MÔ TẢ<|>ĐIỂM)##
<|COMPLETE|>
```

`MÔ TẢ` chỉ lấy từ đoạn văn; với cạnh ghi cả khoản, điểm liên quan. `ĐIỂM` 1–10 (độ mạnh).

## 6. Sau trích xuất

1. Chuẩn hóa tên (chỉ đổi tên, không tạo quan hệ): bỏ chữ "số" sau loại văn bản; tên Điều bỏ phần khoản/điểm; tên
   văn bản không số hiệu gộp vào tên có số hiệu khi trong các tên đã trích chỉ có **đúng một** ứng viên cùng tên gọi
   ("Luật Đất đai" → "Luật Đất đai 31/2024/QH15"), áp dụng cả cho tên Điều. Trong đoạn văn có header "> Quy định kèm
   theo >", Điều của chính văn bản đó quy về `Điều <số> Quy định kèm theo <văn bản>` (gpt-4o-mini hay bỏ cụm này).
   Gộp nút theo tên (như Microsoft); loại của nút = loại được trích nhiều nhất. Gộp cạnh theo (nguồn, đích, loại quan
   hệ), giữ hướng. Mô tả được nối hoặc tóm tắt bằng LLM như Microsoft.
   Văn bản ghi kèm năm ("Luật Đất đai năm 2024") gộp vào văn bản có số hiệu cùng năm; có nhiều ứng viên (Luật Đất đai
   2024 và 2013 được nhắc tới) thì chọn văn bản của KB (nguồn cạnh `CHUA`). Dấu thanh về kiểu mới ("Uỷ" → "Ủy").
2. Đối chiếu từng lần nhắc với đúng đoạn văn LLM đã đọc (header đầy đủ + text): số Điều không có trong đoạn văn → bỏ
   lần nhắc đó và các cạnh của nó trong chunk đó (vd "khoản 3 Điều này" bị ghi "Điều 3"); số hiệu văn bản không có
   trong đoạn văn → bỏ số hiệu khỏi tên (LLM tự thêm "31/2024/QH15" thì gộp lại qua bí danh, số hiệu bịa thì tên còn lại
   không nêu đích danh và bị lọc).
3. Bỏ nút có loại ngoài mục 2, cạnh có loại ngoài mục 3, cạnh sai miền/đích của mục 3 (vd `THAM_CHIEU` Điều → khái
   niệm), nút chung chung, `DIEU_KHOAN` tên không phải `Điều <số> <văn bản>`/`Phụ lục <số> <văn bản>` (Chương, Mục, "Cụm
   từ ... tại khoản 1 Điều 11"), `VAN_BAN` không nêu đích danh ("nghị định của Chính phủ"); đếm số bị bỏ để báo cáo.
4. Cạnh được sửa cho khớp câu mô tả chính LLM viết cho nó (đếm riêng; không đọc lại đoạn văn, không tạo cạnh mới):
   `SUA_DOI` có mô tả "<nguồn> được sửa đổi ... bởi <đích>" hoặc "<đích> sửa đổi, bổ sung ..." thì đổi chiều;
   `THAM_CHIEU` có mô tả "<nguồn> sửa đổi, bổ sung ..." thì thành `SUA_DOI`.
5. Leiden phân cấp và báo cáo cộng đồng như Microsoft.
6. Nạp Neo4j (chạy bằng `docker-compose.yml`, có APOC) bằng `Neo4jGraph.query` của LangChain (Cypher, để giữ
   `chunk_ids`), cùng schema mà `add_graph_documents(include_source=True, baseEntityLabel=True)` tạo ra:
   ```
   (:Chunk {id: chunk_id})-[:MENTIONS]->(:__Entity__:<LOẠI> {id: tên, description})
   (:__Entity__)-[:<LOẠI_QUAN_HỆ> {description, weight}]->(:__Entity__)
   (:__Entity__)-[:IN_COMMUNITY]->(:__Community__ {level, title, report})
   ```

## 7. Hiển thị

Trang HTML vẽ đồ thị quanh văn bản được chọn, như sơ đồ mẫu: văn bản chính ở giữa (màu đậm), văn bản liên quan (màu
nhạt), Điều, khái niệm và cơ quan mỗi loại một màu, có chú thích; nhãn cạnh tiếng Việt ("hướng dẫn", "tham chiếu",
"chứa", "sửa đổi", "áp dụng cho"). Cạnh đi từ một Điều của văn bản chính tới văn bản khác được vẽ gộp thành cạnh văn
bản → văn bản. Mỗi Điều hiện tối đa 5 cạnh `AP_DUNG_CHO` có điểm cao nhất.

## 8. Khác biệt so với Microsoft GraphRAG gốc

1. Loại nút cố định theo mục 2 (Microsoft có sẵn tham số `entity_types`); không dùng `prompt-tune`, few-shot viết tay.
2. Tuple quan hệ thêm trường `LOẠI_QUAN_HỆ`; cạnh có hướng, gộp theo (nguồn, đích, loại). Bản gốc chỉ có mô tả và điểm,
   cạnh vô hướng.
3. Chuẩn hóa tên trước khi gộp (mục 6.1). Bản gốc gộp theo tên nguyên văn; chạy thử cho thấy cùng một văn bản bị tách
   thành nhiều nút ("Luật Đất đai" / "Luật Đất đai 31/2024/QH15", "Nghị quyết số 254/2025/QH15" / "Nghị quyết
   254/2025/QH15") nên đồ thị không có nút trung tâm như sơ đồ mẫu.

Các bước còn lại (gleaning, gộp nút theo tên, tóm tắt mô tả, Leiden, báo cáo cộng đồng, local/global search) giữ nguyên.

## 9. Giới hạn

- Quan hệ chỉ có khi đoạn văn nói ra; văn bản không ghi "quy định chi tiết" thì thiếu cạnh `HUONG_DAN`.
- Tên văn bản không số hiệu chỉ được gộp khi có đúng một ứng viên; nếu có hai (vd đồ thị có cả Luật Đất đai 2024 và
  2013) thì giữ nút riêng. Các bí danh đã gộp ghi ở `graph.json` (`aliases`).
- Không có: nút cấp Khoản (nút `Chunk` đã ở cấp khoản), bản án (KB không có), căn cứ ban hành (lời mở đầu không thuộc text
  unit), ngày hiệu lực dưới dạng cạnh.

## 10. Kiểm tra

- Chạy thử 4–5 chunk (Điều 1 NĐ 102/2024, Điều 220 Luật Đất đai, một khoản dẫn chiếu Bộ luật Dân sự, Điều 3 Quy định kèm
  theo QĐ 75/2026), so đồ thị với sơ đồ mẫu.
- Kiểm tra thủ công khoảng 50 chunk: tỷ lệ nút đúng loại, tỷ lệ cạnh có căn cứ, số văn bản bị tách nhiều nút.
- Đánh giá trong hệ thống: hybrid + mở rộng đồ thị trên 50 câu `ragas_testset_50`; global search trên
  `global_questions_24`.
