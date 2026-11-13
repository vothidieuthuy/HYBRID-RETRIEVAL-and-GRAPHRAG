// Truy vấn kiểm tra đồ thị (LEGAL_ONTOLOGY.md v2.0). Dán TỪNG câu vào Neo4j Browser (http://localhost:7474).
// Câu trả về bảng: xem ở tab Table. Câu trả về đường đi (p): xem ở tab Graph.
// Chú thích không dùng dấu chấm phẩy (Neo4j tách câu lệnh theo dấu chấm phẩy).

// ===== A. TỔNG QUAN =====
// A1. Sơ đồ schema: loại nút nào nối với loại nút nào
CALL db.schema.visualization();

// A2. Số nút theo loại
MATCH (n) UNWIND labels(n) AS loai WITH loai WHERE loai <> '__Entity__'
RETURN loai, count(*) AS so_nut ORDER BY so_nut DESC;

// A3. Số cạnh theo loại
MATCH ()-[r]->() RETURN type(r) AS quan_he, count(*) AS so_canh ORDER BY so_canh DESC;

// ===== B. XEM ĐỒ THỊ (tab Graph) =====
// B1. Bộ khung văn bản – Điều (giống sơ đồ mẫu, bỏ chunk và khái niệm)
MATCH p=()-[:CHUA|HUONG_DAN|THAM_CHIEU|SUA_DOI]->() RETURN p;

// B2. Quanh Luật Đất đai (văn bản chính)
MATCH p=(v:VAN_BAN {id: 'Luật Đất đai 31/2024/QH15'})-[r]-(x) WHERE type(r) <> 'MENTIONS' RETURN p;

// B3. Quanh Điều 220 (ai dẫn chiếu, áp dụng cho gì)
MATCH p=(d:DIEU_KHOAN {id: 'Điều 220 Luật Đất đai 31/2024/QH15'})-[r]-(x) WHERE type(r) <> 'MENTIONS' RETURN p;

// B4. Toàn bộ thực thể và quan hệ (không có chunk)
MATCH p=(:__Entity__)-[r]->(:__Entity__) RETURN p;

// ===== C. SOÁT CHI TIẾT (tab Table) =====
// C1. Tất cả quan hệ (trừ MENTIONS) kèm mô tả
MATCH (a)-[r]->(b) WHERE type(r) <> 'MENTIONS'
RETURN a.id AS nguon, type(r) AS quan_he, b.id AS dich, r.weight AS do_manh, r.description AS mo_ta
ORDER BY quan_he, nguon;

// C2. LLM trích được gì từ MỘT chunk (đổi id chunk để xem chunk khác)
MATCH (c:Chunk {id: 'VN.31-2024-QH15::D220::K3-4'})-[:MENTIONS]->(e)
RETURN [l IN labels(e) WHERE l <> '__Entity__'][0] AS loai, e.id AS ten, e.description AS mo_ta ORDER BY loai;

// C3. Đọc nguyên văn một chunk để đối chiếu với C2
MATCH (c:Chunk {id: 'VN.31-2024-QH15::D220::K3-4'}) RETURN c.header AS header, c.text AS noi_dung;

// C4. Danh sách văn bản và số chunk nhắc tới mỗi văn bản
MATCH (v:VAN_BAN) RETURN v.id AS van_ban, size(v.chunk_ids) AS so_chunk_nhac_toi, v.description AS mo_ta
ORDER BY so_chunk_nhac_toi DESC;

// C5. Quan hệ sửa đổi và hướng dẫn (kiểm tra CHIỀU mũi tên)
MATCH (a)-[r:SUA_DOI|HUONG_DAN]->(b) RETURN a.id AS nguon, type(r) AS quan_he, b.id AS dich, r.description AS mo_ta;

// ===== D. KIỂM TRA LỖI (kết quả RỖNG là đạt) =====
// D1. Sai miền/đích: CHUA phải VAN_BAN -> DIEU_KHOAN, AP_DUNG_CHO phải DIEU_KHOAN -> KHAI_NIEM/CO_QUAN
MATCH (a)-[r]->(b)
WHERE (type(r) = 'CHUA' AND NOT (a:VAN_BAN AND b:DIEU_KHOAN))
   OR (type(r) = 'AP_DUNG_CHO' AND NOT (a:DIEU_KHOAN AND (b:KHAI_NIEM OR b:CO_QUAN)))
   OR (type(r) = 'HUONG_DAN' AND NOT b:VAN_BAN)
RETURN a.id, type(r), b.id;

// D2. Thực thể không có chunk nguồn
MATCH (e:__Entity__) WHERE NOT ()-[:MENTIONS]->(e) RETURN e.id;

// D3. Thực thể đứng riêng, không có quan hệ ontology nào
MATCH (e:__Entity__) WHERE NOT (e)-[:CHUA|HUONG_DAN|THAM_CHIEU|SUA_DOI|AP_DUNG_CHO]-() RETURN e.id;

// D4. Hai nút chỉ khác chữ hoa/thường (trùng)
MATCH (a:__Entity__), (b:__Entity__) WHERE elementId(a) < elementId(b) AND toLower(a.id) = toLower(b.id)
RETURN a.id, b.id;

// D5. Tên Điều sai dạng (phải "Điều <số> <tên văn bản>")
MATCH (d:DIEU_KHOAN) WHERE NOT d.id =~ 'Điều \\d+[a-z]? .+' OR d.id =~ '(?i).*(khoản|điểm) \\w.*' RETURN d.id;

// ===== E. MÔ PHỎNG CHẾ ĐỘ hybrid-graph (M2) =====
// E1. Từ một chunk hạt giống (kết quả hybrid), đồ thị sẽ kéo thêm những chunk nào
//     Ví dụ Q04: hạt giống NQ 254 Điều 11 K3 -> kéo LĐĐ Điều 220 K1, K3-4 qua "cùng Điều 220"
MATCH (s:Chunk {id: 'VN.254-2025-QH15::D11::K3'})-[:MENTIONS]->(e:DIEU_KHOAN)
CALL (s, e) {
  MATCH (e)<-[:MENTIONS]-(c:Chunk) WHERE c.id <> s.id
  RETURN c.id AS chunk, 'CUNG_DIEU' AS qua
  UNION
  MATCH (e)-[r:SUA_DOI|THAM_CHIEU]-(n:DIEU_KHOAN)<-[:MENTIONS]-(c:Chunk) WHERE c.id <> s.id
  RETURN c.id AS chunk, type(r) AS qua
}
RETURN e.id AS dieu_trung_gian, qua, collect(DISTINCT chunk) AS chunk_mo_rong;

// E2. Cùng ví dụ, xem dạng đồ thị
MATCH p=(s:Chunk {id: 'VN.254-2025-QH15::D11::K3'})-[:MENTIONS]->(e:DIEU_KHOAN)<-[:MENTIONS]-(c:Chunk) RETURN p;

// ===== F. HIỂN THỊ TẤT CẢ =====
// F1. Mọi nút (thực thể và chunk) và mọi quan hệ, kể cả nút đứng riêng
MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m;

// F2. Chỉ thực thể và quan hệ ontology (bỏ Chunk và MENTIONS)
MATCH (n:__Entity__) OPTIONAL MATCH (n)-[r]->(m:__Entity__) RETURN n, r, m;

// ===== G. TRA CỨU THEO NHU CẦU (mọi câu vẽ hình ở tab Graph, không ghi gì vào database) =====
// Điều thuộc văn bản nào: suy từ tên ("Điều 220 Luật Đất đai 31/2024/QH15" kết thúc bằng tên văn bản), vì Điều chỉ
// được dẫn chiếu tới không có cạnh CHUA. Cạnh suy ra được vẽ bằng cạnh ảo của APOC (apoc.create.vRelationship, có
// thuộc tính suy_tu_ten = true): chỉ có trong kết quả, không lưu. Chạy dòng :param riêng trước câu truy vấn.

// G1. Muốn làm thủ tục này thì liên quan đến văn bản nào, luật nào, cơ quan nào
//     Điều quy định thủ tục -> thủ tục, cơ quan thực hiện, văn bản chứa Điều, Điều/văn bản được dẫn chiếu
:param thu_tuc => 'tách thửa';
MATCH p1 = (d:DIEU_KHOAN)-[:AP_DUNG_CHO]->(k:KHAI_NIEM) WHERE toLower(k.id) CONTAINS toLower($thu_tuc)
OPTIONAL MATCH p2 = (d)-[:AP_DUNG_CHO]->(:CO_QUAN)
OPTIONAL MATCH p3 = (d)-[:THAM_CHIEU|HUONG_DAN|SUA_DOI]->(x:__Entity__)
OPTIONAL MATCH (v:VAN_BAN) WHERE d.id ENDS WITH ' ' + v.id
OPTIONAL MATCH (xv:VAN_BAN) WHERE x.id ENDS WITH ' ' + xv.id
WITH collect(DISTINCT p1) + collect(DISTINCT p2) + collect(DISTINCT p3) AS duong,
     collect(DISTINCT [v, d]) + collect(DISTINCT [xv, x]) AS so_huu
UNWIND so_huu AS o
WITH duong, o[0] AS vb, o[1] AS dk
OPTIONAL CALL (vb, dk) {
  WITH vb, dk WHERE vb IS NOT NULL AND dk IS NOT NULL AND vb <> dk AND NOT (vb)-[:CHUA]->(dk)
  CALL apoc.create.vRelationship(vb, 'CHUA', {suy_tu_ten: true}, dk) YIELD rel
  RETURN rel
}
RETURN duong, collect(DISTINCT vb) AS van_ban, collect(rel) AS chua_suy_ra;

// G2. Các văn bản Đà Nẵng có trong đồ thị (văn bản của UBND, HĐND thành phố)
MATCH (v:VAN_BAN) WHERE v.id CONTAINS 'UBND' OR v.id CONTAINS 'HĐND' RETURN v;

// G3. Một văn bản Đà Nẵng liên quan đến văn bản nào: mỗi văn bản liên quan một nút, cạnh ảo theo loại quan hệ
//     (so_canh = số cặp Điều tạo ra liên kết), chiều mũi tên như quan hệ gốc
:param vb => 'Quyết định 135/2026/QĐ-UBND';
MATCH (v:VAN_BAN {id: $vb})
MATCH (a:__Entity__) WHERE a = v OR a.id ENDS WITH ' ' + v.id
MATCH (a)-[r:THAM_CHIEU|HUONG_DAN|SUA_DOI]-(b:__Entity__)
MATCH (w:VAN_BAN) WHERE (w = b OR b.id ENDS WITH ' ' + w.id) AND w <> v
WITH v, w, type(r) AS loai, startNode(r) = a AS di, count(*) AS n
CALL apoc.create.vRelationship(CASE WHEN di THEN v ELSE w END, loai, {so_canh: n, suy_tu_ten: true},
                               CASE WHEN di THEN w ELSE v END) YIELD rel
RETURN v, w, rel;

// G4. Cùng câu hỏi, chi tiết tới Điều: văn bản -> các Điều của nó -> Điều/văn bản được dẫn chiếu -> văn bản chứa
MATCH (v:VAN_BAN {id: $vb})
MATCH (a:__Entity__) WHERE a = v OR a.id ENDS WITH ' ' + v.id
MATCH p = (a)-[:THAM_CHIEU|HUONG_DAN|SUA_DOI]-(b:__Entity__) WHERE b <> v AND NOT b.id ENDS WITH ' ' + v.id
OPTIONAL MATCH c = (v)-[:CHUA]->(a)
OPTIONAL MATCH (w:VAN_BAN) WHERE b.id ENDS WITH ' ' + w.id
OPTIONAL CALL (w, b) {
  WITH w, b WHERE w IS NOT NULL AND NOT (w)-[:CHUA]->(b)
  CALL apoc.create.vRelationship(w, 'CHUA', {suy_tu_ten: true}, b) YIELD rel
  RETURN rel
}
RETURN p, c, w, rel;

// G5. Phân cấp tới một Luật: Luật <- Nghị định <- Thông tư <- văn bản Đà Nẵng. Gộp HUONG_DAN, THAM_CHIEU của văn bản
//     và các Điều của nó thành cạnh ảo văn bản -> văn bản (LIEN_KET_VB {loai, so_canh}), chỉ giữ cạnh từ cấp dưới lên
//     cấp trên (cap: 1 Luật, Bộ luật, Nghị quyết Quốc hội - 2 Nghị định, Nghị quyết Chính phủ - 3 Thông tư, Quyết
//     định Thủ tướng - 4 văn bản UBND, HĐND - 5 khác), rồi lấy mọi văn bản nối được về Luật trong tối đa 3 bước.
//     SUA_DOI không tính (sửa đổi không phải quan hệ cấp trên - cấp dưới)
:param luat => 'Luật Đất đai 31/2024/QH15';
MATCH (a:__Entity__)-[r:HUONG_DAN|THAM_CHIEU]->(b:__Entity__)
MATCH (A:VAN_BAN) WHERE A = a OR a.id ENDS WITH ' ' + A.id
MATCH (B:VAN_BAN) WHERE B = b OR b.id ENDS WITH ' ' + B.id
WITH A, B, collect(DISTINCT type(r)) AS loai, count(*) AS n WHERE A <> B
WITH A, B, loai, n, [x IN [A, B] | CASE
  WHEN x.id STARTS WITH 'Luật' OR x.id STARTS WITH 'Bộ luật' OR x.id STARTS WITH 'Hiến pháp'
       OR (x.id STARTS WITH 'Nghị quyết' AND x.id CONTAINS '/QH') THEN 1
  WHEN x.id STARTS WITH 'Nghị định' OR (x.id STARTS WITH 'Nghị quyết' AND x.id CONTAINS 'NQ-CP') THEN 2
  WHEN x.id STARTS WITH 'Thông tư' OR x.id CONTAINS 'TTg' THEN 3
  WHEN x.id CONTAINS 'UBND' OR x.id CONTAINS 'HĐND' THEN 4
  ELSE 5 END] AS cap
WHERE cap[0] > cap[1]
WITH collect({a: A, b: B, loai: loai, n: n, cap_a: cap[0]}) AS E
MATCH (L:VAN_BAN {id: $luat})
WITH E, [e IN E WHERE e.b = L] AS h1
WITH E, h1, [e IN E WHERE e.b IN [x IN h1 | x.a]] AS h2
WITH h1 + h2 + [e IN E WHERE e.b IN [x IN h2 | x.a]] AS giu
UNWIND giu AS e
WITH DISTINCT e
CALL apoc.create.vRelationship(e.a, 'LIEN_KET_VB', {loai: e.loai, so_canh: e.n, suy_tu_ten: true}, e.b) YIELD rel
RETURN e.a, rel, e.b;

// G6. Như G5 nhưng chỉ giữ các nhánh xuất phát từ văn bản Đà Nẵng: Đà Nẵng -> (Thông tư) -> (Nghị định) -> Luật
MATCH (a:__Entity__)-[r:HUONG_DAN|THAM_CHIEU]->(b:__Entity__)
MATCH (A:VAN_BAN) WHERE A = a OR a.id ENDS WITH ' ' + A.id
MATCH (B:VAN_BAN) WHERE B = b OR b.id ENDS WITH ' ' + B.id
WITH A, B, collect(DISTINCT type(r)) AS loai, count(*) AS n WHERE A <> B
WITH A, B, loai, n, [x IN [A, B] | CASE
  WHEN x.id STARTS WITH 'Luật' OR x.id STARTS WITH 'Bộ luật' OR x.id STARTS WITH 'Hiến pháp'
       OR (x.id STARTS WITH 'Nghị quyết' AND x.id CONTAINS '/QH') THEN 1
  WHEN x.id STARTS WITH 'Nghị định' OR (x.id STARTS WITH 'Nghị quyết' AND x.id CONTAINS 'NQ-CP') THEN 2
  WHEN x.id STARTS WITH 'Thông tư' OR x.id CONTAINS 'TTg' THEN 3
  WHEN x.id CONTAINS 'UBND' OR x.id CONTAINS 'HĐND' THEN 4
  ELSE 5 END] AS cap
WHERE cap[0] > cap[1]
WITH collect({a: A, b: B, loai: loai, n: n, cap_a: cap[0]}) AS E
MATCH (L:VAN_BAN {id: $luat})
WITH E, [e IN E WHERE e.b = L] AS h1
WITH E, h1, [e IN E WHERE e.b IN [x IN h1 | x.a]] AS h2
WITH h1 + h2 + [e IN E WHERE e.b IN [x IN h2 | x.a]] AS giu
WITH giu, [e IN giu WHERE e.cap_a = 4] AS k3
WITH giu, k3, [e IN giu WHERE e.a IN [x IN k3 | x.b]] AS k2
WITH k3 + k2 + [e IN giu WHERE e.a IN [x IN k2 | x.b]] AS nhanh
UNWIND nhanh AS e
WITH DISTINCT e
CALL apoc.create.vRelationship(e.a, 'LIEN_KET_VB', {loai: e.loai, so_canh: e.n, suy_tu_ten: true}, e.b) YIELD rel
RETURN e.a, rel, e.b;

