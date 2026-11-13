# Hybrid Retrieval và GraphRAG cho hỏi đáp pháp luật đất đai – nhà ở

Hệ thống hỏi đáp trên văn bản pháp luật đất đai và nhà ở đang còn hiệu lực, có cả văn bản của Đà Nẵng. Repo so sánh các chế độ truy hồi sau:

| Chế độ | Mô tả |
|---|---|
| `dense` | FAISS với embedding `AITeamVN/Vietnamese_Embedding` |
| `bm25` | BM25 có tách từ tiếng Việt |
| `hybrid-rrf` | Gộp dense và BM25 bằng Reciprocal Rank Fusion |
| `graph-augmented-hybrid` | Lấy kết quả hybrid rồi mở rộng qua đồ thị theo quan hệ sửa đổi và dẫn chiếu |
| `graph-global` | GraphRAG global search kiểu Microsoft: tóm tắt theo cộng đồng rồi map-reduce |

Mô hình sinh câu trả lời là `gpt-4o-mini`. Kết quả được đánh giá bằng chỉ số IR (Hit@k, MRR, Recall) và RAGAS.

## Cấu trúc

```
kb_pipeline/   Xây KB: metadata → trích văn bản → làm sạch → chunk
lc_app/        Truy hồi, sinh câu trả lời (LangChain), đánh giá IR/RAGAS
lc_app/graph/  GraphRAG: trích thực thể và quan hệ, dựng đồ thị, cộng đồng, Neo4j
streamlit/     Giao diện hỏi đáp
LEGAL_ONTOLOGY.md  Ontology của đồ thị (loại nút và loại quan hệ)
```

Repo chỉ chứa code. Dữ liệu (`data/`) và kết quả (`results*/`) không được đưa lên.

## Cài đặt

```powershell
python -m venv .venv-lc
.venv-lc/Scripts/python -m pip install -r requirements.txt
copy .env.example .env      # điền OPENAI_API_KEY và NEO4J_PASSWORD
docker compose up -d        # Neo4j, chỉ cần khi dùng lc_app.graph.load_neo4j
```

## Chạy

Mọi lệnh chạy từ thư mục gốc bằng `.venv-lc/Scripts/python -m ...`.

```powershell
# 1. Xây KB, chạy theo thứ tự
kb_pipeline.build_metadata ; kb_pipeline.extract ; kb_pipeline.clean ; kb_pipeline.chunk ; kb_pipeline.clean_chunks

# 2. Chỉ mục và đồ thị
lc_app.build_index
lc_app.graph.extract --all ; lc_app.graph.build ; lc_app.graph.community

# 3. Hỏi thử
lc_app.chain "Hạn mức giao đất ở tại Đà Nẵng là bao nhiêu?" --mode hybrid-rrf

# 4. Đánh giá
lc_app.evaluate_ir --mode dense --mode bm25 --mode hybrid-rrf --mode graph-augmented-hybrid
lc_app.evaluate_ragas
lc_app.evaluate_global
```

Giao diện: `cd streamlit` rồi chạy `..\.venv-lc\Scripts\streamlit run app.py`.

Các bước gọi OpenAI (`graph.extract`, `graph.community`, `evaluate_ragas`, `evaluate_global`) đều có cache SQLite, nên chạy lại chỉ tốn phí cho phần mới.
