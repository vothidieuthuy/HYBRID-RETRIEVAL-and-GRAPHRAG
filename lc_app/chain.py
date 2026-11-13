"""Chuỗi hỏi đáp (LCEL): retriever → cắt theo ngân sách ký tự → prompt → ChatOpenAI → câu trả lời + nguồn.

    .venv-lc/Scripts/python -m lc_app.chain "Hạn mức giao đất ở tại Đà Nẵng là bao nhiêu?" [--mode hybrid-rrf]
"""
from __future__ import annotations

import argparse
import json

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from .config import Config
from .loader import citation
from .retrievers import MODES, Retrievers

# Prompt sinh câu trả lời (bản thứ ba, chốt ngày 05/11/2026). Hai bản trước, lưu ở _archive/2026-11-05:
# bản 1 để LLM từ chối dù NGỮ CẢNH có đáp án và trả lời rào đón; bản 2 làm LLM mở đầu "Có," cả với câu hỏi không
# phải dạng có/không, có khi kết luận ngược nội dung.
INSTRUCTIONS = """Bạn là trợ lý hỏi đáp pháp luật đất đai bằng tiếng Việt.
Chỉ sử dụng thông tin trong NGỮ CẢNH. Phân biệt rõ trích dẫn nguồn và diễn giải.
Câu đầu tiên trả lời thẳng vào câu hỏi:
- Chỉ khi câu hỏi là dạng có/không (có được ... không, có phải ... không, có bắt buộc ... không, ... hay không) thì
  mở đầu bằng kết luận "Có", "Không", "Được", "Không được", "Phải" hoặc "Không phải", sau đó mới nêu căn cứ; kết luận
  phải khớp với nội dung trả lời phía sau.
- Câu hỏi khác (là gì, gồm những gì, điều kiện gì, như thế nào, bao nhiêu, khi nào, ai, ở đâu) thì KHÔNG mở đầu bằng
  "Có" hay "Không" mà nêu thẳng nội dung trả lời.
Nếu câu hỏi có nhiều ý hoặc nêu nhiều trường hợp (ví dụ chuyển nhượng, tặng cho, cho thuê), trả lời đủ từng ý,
từng trường hợp.
Nếu kết luận phụ thuộc điều kiện thì nêu cụ thể các điều kiện đó theo NGỮ CẢNH; không viết chung chung như
"phụ thuộc vào một số điều kiện" hoặc "có thể cần".
Trước khi kết luận chưa đủ thông tin, đọc lại toàn bộ NGỮ CẢNH: quy định cần tìm có thể nằm ở bất kỳ mục nào.
Ý nào có căn cứ trong NGỮ CẢNH thì trả lời ý đó; ý nào không đủ căn cứ thì nói rõ ý đó "Chưa đủ thông tin trong
nguồn truy xuất". Chỉ trả lời "Chưa đủ thông tin trong nguồn truy xuất" cho cả câu khi không ý nào có căn cứ.
Không thêm thông tin nằm ngoài phạm vi câu hỏi.
Không tự tạo điều luật và không khẳng định văn bản còn hiệu lực nếu nguồn không xác nhận.
Mỗi nhận định pháp lý phải gắn [số] trỏ đến đúng mục trong NGỮ CẢNH.
Chỉ dùng dạng [1], [2], ...; không viết "[số]" trong câu trả lời.
Không được suy đoán, bổ sung quy trình thường gặp, hoặc đưa khuyến nghị không có trong NGỮ CẢNH.
Không tự tạo Điều/Khoản cho công văn, bài viết hoặc tài liệu không có Điều/Khoản trong metadata.
Nếu câu hỏi hỏi một văn bản/số hiệu cụ thể mà NGỮ CẢNH không chứa văn bản đó,
hãy trả lời ngắn gọn rằng chưa đủ thông tin và không liệt kê nội dung thay thế.
Cuối câu trả lời, liệt kê các căn cứ đã sử dụng với số hiệu văn bản và Điều/Khoản nếu có."""

# một tin nhắn người dùng gồm hướng dẫn + ngữ cảnh + câu hỏi (giống cách đã đánh giá)
PROMPT = ChatPromptTemplate.from_messages([("human", INSTRUCTIONS + "\n\nNGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}")])
NO_CONTEXT = "Chưa đủ thông tin trong nguồn truy xuất."


def cut_to_budget(documents: list[Document], depth: int, budget: int) -> list[Document]:
    """Giữ theo thứ tự cho tới khi tổng độ dài nội dung đạt ngân sách (chunk cuối có thể vượt nhẹ)."""
    kept, used = [], 0
    for doc in documents[:depth]:
        if used >= budget:
            break
        kept.append(doc)
        used += len(doc.metadata["text"])
    return kept


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {citation(d)}\n" + (f"{d.metadata['notes']}\n" if d.metadata.get("notes") else "")
                       + d.metadata["text"] for i, d in enumerate(documents, 1))


def build_chain(config: Config, retrievers: Retrievers, mode: str | None = None) -> Runnable:
    """Đầu vào: câu hỏi (str). Đầu ra: {"question", "documents", "answer"}."""
    retrieve = retrievers.get(mode or config.retrieval_mode) | RunnableLambda(
        lambda docs: cut_to_budget(docs, config.depth, config.context_chars))
    generate = (RunnableLambda(lambda x: {"context": format_context(x["documents"]), "question": x["question"]})
                | PROMPT | ChatOpenAI(model=config.llm_model, temperature=0) | StrOutputParser())
    answer = RunnableLambda(lambda x: generate.invoke(x) if x["documents"] else NO_CONTEXT)
    return {"question": RunnablePassthrough(), "documents": retrieve} | RunnablePassthrough.assign(answer=answer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES)
    args = parser.parse_args()
    config = Config.from_env()
    result = build_chain(config, Retrievers(config), args.mode).invoke(args.question)
    print(result["answer"])
    print(json.dumps([{"chunk_id": d.id, "citation": citation(d)} for d in result["documents"]],
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
