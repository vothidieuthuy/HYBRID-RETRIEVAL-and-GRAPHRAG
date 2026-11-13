"""Tách từ tiếng Việt cho BM25Retriever (truyền qua preprocess_func).

Token = âm tiết + cặp âm tiết liền kề trong cùng mệnh đề (xấp xỉ từ ghép) + token đặc biệt cho số hiệu văn bản
("#vb:31/2024/qh15") và số Điều ("#dieu:220"), để câu hỏi nêu "Điều 220 Luật 31/2024/QH15" khớp đúng chunk.
"""
from __future__ import annotations

import re
import unicodedata

DOC_NUMBER = re.compile(
    r"\b(\d{1,5})\s*/\s*(\d{4})\s*/\s*((?:qh|nđ-cp|nd-cp|nq-cp|tt-|ttlt-|nq-hđnd|nq-hdnd|qđ-ubnd|qd-ubnd|qđ-ttg)"
    r"[\wđ-]*)", re.IGNORECASE)
ARTICLE = re.compile(r"\bđiều\s+(\d+[a-z]?)\b", re.IGNORECASE)
CLAUSE_BREAK = re.compile(r"[.,;:!?()\[\]{}\"“”'‘’>|\n]+")
WORD = re.compile(r"\w+", re.UNICODE)
# chỉ bỏ từ để hỏi/xưng hô; các từ như "được", "trường hợp" mang nghĩa pháp lý nên giữ lại
STOPWORDS = frozenset("tôi mình xin hỏi ạ gì nào sao thế bao nhiêu".split())


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("Ð", "Đ").replace("ð", "đ").lower()


def legal_tokens(text: str) -> list[str]:
    numbers = {f"{int(n)}/{y}/{s.replace('nd-', 'nđ-').replace('qd-', 'qđ-').replace('hdnd', 'hđnd')}"
               for n, y, s in DOC_NUMBER.findall(text)}
    return [f"#vb:{n}" for n in numbers] + [f"#dieu:{a}" for a in set(ARTICLE.findall(text))]


def tokenize(text: str) -> list[str]:
    text = normalize(text)
    tokens = legal_tokens(text)
    for clause in CLAUSE_BREAK.split(text):
        words = [w for w in WORD.findall(clause) if w not in STOPWORDS]
        tokens += words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return tokens
