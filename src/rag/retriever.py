from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import CHUNKS_STORE, VECTOR_STORE
from .embeddings import embed_texts
from .loader import DocumentChunk


@dataclass
class RetrievalResult:
    chunks: list[DocumentChunk]
    suggested_questions: list[str] = None  # 新增：推荐的标准问题
    error: str = None  # 新增：错误信息

    def __post_init__(self):
        if self.suggested_questions is None:
            self.suggested_questions = []


class KnowledgeRetriever:
    def __init__(self) -> None:
        self.store_path: Path = CHUNKS_STORE
        self.vector_store_path: Path = VECTOR_STORE
        self._standard_questions: list[str] = []  # 缓存标准问题

    def _extract_standard_questions(self) -> list[str]:
        """从FAQ文档中提取所有标准问题（Q1-Q20格式）"""
        if self._standard_questions:
            return self._standard_questions
        
        chunks = self._load_chunks()
        questions = []
        seen = set()  # 用于去重
        
        for chunk in chunks:
            # 匹配 "Q数字：问题内容" 的格式
            matches = re.findall(r'Q(\d+)\s*：\s*(.+?)(?=\s*A：|\n\n|$)', chunk.content)
            for _, question in matches:
                q = question.strip()
                if q and q not in seen:
                    questions.append(q)
                    seen.add(q)
        
        self._standard_questions = questions
        return questions

    def _suggest_related_questions(self, query: str, top_n: int = 3) -> list[str]:
        """根据用户查询推荐相关度高的标准问题"""
        standard_questions = self._extract_standard_questions()
        if not standard_questions:
            return []
        
        # 停用词列表：过滤常见无意义词汇
        stop_words = {'我', '你', '他', '她', '它', '的', '了', '是', '在', '有', '和', '就', 
                      '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', 
                      '去', '会', '着', '没有', '看', '好', '自己', '这', '那', '怎么办', '怎么', '什么', '吗'}
        
        query_terms = [t for t in self._extract_terms(query.lower()) if t not in stop_words]
        if not query_terms:
            return standard_questions[:top_n]
        
        # 计算每个标准问题与查询的相似度（基于关键词匹配）
        scored = []
        for question in standard_questions:
            question_lower = question.lower()
            score = 0
            for term in query_terms:
                if term in question_lower:
                    score += 1
                    # 位置权重：关键词出现在开头得分更高
                    if question_lower.startswith(term):
                        score += 0.5
            if score > 0:
                scored.append((score, question))
        
        # 按相似度排序
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # 如果有匹配结果，返回前top_n个
        if scored:
            return [item[1] for item in scored[:top_n]]
        
        # 如果没有直接匹配，尝试模糊匹配（检查是否有重叠的字符）
        fuzzy_scored = []
        query_chars = set(''.join(query_terms))
        for question in standard_questions:
            question_chars = set(question.lower())
            overlap = len(query_chars & question_chars)
            if overlap >= 2:  # 至少有2个字符重叠
                fuzzy_scored.append((overlap, question))
        
        if fuzzy_scored:
            fuzzy_scored.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in fuzzy_scored[:top_n]]
        
        # 如果还是没有匹配，返回热门问题
        return standard_questions[:top_n]

    @staticmethod
    def _extract_terms(text: str) -> list[str]:
        # 常见单字过滤（这些字出现太频繁，不适合作为关键词）
        common_chars = {'的', '了', '和', '是', '就', '都', '而', '及', '与', '着', '或', '有', '不', '在', '我', '你', '他', '她', '它', '这', '那', '此', '其', '某', '每', '各', '诸', '所', '以', '于', '为', '因', '由', '随', '对', '同', '向', '从', '到', '比', '被', '把', '将', '用', '使', '让', '叫', '令', '能', '会', '可以', '应该', '必须', '需要', '得', '要', '应', '该', '可', '须', '需', '之', '乎', '者', '也', '矣', '焉', '哉', '乎', '尔', '汝', '吾', '余', '予', '朕', '臣', '妾', '奴', '仆', '鄙人', '在下', '愚', '蒙', '窃', '伏', '谨', '敬', '幸', '敢', '请', '愿', '欲', '肯', '忍', '奈', '莫', '勿', '毋', '休', '别', '甭', '叵', '叵耐', '叵测', '叵罗', '叵信', '叵耐', '叵', '币', '见'}
        
        # 提取所有中文词和英文/数字
        terms = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", text)
        
        # 额外提取单字和双字词（提高匹配精度）
        all_terms = []
        for term in terms:
            term = term.lower().strip()
            if term:
                all_terms.append(term)
                # 对于中文词，拆分出单字和双字词
                if len(term) > 1:
                    for i in range(len(term)):
                        char = term[i]
                        # 过滤常见单字
                        if char not in common_chars:
                            all_terms.append(char)  # 单字
                        if i < len(term) - 1:
                            all_terms.append(term[i:i+2])  # 双字词
        
        return list(set(all_terms))  # 去重

    def _load_chunks(self) -> list[DocumentChunk]:
        if not self.store_path.exists():
            return []
        raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        chunks: list[DocumentChunk] = []
        for item in raw:
            chunks.append(
                DocumentChunk(
                    chunk_id=item.get("chunk_id", ""),
                    content=item.get("content", ""),
                    source=item.get("source", "unknown"),
                )
            )
        return chunks

    def _search_text(self, query: str, top_k: int = 4) -> RetrievalResult:
        chunks = self._load_chunks()
        if not chunks:
            return RetrievalResult(chunks=[])

        query_terms = self._extract_terms(query)
        if not query_terms:
            return RetrievalResult(chunks=chunks[:top_k])

        scored: list[tuple[int, DocumentChunk]] = []
        for chunk in chunks:
            text = chunk.content.lower()
            score = sum(text.count(term) for term in query_terms)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [item[1] for item in scored[:top_k]]
        
        # 如果匹配结果较少或相似度不高，返回推荐问题
        suggested_questions = []
        if len(selected) < top_k or not scored:
            suggested_questions = self._suggest_related_questions(query, top_n=3)
        elif scored and scored[0][0] <= 20:  # 如果最高分较低，也返回推荐问题
            suggested_questions = self._suggest_related_questions(query, top_n=3)
        
        if not selected:
            selected = chunks[:top_k]
        
        return RetrievalResult(chunks=selected, suggested_questions=suggested_questions)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _search_embedding(self, query: str, top_k: int = 4, similarity_threshold: float = 0.3) -> RetrievalResult:
        if not self.vector_store_path.exists():
            return RetrievalResult(chunks=[])

        raw = json.loads(self.vector_store_path.read_text(encoding="utf-8"))
        query_vec = embed_texts([query])[0]
        scored: list[tuple[float, DocumentChunk]] = []
        for item in raw:
            emb = item.get("embedding")
            if not emb:
                continue
            score = self._cosine(query_vec, emb)
            scored.append(
                (
                    score,
                    DocumentChunk(
                        chunk_id=item.get("chunk_id", ""),
                        content=item.get("content", ""),
                        source=item.get("source", "unknown"),
                    ),
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # 过滤相似度低于阈值的结果，并返回推荐问题
        selected = [x[1] for x in scored[:top_k] if x[0] >= similarity_threshold]
        suggested_questions = []
        
        # 如果匹配结果较少或相似度低，返回推荐问题
        if len(selected) < top_k or (scored and scored[0][0] < similarity_threshold):
            suggested_questions = self._suggest_related_questions(query, top_n=3)
        
        # 如果没有有效匹配，返回空列表（后续会降级到文本检索）
        return RetrievalResult(chunks=selected, suggested_questions=suggested_questions)

    def search(self, query: str, top_k: int = 4, mode: str = "text") -> RetrievalResult:
        if mode == "text":
            return self._search_text(query, top_k=top_k)
        if mode == "embedding":
            try:
                result = self._search_embedding(query, top_k=top_k)
                return result
            except Exception as e:
                # 返回包含错误信息的结果，不降级
                error_msg = self._analyze_embedding_error(str(e))
                return RetrievalResult(
                    chunks=[],
                    suggested_questions=[],
                    error=error_msg
                )
        raise ValueError("mode must be 'text' or 'embedding'")
    
    def _analyze_embedding_error(self, error_str: str) -> str:
        """分析 embedding 服务不可用的原因"""
        if "404" in error_str:
            return (
                "Embedding 服务不可用（404错误）。可能原因：\n"
                "1. Ollama 服务未启动，请运行 'ollama serve'\n"
                "2. 嵌入模型未拉取，请运行 'ollama pull bge-m3'\n"
                "3. API 端点配置错误，请检查 OLLAMA_BASE_URL"
            )
        elif "connection refused" in error_str.lower():
            return (
                "无法连接到 Ollama 服务。可能原因：\n"
                "1. Ollama 服务未启动，请运行 'ollama serve'\n"
                "2. 服务端口被占用或防火墙阻止\n"
                "3. OLLAMA_BASE_URL 配置不正确"
            )
        elif "timeout" in error_str.lower():
            return (
                "Embedding 请求超时。可能原因：\n"
                "1. Ollama 服务响应过慢\n"
                "2. 网络延迟过高\n"
                "3. 嵌入模型过大，推理时间长"
            )
        else:
            return f"Embedding 服务异常：{error_str}\n请检查 Ollama 服务状态和模型配置。"

