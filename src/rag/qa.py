from __future__ import annotations

import json
import os

from dotenv import load_dotenv
import requests

from .loader import format_context
from .retriever import KnowledgeRetriever


SYSTEM_PROMPT = """你是一名游戏客服机器人。
请严格根据提供的知识库片段回答：
1) 优先给出可执行步骤。
2) 如果知识不足，明确说“当前知识库暂无明确信息，建议转人工客服”。
3) 不要编造活动规则、补偿政策或处罚结论。
"""


def _build_user_prompt(question: str, context: str, suggested_questions: list[str] = None) -> str:
    prompt = (
        f"用户问题：{question}\n\n"
        f"知识库片段：\n{context}\n\n"
    )
    
    if suggested_questions:
        prompt += "相关问题推荐：\n"
        for i, q in enumerate(suggested_questions, 1):
            prompt += f"{i}. {q}\n"
        prompt += "\n"
    
    prompt += "请给出中文客服回复。\n"
    prompt += "如果知识库信息不足或匹配度不高，请参考相关问题推荐，引导用户选择更合适的问题。"
    
    return prompt


def _ask_with_ollama(model: str, base_url: str, user_prompt: str) -> tuple[str, dict]:
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = requests.post(endpoint, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    message = data.get("message", {})
    return str(message.get("content", "")).strip(), payload


def _ask_with_ollama_stream(model: str, base_url: str, user_prompt: str):
    """流式调用 Ollama API"""
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    
    response = requests.post(endpoint, json=payload, stream=True, timeout=120)
    response.raise_for_status()
    
    for line in response.iter_lines():
        if line:
            try:
                data = line.decode('utf-8').strip()
                # Ollama的响应没有data:前缀，直接解析JSON
                json_data = json.loads(data)
                message = json_data.get('message', {})
                content = message.get('content', '')
                if content:
                    yield {'type': 'content', 'data': content}
            except Exception:
                continue


def _ask_with_openai(model: str, api_key: str, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.output_text.strip()


def ask_with_rag(question: str, top_k: int = 4, retrieval_mode: str = "text") -> dict:
    retriever = KnowledgeRetriever()
    retrieval = retriever.search(question, top_k=top_k, mode=retrieval_mode)
    
    # 检查是否有 embedding 错误
    if retrieval.error:
        return {
            "answer": f"检索失败：{retrieval.error}",
            "context": "",
            "retrieval_mode": retrieval_mode,
            "suggested_questions": [],
            "error": retrieval.error
        }
    
    context = format_context(retrieval.chunks)
    suggested_questions = retrieval.suggested_questions if retrieval.suggested_questions else []

    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    user_prompt = _build_user_prompt(question, context, suggested_questions)

    # 存储实际发送给 API 的请求内容
    api_payload = None
    
    try:
        if provider == "ollama":
            ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct").strip()
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
            answer, api_payload = _ask_with_ollama(ollama_model, ollama_base_url, user_prompt)
            if not answer:
                answer = "Ollama 返回为空，建议检查模型是否已正确拉取并运行。"
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
            if not api_key:
                return {
                    "answer": "LLM_PROVIDER=openai 但未检测到 OPENAI_API_KEY，当前返回检索结果供人工参考。",
                    "context": context,
                }
            answer = _ask_with_openai(model, api_key, user_prompt)
        else:
            answer = (
                f"不支持的 LLM_PROVIDER={provider}，请使用 ollama 或 openai。"
            )
    except Exception as exc:
        answer = (
            f"LLM 调用失败（{exc}）。当前返回检索结果供人工调试，请检查模型服务与配置。"
        )
    
    return {
            "answer": answer, 
            "context": context, 
            "retrieval_mode": retrieval_mode,
            "suggested_questions": suggested_questions,
            "prompt": user_prompt,  # 用户提示词（不含系统提示词）
            "api_payload": api_payload  # 实际发送给 Ollama API 的请求内容
        }


def ask_with_rag_stream(question: str, top_k: int = 3, retrieval_mode: str = "text", debug: bool = False):
    """流式版本的 RAG 问答"""
    from rag.retriever import KnowledgeRetriever
    
    retriever = KnowledgeRetriever()
    
    # 存储检索结果用于调试
    retrieval_results = []
    
    # 混合检索模式
    if retrieval_mode == "hybrid":
        text_result = retriever.search(question, top_k=top_k, mode="text")
        embedding_result = retriever.search(question, top_k=top_k, mode="embedding")
        
        # 融合检索结果（去重）- 检索器已返回排序好的结果
        merged_chunks = []
        seen_sources = set()
        
        # 先添加文本检索结果
        for chunk in text_result.chunks:
            source_key = f"{chunk.source}:{chunk.content[:50]}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                merged_chunks.append(chunk)
                # 记录检索结果用于调试
                retrieval_results.append({
                    'content': chunk.content,
                    'source': chunk.source,
                    'score': getattr(chunk, 'score', None),
                    'mode': 'text'
                })
        
        # 再添加向量检索结果（去重）
        for chunk in embedding_result.chunks:
            source_key = f"{chunk.source}:{chunk.content[:50]}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                merged_chunks.append(chunk)
                # 记录检索结果用于调试
                retrieval_results.append({
                    'content': chunk.content,
                    'source': chunk.source,
                    'score': getattr(chunk, 'score', None),
                    'mode': 'embedding'
                })
        
        # 取前 top_k 个
        merged_chunks = merged_chunks[:top_k]
        
        context = "\n\n".join([f"[片段{i+1}] 来源: {chunk.source}\n{chunk.content}" 
                              for i, chunk in enumerate(merged_chunks)])
        
        suggested_questions = list(dict.fromkeys(
            text_result.suggested_questions + embedding_result.suggested_questions
        ))[:5]
    else:
        retrieval_result = retriever.search(question, top_k=top_k, mode=retrieval_mode)
        
        # 记录检索结果用于调试
        for chunk in retrieval_result.chunks:
            retrieval_results.append({
                'content': chunk.content,
                'source': chunk.source,
                'score': getattr(chunk, 'score', None),
                'mode': retrieval_mode
            })
        
        context = "\n\n".join([f"[片段{i+1}] 来源: {chunk.source}\n{chunk.content}" 
                              for i, chunk in enumerate(retrieval_result.chunks)])
        suggested_questions = retrieval_result.suggested_questions
    
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    user_prompt = _build_user_prompt(question, context, suggested_questions)
    
    try:
        if provider == "ollama":
            ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct").strip()
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
            
            # 先返回检索信息
            yield {'type': 'info', 'retrieval_mode': retrieval_mode, 'suggested_questions': suggested_questions[:3]}
            
            # 如果是调试模式，返回检索结果
            if debug:
                yield {'type': 'retrieval', 'results': retrieval_results}
            
            # 如果是调试模式，返回prompt
            if debug:
                yield {'type': 'prompt', 'content': user_prompt}
            
            # 然后流式返回内容
            for chunk in _ask_with_ollama_stream(ollama_model, ollama_base_url, user_prompt):
                yield chunk
                
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                yield {'type': 'error', 'data': "LLM_PROVIDER=openai 但未检测到 OPENAI_API_KEY"}
                return
            
            # 如果是调试模式，返回检索结果和prompt
            if debug:
                yield {'type': 'retrieval', 'results': retrieval_results}
                yield {'type': 'prompt', 'content': user_prompt}
                
            answer = _ask_with_openai(os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(), api_key, user_prompt)
            yield {'type': 'info', 'retrieval_mode': retrieval_mode, 'suggested_questions': suggested_questions[:3]}
            yield {'type': 'content', 'data': answer}
        else:
            yield {'type': 'error', 'data': f"不支持的 LLM_PROVIDER={provider}"}
            
    except Exception as exc:
        yield {'type': 'error', 'data': f"LLM 调用失败（{exc}）"}


def ask_with_rag_hybrid(question: str, top_k: int = 3) -> dict:
    """混合检索：同时使用文本检索和向量检索，选择最优结果"""
    from rag.retriever import KnowledgeRetriever
    
    retriever = KnowledgeRetriever()
    
    # 并行执行两种检索
    text_result = retriever.search(question, top_k=top_k, mode="text")
    embedding_result = retriever.search(question, top_k=top_k, mode="embedding")
    
    # 融合检索结果（去重）- 检索器已返回排序好的结果
    merged_chunks = []
    seen_sources = set()
    
    # 先添加文本检索结果
    for chunk in text_result.chunks:
        source_key = f"{chunk.source}:{chunk.content[:50]}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            merged_chunks.append(chunk)
    
    # 再添加向量检索结果（去重）
    for chunk in embedding_result.chunks:
        source_key = f"{chunk.source}:{chunk.content[:50]}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            merged_chunks.append(chunk)
    
    # 取前 top_k 个
    merged_chunks = merged_chunks[:top_k]
    
    # 构建上下文
    context = "\n\n".join([f"[片段{i+1}] 来源: {chunk.source}\n{chunk.content}" 
                          for i, chunk in enumerate(merged_chunks)])
    
    # 获取推荐问题（合并并去重）
    suggested_questions = list(dict.fromkeys(
        text_result.suggested_questions + embedding_result.suggested_questions
    ))[:5]
    
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    user_prompt = _build_user_prompt(question, context, suggested_questions)
    
    answer = ""
    api_payload = None
    
    try:
        if provider == "ollama":
            ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct").strip()
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
            answer, api_payload = _ask_with_ollama(ollama_model, ollama_base_url, user_prompt)
            if not answer:
                answer = "Ollama 返回为空，建议检查模型是否已正确拉取并运行。"
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
            if not api_key:
                return {
                    "answer": "LLM_PROVIDER=openai 但未检测到 OPENAI_API_KEY，当前返回检索结果供人工参考。",
                    "context": context,
                }
            answer = _ask_with_openai(model, api_key, user_prompt)
        else:
            answer = f"不支持的 LLM_PROVIDER={provider}，请使用 ollama 或 openai。"
    except Exception as exc:
        answer = f"LLM 调用失败（{exc}）。当前返回检索结果供人工调试，请检查模型服务与配置。"
    
    return {
        "answer": answer,
        "context": context,
        "retrieval_mode": "hybrid",
        "suggested_questions": suggested_questions,
        "prompt": user_prompt,
        "api_payload": api_payload,
        "text_chunks": [c.content[:100] for c in text_result.chunks],
        "embedding_chunks": [c.content[:100] for c in embedding_result.chunks]
    }
