#!/usr/bin/env python3
"""
简单查询扩展器
使用 LLM 将普通查询转换为专业术语
遵循 Unix 哲学：简单、专注、可组合
"""

import os
import requests
import json
from dotenv import load_dotenv


def expand_query_simple(original_query: str) -> list[str]:
    """
    将查询扩展为专业术语

    原则：
    1. 专注单一功能
    2. 输入输出简单明确
    3. 错误时优雅降级
    """

    # 加载环境变量
    load_dotenv()
    api_key = os.getenv("SILICONFLOW_API_KEY")
    base_url = os.getenv("SILICONFLOW_BASE_URL")
    model = os.getenv("LLM_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")

    if not api_key:
        # 优雅降级：没有 API key 时返回原查询
        return [original_query]

    # 简单直接的提示词
    prompt = f"""将以下用户查询转换为3个更专业的技术搜索查询，专注于技术栈、依赖、说明、教程等主题：

原始查询: "{original_query}"

返回格式：每行一个查询，不要其他内容"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "max_tokens": 200,
        "temperature": 0.3
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()

        # 简单解析：按行分割，过滤空行
        expanded_queries = [
            line.strip()
            for line in content.split('\n')
            if line.strip() and len(line.strip()) > 5
        ]

        # 确保包含原始查询
        if original_query not in expanded_queries:
            expanded_queries.insert(0, original_query)

        return expanded_queries[:4]  # 限制数量，保持简单

    except Exception:
        # 优雅降级：API 失败时返回原查询（静默）
        return [original_query]


def demonstrate_expansion():
    """演示查询扩展功能"""
    test_queries = [
        "这个项目是干什么的?",
        "怎么安装使用?",
        "有什么功能?"
    ]

    print("=" * 50)
    print("简单查询扩展演示")
    print("=" * 50)

    for query in test_queries:
        print(f"\n原始查询: {query}")
        expanded = expand_query_simple(query)
        print(f"扩展后: {expanded}")


if __name__ == "__main__":
    demonstrate_expansion()
