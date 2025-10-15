#!/usr/bin/env python3
"""
SiliconFlow 嵌入 API 包装器
直接调用 SiliconFlow API 进行嵌入，避免下载本地模型
"""

import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings


class SiliconFlowEmbeddings(Embeddings):
    """SiliconFlow 嵌入 API 包装器"""

    def __init__(self):
        # 加载环境变量
        load_dotenv()
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        self.base_url = os.getenv("SILICONFLOW_BASE_URL")
        self.model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")  # 支持8k上下文

        if not self.api_key:
            raise ValueError("请设置 SILICONFLOW_API_KEY 环境变量")
        if not self.base_url:
            raise ValueError("请设置 SILICONFLOW_BASE_URL 环境变量")

    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """调用 SiliconFlow 嵌入 API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()

            # 提取嵌入向量
            embeddings = []
            for item in result.get("data", []):
                embeddings.append(item["embedding"])

            return embeddings

        except Exception as e:
            print(f"嵌入 API 调用失败: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表（分批）"""
        batch_size = int(os.getenv("EMBED_BATCH", "64"))
        output: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            output.extend(self._call_embedding_api(texts[i : i + batch_size]))
        return output

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        embeddings = self._call_embedding_api([text])
        return embeddings[0] if embeddings else []
