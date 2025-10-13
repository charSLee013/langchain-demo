#!/usr/bin/env python3
"""
结构切分演示
基于文档结构进行智能切分
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


def structural_chunking(text: str) -> List[Document]:
    """基于文档结构进行切分"""

    # 定义要分割的头部级别
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # 使用 Markdown 头部切分器
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    chunks = splitter.split_text(text)
    return chunks


def demonstrate_structural_chunking():
    """演示结构切分功能"""
    print("=" * 60)
    print("结构切分技术演示")
    print("=" * 60)

    # 示例结构化文档
    sample_text = """
# 人工智能概述

人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器。

## 机器学习

机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。

### 监督学习

监督学习使用标记数据训练模型，用于分类和回归任务。

### 无监督学习

无监督学习在没有标签的数据中发现模式和结构。

## 深度学习

深度学习使用多层神经网络进行复杂的模式识别。

### 神经网络

神经网络由相互连接的节点组成，模拟人脑的工作方式。

### 卷积神经网络

卷积神经网络专门用于图像识别和计算机视觉任务。

## 自然语言处理

自然语言处理使计算机能够理解和生成人类语言。

### 文本分类

文本分类将文档分配到预定义的类别中。

### 情感分析

情感分析确定文本中表达的情感倾向。
"""

    print(f"\n原始文档长度: {len(sample_text)} 字符")
    print(f"原始文档内容预览:")
    print("-" * 40)
    print(sample_text[:200] + "...")

    # 进行结构切分
    chunks = structural_chunking(sample_text)

    print(f"\n结构切分结果: {len(chunks)} 个块")
    print("-" * 40)

    for i, chunk in enumerate(chunks[:5]):  # 显示前5个块
        print(f"\n块 {i+1} (长度: {len(chunk.page_content)} 字符):")
        print(f"元数据: {chunk.metadata}")
        preview = chunk.page_content[:80] + "..." if len(chunk.page_content) > 80 else chunk.page_content
        print(f"内容: {preview}")

    print(f"\n总结: 结构切分基于文档层级结构，保持语义完整性")


if __name__ == "__main__":
    demonstrate_structural_chunking()