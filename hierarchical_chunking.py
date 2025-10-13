from langchain_text_splitters import MarkdownHeaderTextSplitter


def load_readme():
    with open("/private/tmp/langchain/README.md", "r", encoding="utf-8") as f:
        return f.read()


def main():
    text = load_readme()

    # 使用 LangChain 的 Markdown 头部切分器实现分层切分
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    chunks = splitter.split_text(text)

    print(f"分层切分: {len(chunks)} 块")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n块 {i+1} (长度: {len(chunk.page_content)} 字符):")
        print("-" * 40)
        preview = chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
        print(preview)
        if chunk.metadata:
            print(f"元数据: {chunk.metadata}")


if __name__ == "__main__":
    main()