from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_readme():
    with open("/private/tmp/langchain/README.md", "r", encoding="utf-8") as f:
        return f.read()


def main():
    text = load_readme()

    # 使用 LangChain 的递归字符切分器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    print(f"递归字符切分: {len(chunks)} 块")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n块 {i+1} (长度: {len(chunk)} 字符):")
        print("-" * 40)
        preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
        print(preview)


if __name__ == "__main__":
    main()