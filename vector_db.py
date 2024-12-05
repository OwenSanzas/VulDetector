import json
import os
import shutil
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # 更新导入路径
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

def load_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

def create_documents(data):
    documents = []
    for cwe_id, entries in data.items():
        for entry in entries:
            func_before = entry.get('func_before', '')
            description = entry.get('description', '')
            vul = entry.get('vul', 0)
            metadata = {
                'CWE ID': cwe_id,
                'vul': vul,
                'description': description
            }
            content = f"{func_before}\n\nDescription:\n{description}"
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"将 {len(documents)} 个文档分割成 {len(chunks)} 个文本块。")
    return chunks

def create_vector_db(chunks, db_path):
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # 加载环境变量，获取 OpenAI API 密钥
    load_dotenv()

    # 使用 OpenAI 的嵌入模型
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    # 创建向量数据库
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )
    vector_db.persist()
    print(f"向量数据库已创建并保存到 {db_path}")
    return vector_db

def main():
    json_file_path = 'data/rag_data.json'
    db_path = 'chroma_db_non_vul'

    data = load_data(json_file_path)
    documents = create_documents(data)
    chunks = split_text(documents)
    vector_db = create_vector_db(chunks, db_path)

    # 示例查询
    query = "如何防止缓冲区溢出？"
    results = vector_db.similarity_search(query, k=5)
    for idx, doc in enumerate(results):
        print(f"结果 {idx+1}:")
        print(doc.page_content)
        print(doc.metadata)
        print('-' * 50)

if __name__ == "__main__":
    main()
