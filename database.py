import os
import hashlib
import pickle
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def load_vector_db(db_path):
    load_dotenv()
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    vector_db = Chroma(
        embedding_function=embedding_model,
        persist_directory=db_path
    )
    print(f"向量数据库已从 {db_path} 加载")
    return vector_db


def get_searching_results(query, vector_db, embedding_model, res_num=5):
    """
    使用向量相似度搜索获取相似代码片段
    """
    try:
        query_embedding = embedding_model.embed_query(query)

        results = vector_db.similarity_search_by_vector(query_embedding, k=res_num)

        searching_results = "\nResults:\n" + "\n\nResult:\n\n".join([doc.page_content for doc in results])

        return searching_results

    except Exception as e:
        error_msg = f"搜索过程发生错误: {str(e)}"
        print(error_msg)
        return error_msg

def main():
    db_path = 'chroma_db'

    vector_db = load_vector_db(db_path)

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    results = get_searching_results("如何防止缓冲区溢出？", vector_db, embedding_model,  5)

    for idx, doc in enumerate(results):
        print(f"结果 {idx+1}:")
        print(doc.page_content)
        print(doc.metadata)
        print('-' * 50)



if __name__ == "__main__":
    main()
