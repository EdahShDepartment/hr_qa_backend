from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from langchain.chat_models import ChatOpenAI
import os
from langchain.embeddings import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# CrossEncoder reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# BM25 corpus
bm25_corpus = []
bm25_model = None

def load_bm25_corpus(docs):
    global bm25_corpus, bm25_model
    bm25_corpus = [doc.page_content.split() for doc in docs]
    bm25_model = BM25Okapi(bm25_corpus)

# 檢索模式選擇：'faiss', 'hybrid', 'hybrid_rerank'
RETRIEVAL_MODE = 'hybrid_rerank'  # 可改成 'hybrid' 或 'hybrid_rerank'

def faiss_search(query, retriever, top_k=10):
    return retriever.get_relevant_documents(query, top_k=top_k)

def hybrid_search(query, retriever, top_k=10):
    # 向量檢索
    vector_docs = retriever.get_relevant_documents(query, top_k=top_k)
    # BM25 檢索 (若有建)
    bm25_results = []
    if bm25_model:
        tokens = query.split()
        bm25_scores = bm25_model.get_scores(tokens)
        bm25_results = sorted(
            zip(vector_docs, bm25_scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
    # 合併去重
    all_docs = {doc.page_content: doc for doc in vector_docs}
    for doc, _ in bm25_results:
        all_docs[doc.page_content] = doc
    combined_docs = list(all_docs.values())
    return combined_docs[:top_k]

def hybrid_rerank_search(query, retriever, top_k=10, rerank_k=10):
    # Hybrid (向量+BM25)
    combined_docs = hybrid_search(query, retriever, top_k=top_k)
    # Rerank
    pairs = [(query, doc.page_content) for doc in combined_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(combined_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in ranked[:rerank_k]]
    return top_docs


def llm_qa(query, db_ver, retrieval_mode=RETRIEVAL_MODE):
    embedding = OpenAIEmbeddings()
    # Load FAISS
    if db_ver == 'v1':
        db_path = './tools/faiss_hr_db'
    else:
        db_path = './tools/faiss_hr_db_v2'
    db = FAISS.load_local(
        db_path,
        embedding,
        allow_dangerous_deserialization=True
    )
    print(db_path)
    retriever = db.as_retriever(search_kwargs={"k": 20})

    # Build BM25 corpus once
    global bm25_corpus, bm25_model
    if not bm25_corpus:
        all_docs = list(db.docstore._dict.values())
        load_bm25_corpus(all_docs)

    # 檢索三模式
    if retrieval_mode == 'faiss':
        relevant_docs = faiss_search(query, retriever, top_k=10)
    elif retrieval_mode == 'hybrid':
        relevant_docs = hybrid_search(query, retriever, top_k=10)
    elif retrieval_mode == 'hybrid_rerank':
        relevant_docs = hybrid_rerank_search(query, retriever, top_k=10, rerank_k=10)
    else:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")

    # Prepare LLM prompt
    context = "\n\n".join([
        f"【來源: {doc.metadata.get('source', '未知')} | 條號: {doc.metadata.get('section', '未知')}】\n{doc.page_content}"
        for doc in relevant_docs
    ])
    prompt = f"""
              你是資深人資專員，請僅根據提供的文件回答。
                1.  若內容中包含明確答案、或可合理推論出答案：
                        * 先直接提供精簡的答案，不要加上「結論」等任何標題。
                        * 換行後，以此格式提供引證：「引用條例：【來源: <來源文件> | 條號: <來源條號>】<不包含條號的引用內容>」。

                2.  若內容完全無法回答或推論：
                        * 請明確回答「查無相關規定」。

                3.  所有回答與推論皆須完全基於提供的資料，絕不自行延伸或杜撰。

                資料如下：
                {context}
                問題：{query}
              """
              
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    answer = llm.predict(prompt)

    response = {
        "query": query,
        "answer": answer,
        "source_documents": [
            {
                "page_content": doc.page_content,
                "source": doc.metadata.get('source', '未知'),
                "section": doc.metadata.get('section', '未知')
            } for doc in relevant_docs
        ]
    }
    return response

