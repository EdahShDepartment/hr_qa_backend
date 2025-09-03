# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:10:07 2025

@author: ed116025
"""

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
def llm_qa(query):

    embedding = OpenAIEmbeddings()

    retriever = FAISS.load_local("./tools/faiss_hr_db",
                                embedding,
                                allow_dangerous_deserialization=True
                                ).as_retriever()
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa(query)
    response = {}
    response["query"] = result["query"]
    response["answer"] = result["result"]
    response["source_documents"] = [
        {
            "page_content": doc.page_content,
            "source": doc.metadata['source'],
            "section": doc.metadata['section'],
        } for doc in result["source_documents"]
    ]
    return response
