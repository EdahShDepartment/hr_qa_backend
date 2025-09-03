# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:15:15 2025

@author: ed116025
"""

from docx import Document
import glob
import re
from langchain.docstore.document import Document as LangChainDoc


def extract_section(text):
    """
    從段落文字提取條款號，例如 '3.1.9 生理假'
    """
    match = re.match(r"^(\d+(\.\d+)+)", text)
    return match.group(1) if match else None

def load_all_docx_with_metadata(folder_path):
    all_docs = []
    for filepath in glob.glob(f"{folder_path}/*.docx"):
        file_name = filepath.split("/")[-1].replace(".docx", "")
        doc = Document(filepath)
        current_section = None  # 記住上一個條款號

        for para in doc.paragraphs:
            content = para.text.strip()
            if content:
                match = re.match(r"^(\d+(\.\d+)+)\s", content)
                if match:
                    current_section = match.group(1)
                metadata = {
                    "source": file_name,
                    "section": current_section if current_section else "無條款編號"
                }
                all_docs.append(LangChainDoc(page_content=content, metadata=metadata))
    return all_docs

#Read documents in folder
docs = load_all_docx_with_metadata("HR_docs")

# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# import os


# embedding = OpenAIEmbeddings()


# db = FAISS.from_documents(docs, embedding)
# db.save_local("faiss_hr_db")
# print("✅ 向量資料庫已建立並儲存為 faiss_hr_db")

