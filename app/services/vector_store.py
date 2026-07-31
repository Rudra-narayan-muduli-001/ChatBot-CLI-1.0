import json
import logging
from pathlib import Path
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    LEARNING_DATA_DIR,
    CHATS_DATA_DIR,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


logger = logging.getLogger(__name__)


class VectorStoreService:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.vector_store: Optional[FAISS] = None

    def load_learning_data(self) -> List[Document]:
        docoments = []
        for file_path in list(LEARNING_DATA_DIR.glob("*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        docoments.append(Document(page_content=content, metadata={"source": str(file_path.name)}))
            except Exception as e:
                logger.warning("Could not load learning data file %s: %s", file_path, e)
        return docoments

    def load_chat_history(self) -> List[Document]:
        documents = []
        for file_path in list(CHATS_DATA_DIR.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chat_data = json.load(f)
                messages = chat_data.get("messages", [])
                chat_content = "\n".join([
                    f"User: {msg.get('content', '')}" if msg.get("role") == "user"
                    else f"Assistant: {msg.get('content', '')}"
                    for msg in messages
                ])
                if chat_content.strip():
                    documents.append(Document(page_content=chat_content, metadata={"source": f"chat_{file_path.stem}"}))
            except Exception as e:
                logger.warning("Could not load chat history file %s: %s", file_path, e)
        return documents

    def create_vector_store(self) -> FAISS:
        learning_docs = self.load_learning_data()
        chat_docs = self.load_chat_history()
        all_documents = learning_docs + chat_docs

        if not all_documents:
            self.vector_store = FAISS.from_texts(["No data available yet."], self.embeddings)
        else:
            chunks = self.text_splitter.split_documents(all_documents)
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        self.save_vector_store()
        return self.vector_store

    def save_vector_store(self):
        if not self.vector_store:
            try:
                self.vector_store.save_local(str(VECTOR_STORE_DIR))
            except Exception as e:
                logger.error("Failed to save vector store to disk: %s", e)

    def get_retriever(self, k: int = 10):
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. This should not happen.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})
