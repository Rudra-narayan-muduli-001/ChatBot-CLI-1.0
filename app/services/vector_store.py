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


logger = logging.getLogger("J.A.R.V.I.S")


# =========================================================================
# VECTOR STORE SERVICE CLASS
# =========================================================================

class VectorStoreService:
    
    def __init__(self):
        """Creat the embedding model (local) and text splitter; vector_store is set in create_vector_store()."""
        # Embedding run locally (no API key); used to convert text into vectors for similarity search.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.vector_store: Optional[FAISS] = None

    # --------------------------------------------------------------
    # LOAD DOCOMENTS FROM DISK
    # --------------------------------------------------------------

    def load_learning_data(self) -> List[Document]:
        """Read all  .txt files in database/learning_data/ and return one Docoment per file (content + source name)."""
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
        """Load all .json files in database/chat_data/; turn each into one Docoment (User:/Assistant: lines)."""
        documents = []
        for file_path in list(CHATS_DATA_DIR.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chat_data = json.load(f)
                messages = chat_data.get("messages", [])
                # Format as "User: ..."/ "Assistant: ..." so the retriever can match past conversations.
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
    
    # ------------------------------------------------------------------
    # BUILD AND SAVE FAISS INDEX
    # ------------------------------------------------------------------

    def create_vector_store(self) -> FAISS:
        """
        Load learning_data + chats_data, chunk, embed, build FAISS index, save to disk.
        Called once at startup. If there are no docoments we create a tiny placeholder index.
        """
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
        """Write the current FAISS index to database/vector_store/. On error we only log."""

        if not self.vector_store:
            try:
                self.vector_store.save_local(str(VECTOR_STORE_DIR))
            except Exception as e:
                logger.error("Failed to save vector store to disk: %s", e)

    # -----------------------------------------------------------
    # RETRIEVER FOR CONTEXT
    # -----------------------------------------------------------

    def get_retriever(self, k: int = 10):
        """Return a retriever that returns the k most similar chunks for a query string."""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. This should not happen.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    