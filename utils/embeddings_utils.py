# utils/embeddings_utils.py
import os
from neo4j_graphrag.embeddings import OllamaEmbeddings

def get_embeddings():
    model = os.getenv("OLLAMA_EMBEDDING_MODEL")
    if not model:
        raise ValueError("OLLAMA_EMBEDDING_MODEL not set in .env")
    emb = OllamaEmbeddings(model=model)
    # Provide a small wrapper to match methods used in app:
    class E:
        def embed_documents(self, texts):
            # Returns list[list[float]]
            return emb.embed_documents(texts)
        def embed_query(self, text):
            return emb.embed_query(text)
    return E()
