from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def get_vectordb():
    """Collection Chroma unique, partagée par l'API, l'agent et l'évaluation."""
    from langchain_chroma import Chroma

    from app.rag.embeddings import get_embeddings

    return Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.VECTORSTORE_DIR,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
