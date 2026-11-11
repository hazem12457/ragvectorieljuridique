"""Embeddings multilingues pour les textes juridiques français.

Modèle par défaut : intfloat/multilingual-e5-base (768 dimensions, 512 tokens max).
Les modèles E5 attendent un préfixe : « query: » pour les questions et
« passage: » pour les documents. On l'ajoute ici, de façon transparente.
Les vecteurs sont normalisés : produit scalaire = similarité cosinus.

C'est ce même modèle qui sera réutilisé pour les embeddings des nœuds Article
dans Neo4j et pour le Graph RAG (règle de comparaison équitable).
"""
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app import config


class SentenceEmbeddings(Embeddings):
    def __init__(self, model_name: str, batch_size: int = 16, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        # `model` = nom du modèle (chaîne) : RAGAS lit cet attribut pour ses statistiques d'usage
        self.model = self.model_name = model_name
        self._st = SentenceTransformer(model_name, device=device)
        self._st.max_seq_length = min(self._st.max_seq_length or 512, 512)
        self.batch_size = batch_size
        self.is_e5 = "e5" in model_name.lower()

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        if self.is_e5:
            texts = [prefix + t for t in texts]
        vectors = self._st.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, "passage: ")

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], "query: ")[0]


@lru_cache(maxsize=1)
def get_embeddings() -> SentenceEmbeddings:
    return SentenceEmbeddings(
        config.EMBEDDING_MODEL,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        device=config.EMBEDDING_DEVICE,
    )
