"""Reranking par cross-encoder : le modèle lit question + chunk ensemble, bien plus précis que la
similarité cosinus pour classer les candidats finaux.

Règle de comparaison : LE MÊME reranker est utilisé par tous les modes qui rerankent
(advanced, et graph dans le projet Graph RAG).

    RERANKER=local  (défaut) : BAAI/bge-reranker-v2-m3 en local (multilingue, gratuit, reproductible)
    RERANKER=cohere          : Cohere rerank-v3.5 (API, demande COHERE_API_KEY)

En cas d'échec, l'ordre d'entrée est conservé (et un avertissement est journalisé).
"""
import os
import time
from functools import lru_cache

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app import config
from app.monitoring.metrics import RERANK_LATENCY


@lru_cache(maxsize=1)
def get_cross_encoder():
    from sentence_transformers import CrossEncoder

    logger.info(f"Chargement du reranker local {config.LOCAL_RERANK_MODEL}")
    return CrossEncoder(config.LOCAL_RERANK_MODEL, max_length=512, device=config.EMBEDDING_DEVICE)


@lru_cache(maxsize=1)
def get_cohere_client():
    import cohere

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY non définie (RERANKER=cohere)")
    return cohere.ClientV2(api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def _cohere_scores(question: str, texts: list[str]) -> list[float]:
    results = get_cohere_client().rerank(model=config.RERANK_MODEL, query=question, documents=texts,
                                         top_n=len(texts)).results
    scores = [0.0] * len(texts)
    for r in results:
        scores[r.index] = r.relevance_score
    return scores


def score(question: str, texts: list[str]) -> list[float]:
    """Score de pertinence de chaque texte pour la question (plus haut = plus pertinent)."""
    if config.RERANKER == "cohere":
        return _cohere_scores(question, texts)
    model = get_cross_encoder()
    return model.predict([(question, t) for t in texts], batch_size=config.RERANK_BATCH_SIZE,
                         show_progress_bar=False).tolist()


def rerank(question: str, chunks: list, top_k: int | None = config.TOP_K) -> list:
    """Chunks triés par score du cross-encoder. `top_k=None` : renvoie tout le classement."""
    if not chunks:
        return chunks
    start, outcome = time.time(), "failure"
    try:
        scores = score(question, [c.page_content for c in chunks])
        outcome = "success"
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        ranked = [chunks[i] for i in order]
    except Exception as e:
        logger.warning(f"Reranking échoué ({type(e).__name__}: {e}) : ordre d'origine conservé")
        ranked = list(chunks)
    finally:
        RERANK_LATENCY.labels(outcome=outcome).observe(time.time() - start)
    return ranked if top_k is None else ranked[:top_k]
