"""Stratégies de recherche (elles renvoient des CANDIDATS ; le budget final de k articles
distincts est appliqué dans `pipeline.retrieve`).

    vector_search      : similarité cosinus seule (= RAG vectoriel « classique »)
    hybrid_retrieval   : vectoriel + BM25 corpus, fusion RRF
    advanced_retrieval : MultiQuery + HyDE + BM25, fusion RRF -> candidats à reranker (mis en cache)
"""
import json
import threading
from functools import lru_cache

from langchain_core.documents import Document

from app import config
from app.rag.hybrid_search import bm25_search, reciprocal_rank_fusion
from app.rag.hyde import hyde_answer
from app.rag.multiquery import generate_queries


def vector_search(question: str, vectordb, k: int = config.TOP_K, filters: dict | None = None) -> list[Document]:
    return vectordb.similarity_search(question, k=k, filter=filters or None)


def hybrid_retrieval(question: str, vectordb, k: int | None = config.TOP_K, filters: dict | None = None,
                     candidates: int = config.CANDIDATES_K) -> list[Document]:
    """`k=None` : tout le classement fusionné (jusqu'à 2 × `candidates` chunks)."""
    dense = vector_search(question, vectordb, k=candidates, filters=filters)
    sparse = bm25_search(question, top_k=candidates, filters=filters)
    return reciprocal_rank_fusion([dense, sparse], top_n=k)


# ---------------------------------------------------------------------------------------------
# advanced : candidats mis en cache
# ---------------------------------------------------------------------------------------------
_cache_lock = threading.Lock()


@lru_cache(maxsize=1)
def _chunks_by_id() -> dict[str, Document]:
    from app.rag.hybrid_search import get_bm25_index

    return {c.metadata["chunk_id"]: c for c in get_bm25_index().chunks}


def _load_cache() -> dict:
    if config.SEED_CACHE_PATH.exists():
        return json.loads(config.SEED_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _cache_key(question: str, filters: dict | None, per_query: int, candidates: int) -> str:
    return json.dumps({"q": question.strip(), "f": filters, "per_query": per_query, "n": candidates},
                      ensure_ascii=False, sort_keys=True)


def advanced_retrieval(question: str, vectordb, k: int = config.CANDIDATES_K // 2, filters: dict | None = None,
                       candidates: int = config.CANDIDATES_K, use_cache: bool | None = None) -> list[Document]:
    """Renvoie jusqu'à `candidates` chunks fusionnés par RRF (à passer ensuite au reranker).
    `k` = nombre de résultats demandés à chaque requête vectorielle.

    Cache (`config.SEED_CACHE_PATH`) : MultiQuery et HyDE sont générés par le LLM, donc varient d'une
    exécution à l'autre. Le cache fige les candidats d'une question ; le Graph RAG relit les mêmes
    (mêmes graines), et une réévaluation ne reconsomme pas de quota Groq.
    """
    use_cache = config.SEED_CACHE if use_cache is None else use_cache
    key = _cache_key(question, filters, k, candidates)
    if use_cache:
        cached = _load_cache().get(key)
        if cached is not None:
            by_id = _chunks_by_id()
            return [by_id[cid] for cid in cached if cid in by_id]

    result_lists = []
    # 1. MultiQuery : question originale + 3 reformulations
    for query in generate_queries(question):
        result_lists.append(vector_search(query, vectordb, k=k, filters=filters))
    # 2. HyDE : recherche avec un article hypothétique
    result_lists.append(vector_search(hyde_answer(question), vectordb, k=k, filters=filters))
    # 3. BM25 sur tout le corpus (termes exacts : numéros d'articles, sigles, délais)
    result_lists.append(bm25_search(question, top_k=max(k, 10), filters=filters))

    fused = reciprocal_rank_fusion(result_lists, top_n=candidates)
    if use_cache:
        with _cache_lock:
            cache = _load_cache()
            cache[key] = [c.metadata["chunk_id"] for c in fused]
            config.SEED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            config.SEED_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return fused


def build_filter(celex: str | None = None, doc_type: str | None = None,
                 year_min: int | None = None, year_max: int | None = None) -> dict | None:
    """Filtres de métadonnées au format Chroma (remplacent company / year de la version finance)."""
    clauses = []
    if celex:
        clauses.append({"celex": celex})
    if doc_type:
        clauses.append({"doc_type": doc_type.upper()})
    if year_min:
        clauses.append({"year": {"$gte": int(year_min)}})
    if year_max:
        clauses.append({"year": {"$lte": int(year_max)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
