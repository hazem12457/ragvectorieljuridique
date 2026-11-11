"""Pipeline RAG vectoriel : recherche -> contexte -> génération.

Point d'entrée unique utilisé par l'API, l'agent et l'évaluation. Le Graph RAG
(étape suivante du projet) n'aura qu'à fournir sa propre fonction de recherche :
`format_context` et `generate_answer` (même prompt, même LLM, même TOP_K) sont partagés.

Modes de recherche :
    vector   : similarité cosinus seule — le RAG vectoriel « classique »
    hybrid   : vectoriel + BM25, fusion RRF
    advanced : MultiQuery + HyDE + BM25 + fusion RRF + reranking cross-encoder — LA baseline du Graph RAG

Budget identique pour tous les modes (et pour le Graph RAG) : k ARTICLES DISTINCTS dans le contexte,
chacun représenté par son chunk le mieux classé (`top_articles`). Avant, k chunks pouvaient venir
du même article (ex. 3 morceaux de l'article 33 du RGPD), ce qui faussait la comparaison.
"""
from langchain_core.documents import Document
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app import config
from app.llm import get_llm, record_tokens
from app.rag.corpus import get_corpus
from app.rag.prompts import ANSWER_PROMPT
from app.rag.reranker import rerank
from app.rag.retriever import advanced_retrieval, hybrid_retrieval, vector_search

MODES = ("vector", "hybrid", "advanced")
NOT_AVAILABLE = "Le modèle de langage est momentanément indisponible, merci de réessayer."


def top_articles(chunks: list[Document], k: int) -> list[Document]:
    """Garde le premier chunk (le mieux classé) de chaque article, jusqu'à k articles distincts."""
    seen, out = set(), []
    for c in chunks:
        aid = c.metadata.get("article_id") or c.metadata.get("chunk_id")
        if aid not in seen:
            seen.add(aid)
            out.append(c)
            if len(out) == k:
                break
    return out


def retrieve(question: str, mode: str = "vector", k: int = config.TOP_K,
             filters: dict | None = None, vectordb=None) -> list[Document]:
    """k chunks issus de k articles distincts."""
    if vectordb is None:
        from app.dependencies import get_vectordb
        vectordb = get_vectordb()
    if mode == "vector":
        ranked = vector_search(question, vectordb, k=max(config.CANDIDATES_K, 4 * k), filters=filters)
    elif mode == "hybrid":
        ranked = hybrid_retrieval(question, vectordb, k=None, filters=filters,
                                  candidates=max(config.CANDIDATES_K, 4 * k))
    elif mode == "advanced":
        ranked = rerank(question, advanced_retrieval(question, vectordb, filters=filters), top_k=None)
    else:
        raise ValueError(f"mode inconnu : {mode!r} (attendu : {MODES})")
    return top_articles(ranked, k)


def format_context(chunks: list[Document]) -> str:
    """Extraits numérotés [1], [2]… — chaque chunk porte déjà son en-tête « Acte — Article — Titre »."""
    return "\n\n".join(f"[{i}] {c.page_content}" for i, c in enumerate(chunks, start=1))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
def _invoke(chain, payload):
    return chain.invoke(payload)


def generate_answer(question: str, chunks: list[Document], chat_history: str = "",
                    model: str | None = None, prompt=ANSWER_PROMPT) -> str:
    model = model or config.LLM_MODEL
    if not chunks:
        return "Aucun article pertinent n'a été trouvé dans le corpus pour cette question."
    try:
        result = _invoke(prompt | get_llm(model), {
            "context": format_context(chunks),
            "question": question,
            "chat_history": chat_history or "(aucun)",
        })
        record_tokens(result, model, "answer")
        return result.content
    except Exception as e:
        logger.warning(f"Génération échouée ({type(e).__name__}) : {e}")
        return NOT_AVAILABLE


def answer_question(question: str, mode: str = "vector", k: int = config.TOP_K,
                    filters: dict | None = None, chat_history: str = "",
                    search_query: str | None = None) -> dict:
    """`search_query` permet de chercher avec une requête enrichie (ex. historique) tout en
    répondant à la question d'origine."""
    chunks = retrieve(search_query or question, mode=mode, k=k, filters=filters)
    answer = generate_answer(question, chunks, chat_history=chat_history)
    return {"question": question, "mode": mode, "answer": answer, "chunks": chunks}


def article_ids(chunks: list[Document]) -> list[str]:
    """Identifiants d'articles distincts, dans l'ordre de classement (pour Recall@k / MRR)."""
    seen, ids = set(), []
    for c in chunks:
        aid = c.metadata.get("article_id")
        if aid and aid not in seen:
            seen.add(aid)
            ids.append(aid)
    return ids


def sources_payload(chunks: list[Document], preview_chars: int = 300) -> list[dict]:
    corpus = get_corpus()
    out = []
    for i, c in enumerate(chunks, start=1):
        m = c.metadata
        doc = corpus.documents.get(m.get("celex"))
        body = c.page_content.split("\n", 1)[-1]
        out.append({
            "ref": i,
            "article_id": m.get("article_id"),
            "celex": m.get("celex"),
            "acte": m.get("doc_label"),
            "article": m.get("article_num"),
            "titre": m.get("article_title"),
            "type": m.get("doc_type"),
            "annee": m.get("year"),
            "url": doc.url if doc else None,
            "extrait": body[:preview_chars],
        })
    return out
