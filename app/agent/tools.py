"""Outils de l'agent juridique.

Version finance : search_documents, calculate, get_stock_price (yfinance).
Version juridique :
    search_legal_corpus : recherche hybride + reranking dans les articles
    get_article         : texte exact d'un article (« article 17 du RGPD »)
    find_legal_acts     : catalogue des actes du corpus (par nom, sigle ou thème)
"""
from langchain_core.documents import Document
from langchain_core.tools import tool

from app import config
from app.rag.corpus import get_corpus

_last_retrieved_chunks: list[Document] = []


def get_last_chunks() -> list[Document]:
    return _last_retrieved_chunks


def _remember(chunks: list[Document], append: bool = False) -> None:
    global _last_retrieved_chunks
    _last_retrieved_chunks = (_last_retrieved_chunks + chunks) if append else chunks


@tool
def search_legal_corpus(query: str) -> str:
    """Recherche dans les articles des 100 actes européens du corpus (RGPD, AI Act, DSA, DMA,
    NIS2, Data Act, DORA, CRA, ePrivacy…). À utiliser pour toute question de fond : obligations,
    droits, définitions, délais, sanctions. L'entrée est une requête précise en français,
    idéalement avec le vocabulaire juridique et le nom de l'acte."""
    from app.dependencies import get_vectordb
    from app.rag.reranker import rerank
    from app.rag.retriever import hybrid_retrieval

    try:
        candidates = hybrid_retrieval(query, get_vectordb(), k=config.CANDIDATES_K)
        chunks = rerank(query, candidates, top_k=config.TOP_K)
        _remember(chunks)
        if not chunks:
            return "Aucun article pertinent trouvé pour cette requête."
        return "\n\n".join(f"[Source {i}] {c.page_content}" for i, c in enumerate(chunks, start=1))
    except Exception as e:
        return f"Erreur lors de la recherche : {e}"


@tool
def get_article(acte: str, numero: str) -> str:
    """Renvoie le texte officiel complet d'un article précis.
    `acte` : nom usuel, sigle ou numéro de l'acte (ex. 'RGPD', 'AI Act', 'NIS2', '2016/679', 'directive 2002/58/CE').
    `numero` : numéro de l'article (ex. '17')."""
    corpus = get_corpus()
    art = corpus.get_article(acte, numero)
    if art is None:
        celex = corpus.resolve_act(acte)
        if not celex:
            return f"Acte « {acte} » introuvable dans le corpus. Utilise find_legal_acts pour le retrouver."
        return f"L'article {numero} de {corpus.doc_label(celex)} est introuvable dans le corpus."
    header = corpus.article_header(art)
    _remember([Document(page_content=f"{header}\n{art.text}",
                        metadata={"article_id": art.id, "celex": art.celex, "article_num": art.num,
                                  "doc_label": corpus.doc_label(art.celex), "article_title": art.title})],
              append=True)
    text = art.text if len(art.text) <= 8000 else art.text[:8000] + " […] (article tronqué)"
    return f"{header}\n{text}"


@tool
def find_legal_acts(query: str) -> str:
    """Liste les actes du corpus correspondant à un nom, un sigle ou un thème
    (ex. 'cybersécurité', 'plateformes', 'NIS2', 'données de santé'). Renvoie le nom, le type,
    la date, le nombre d'articles et l'identifiant CELEX de chaque acte."""
    docs = get_corpus().search_documents(query)
    if not docs:
        return f"Aucun acte du corpus ne correspond à « {query} »."
    lines = [f"- {d.label} — {d.doc_type}, {d.date}, {d.n_articles} articles, CELEX {d.celex}\n  {d.title[:220]}"
             for d in docs]
    return "\n".join(lines)
