"""Découpage du corpus en chunks et indexation dans ChromaDB.

Unité de base = l'ARTICLE (et non une page de PDF comme dans la version finance) :
  * un article court (la majorité, médiane ≈ 1 450 caractères) = 1 chunk ;
  * un article long est sous-découpé (~1 500 caractères, chevauchement 200) en
    coupant de préférence entre paragraphes numérotés « 1. », « 2. », puis entre
    points « a) », « b) », puis entre phrases.
Chaque chunk commence par un en-tête « Acte — Article n — Titre » : sans lui, un
sous-chunk comme « 2. Le paragraphe 1 ne s'applique pas… » serait inexploitable.

Usage :
    python -m app.rag.chunking            # (ré)indexe tout le corpus
    python -m app.rag.chunking --limit 50 # test rapide sur 50 articles
"""
import argparse
import json
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app import config
from app.rag.corpus import Corpus, get_corpus

SEPARATORS = [
    r"\n\n",
    r"\n",
    r"\s(?=\d{1,3}\.\s+[A-ZÀ-ÖØ-Ý«])",      # avant « 2. Le … »
    r"\s(?=[a-z]{1,2}\)\s)",                # avant « a) », « bb) »
    r"\s(?=\d{1,3}\)\s)",                   # avant « 1) » (définitions)
    r"(?<=[.;:])\s",                        # fin de phrase
    r"\s",
    "",
]


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=SEPARATORS,
        is_separator_regex=True,
        # keep_separator=True : le séparateur (un espace) est conservé et les morceaux sont
        # recollés avec "" ; avec False, LangChain recollerait avec la regex littérale.
        keep_separator=True,
    )


def build_chunks(corpus: Corpus | None = None, limit: int | None = None) -> list[Document]:
    corpus = corpus or get_corpus()
    splitter = get_splitter()
    chunks: list[Document] = []

    articles = list(corpus.articles.values())
    if limit:
        articles = articles[:limit]

    for art in articles:
        text = art.text.strip()
        if not text:
            continue
        doc = corpus.documents.get(art.celex)
        header = corpus.article_header(art)
        pieces = [text] if len(text) <= config.CHUNK_SIZE else splitter.split_text(text)
        for i, piece in enumerate(pieces):
            chunks.append(
                Document(
                    page_content=f"{header}\n{piece.strip()}",
                    metadata={
                        "chunk_id": f"{art.id}_c{i}",
                        "article_id": art.id,
                        "celex": art.celex,
                        "doc_label": corpus.doc_label(art.celex),
                        "doc_type": art.doc_type,
                        "year": doc.year if doc else 0,
                        "date": art.date,
                        "article_num": art.num,
                        "article_title": art.title,
                        "eurovoc": "|".join(art.eurovoc),
                        "chunk_index": i,
                        "n_chunks": len(pieces),
                    },
                )
            )
    return chunks


def save_chunks(chunks: list[Document], path=config.CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.page_content, "metadata": c.metadata}, ensure_ascii=False) + "\n")


def load_chunks(path=config.CHUNKS_PATH) -> list[Document]:
    """Chunks sauvegardés lors de l'ingestion (ou reconstruits à la volée s'ils n'existent pas)."""
    if not path.exists():
        logger.warning(f"{path} absent : reconstruction des chunks depuis le corpus")
        chunks = build_chunks()
        save_chunks(chunks, path)
        return chunks
    with open(path, encoding="utf-8") as f:
        return [Document(page_content=d["text"], metadata=d["metadata"]) for d in map(json.loads, f)]


def ingest_corpus(limit: int | None = None, reset: bool = True, batch_size: int = 128) -> dict:
    """Construit les chunks, calcule les embeddings et remplit la collection Chroma."""
    from app.dependencies import get_vectordb

    corpus = get_corpus()
    chunks = build_chunks(corpus, limit=limit)
    save_chunks(chunks)
    logger.info(f"{len(chunks)} chunks construits à partir de "
                f"{len({c.metadata['article_id'] for c in chunks})} articles")

    vectordb = get_vectordb()
    if reset:
        vectordb.reset_collection()
        logger.info("Collection Chroma réinitialisée")

    start = time.time()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectordb.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
        done = i + len(batch)
        elapsed = time.time() - start
        eta = elapsed / done * (len(chunks) - done)
        logger.info(f"indexé {done}/{len(chunks)} chunks — {elapsed:.0f}s écoulées, ~{eta:.0f}s restantes")

    # l'index BM25 en mémoire doit être reconstruit avec les nouveaux chunks
    from app.rag.hybrid_search import get_bm25_index
    get_bm25_index.cache_clear()

    stats = {"chunks": len(chunks), "articles": len({c.metadata["article_id"] for c in chunks}),
             "seconds": round(time.time() - start, 1), "collection": config.COLLECTION_NAME}
    logger.info(f"Ingestion terminée : {stats}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexe le corpus EUR-Lex dans ChromaDB")
    parser.add_argument("--limit", type=int, default=None, help="nombre max d'articles (test)")
    parser.add_argument("--no-reset", action="store_true", help="ne pas vider la collection avant")
    args = parser.parse_args()
    ingest_corpus(limit=args.limit, reset=not args.no_reset)
