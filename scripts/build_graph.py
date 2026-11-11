"""Construction du Knowledge Graph juridique dans Neo4j.

Schéma :
    (:Document {celex, title, short_name, date, year, type})-[:HAS_ARTICLE]->(:Article {id, celex, num, title, embedding})
    (:Document)-[:CITES|AMENDS|REPEALS|BASED_ON]->(:Document)
    (:Article)-[:REFERS_TO]->(:Article)      renvois extraits par regex (cibles présentes dans le corpus)
    (:Article)-[:NEXT]->(:Article)           ordre des articles dans un même acte
    (:Document)-[:HAS_SUBJECT]->(:EuroVoc {label})

Les embeddings des articles ne sont PAS recalculés : on reprend ceux des chunks de l'index
Chroma (même modèle e5 que le RAG vectoriel) et on fait la moyenne par article, normalisée.

Usage (depuis la racine du projet, environnement rag-juridique) :
    python -m scripts.build_graph            # charge le graphe (idempotent : MERGE)
    python -m scripts.build_graph --reset    # vide la base avant de charger
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np
from loguru import logger
from neo4j import GraphDatabase

from app import config
from app.rag.corpus import get_corpus

BATCH = 500
EMBEDDING_DIM = 768  # multilingual-e5-base


def batched(rows, n=BATCH):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def run_batches(session, query: str, rows: list[dict], label: str):
    for batch in batched(rows):
        session.run(query, rows=batch).consume()
    logger.info(f"{label} : {len(rows)}")


def article_embeddings() -> dict[str, list[float]]:
    """Moyenne normalisée des embeddings Chroma des chunks de chaque article."""
    import chromadb

    client = chromadb.PersistentClient(path=config.VECTORSTORE_DIR)
    col = client.get_collection(config.COLLECTION_NAME)
    data = col.get(include=["embeddings", "metadatas"])
    by_article = defaultdict(list)
    for emb, meta in zip(data["embeddings"], data["metadatas"]):
        if meta and meta.get("article_id"):
            by_article[meta["article_id"]].append(emb)
    out = {}
    for aid, vecs in by_article.items():
        v = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
        out[aid] = (v / (np.linalg.norm(v) or 1.0)).tolist()
    logger.info(f"Embeddings : {len(data['ids'])} chunks -> {len(out)} articles")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="supprime tout le contenu de la base avant import")
    args = ap.parse_args()
    if not config.NEO4J_PASSWORD:
        raise SystemExit("NEO4J_PASSWORD manquant dans .env")

    corpus = get_corpus()
    embeddings = article_embeddings()

    documents = [{
        "celex": d.celex, "title": d.title, "short_name": d.label, "date": d.date,
        "year": d.year, "type": d.doc_type, "eurovoc": d.eurovoc,
    } for d in corpus.documents.values()]

    articles, next_edges, previous = [], [], {}
    for a in corpus.articles.values():  # ordre du fichier = ordre dans l'acte
        articles.append({"id": a.id, "celex": a.celex, "num": a.num, "title": a.title,
                         "embedding": embeddings.get(a.id)})
        if a.celex in previous:
            next_edges.append({"src": previous[a.celex], "dst": a.id})
        previous[a.celex] = a.id

    with open(config.RELATIONS_PATH, encoding="utf-8") as f:
        doc_rels = defaultdict(list)
        for r in csv.DictReader(f):
            doc_rels[r["type"]].append({"src": r["source"], "dst": r["target"]})

    with open(config.ARTICLE_REFS_PATH, encoding="utf-8") as f:
        refs = [{"src": r["source_article"], "dst": r["target_article"]} for r in csv.DictReader(f)
                if r["target_in_corpus"] == "True" and r["target_article"] in corpus.articles]

    missing = sum(1 for a in articles if a["embedding"] is None)
    if missing:
        logger.warning(f"{missing} articles sans embedding (absents de l'index Chroma)")

    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    with driver, driver.session(database=config.NEO4J_DATABASE) as s:
        if args.reset:
            s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS").consume()
            s.run("DROP INDEX article_emb IF EXISTS").consume()
            logger.info("Base vidée")

        for stmt in (
            "CREATE CONSTRAINT doc_celex IF NOT EXISTS FOR (d:Document) REQUIRE d.celex IS UNIQUE",
            "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT eurovoc_label IF NOT EXISTS FOR (e:EuroVoc) REQUIRE e.label IS UNIQUE",
        ):
            s.run(stmt).consume()

        run_batches(s, """
            UNWIND $rows AS row
            MERGE (d:Document {celex: row.celex})
            SET d.title = row.title, d.short_name = row.short_name, d.date = row.date,
                d.year = row.year, d.type = row.type
            WITH d, row
            UNWIND row.eurovoc AS label
            MERGE (e:EuroVoc {label: label})
            MERGE (d)-[:HAS_SUBJECT]->(e)""", documents, "Documents")

        run_batches(s, """
            UNWIND $rows AS row
            MATCH (d:Document {celex: row.celex})
            MERGE (a:Article {id: row.id})
            SET a.celex = row.celex, a.num = row.num, a.title = row.title, a.embedding = row.embedding
            MERGE (d)-[:HAS_ARTICLE]->(a)""", articles, "Articles")

        run_batches(s, """
            UNWIND $rows AS row
            MATCH (a:Article {id: row.src}), (b:Article {id: row.dst})
            MERGE (a)-[:NEXT]->(b)""", next_edges, "NEXT")

        for rel_type, rows in doc_rels.items():
            if rel_type not in {"CITES", "AMENDS", "REPEALS", "BASED_ON"}:
                continue
            run_batches(s, f"""
                UNWIND $rows AS row
                MATCH (a:Document {{celex: row.src}}), (b:Document {{celex: row.dst}})
                MERGE (a)-[:{rel_type}]->(b)""", rows, rel_type)

        run_batches(s, """
            UNWIND $rows AS row
            MATCH (a:Article {id: row.src}), (b:Article {id: row.dst})
            MERGE (a)-[:REFERS_TO]->(b)""", refs, "REFERS_TO")

        s.run(f"""
            CREATE VECTOR INDEX article_emb IF NOT EXISTS FOR (a:Article) ON a.embedding
            OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBEDDING_DIM},
                                     `vector.similarity_function`: 'cosine'}}}}""").consume()

        # --- vérifications ---
        logger.info("Nœuds : " + ", ".join(
            f"{r['label']}={r['n']}" for r in s.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label")))
        logger.info("Relations : " + ", ".join(
            f"{r['t']}={r['n']}" for r in s.run(
                "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY t")))
        isolated = s.run("""MATCH (a:Article) WHERE NOT (a)-[:REFERS_TO]-()
                            RETURN count(a) AS n""").single()["n"]
        logger.info(f"Articles sans aucun renvoi (REFERS_TO) : {isolated}")
        ok = s.run("""MATCH (:Article {id: '32016R0679_art_9'})-[:REFERS_TO]->(b:Article {id: '32016R0679_art_89'})
                      RETURN count(b) AS n""").single()["n"]
        logger.info("Test RGPD art. 9 -> art. 89 : " + ("OK" if ok else "ÉCHEC"))


if __name__ == "__main__":
    main()
