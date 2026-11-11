"""Recherche lexicale BM25 sur tout le corpus + fusion RRF avec la recherche vectorielle.

Pourquoi : la recherche sémantique comprend le sens mais rate les termes exacts.
En droit, ces termes exacts comptent énormément : « article 83 », « 2016/679 »,
« NIS2 », « 72 heures », « contrôleur d'accès »… BM25 les attrape.

Différence avec la version finance : le BM25 n'est plus un simple re-tri des chunks
déjà trouvés par le vectoriel, c'est un vrai second moteur sur les ~8 000 chunks.
"""
import re
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app import config
from app.rag.corpus import normalize

STOPWORDS_FR = set("""
a au aux avec ce ces cet cette dans de des du elle en et eux il ils je la le les leur leurs lui ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous y
est sont etre ete etait ont avoir a ai sera seront fait faire peut peuvent doit doivent
dont lorsque lors ainsi afin entre sans sous tout tous toute toutes autre autres meme
quel quelle quels quelles quoi comment combien pourquoi
""".split())

TOKEN_RE = re.compile(r"\d+/\d+|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(normalize(text).replace("'", " "))
    return [t for t in tokens if t not in STOPWORDS_FR and (len(t) > 1 or t.isdigit())]


def matches_filter(metadata: dict, where: dict | None) -> bool:
    """Évalue un filtre au format Chroma ({"celex": x}, {"year": {"$gte": 2020}}, {"$and": [...]})."""
    if not where:
        return True
    if "$and" in where:
        return all(matches_filter(metadata, w) for w in where["$and"])
    if "$or" in where:
        return any(matches_filter(metadata, w) for w in where["$or"])
    for key, cond in where.items():
        value = metadata.get(key)
        if isinstance(cond, dict):
            for op, ref in cond.items():
                if op == "$eq" and value != ref: return False
                if op == "$ne" and value == ref: return False
                if op == "$gte" and not (value is not None and value >= ref): return False
                if op == "$lte" and not (value is not None and value <= ref): return False
                if op == "$gt" and not (value is not None and value > ref): return False
                if op == "$lt" and not (value is not None and value < ref): return False
                if op == "$in" and value not in ref: return False
        elif value != cond:
            return False
    return True


class BM25Index:
    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])

    def search(self, query: str, k: int = 10, filters: dict | None = None) -> list[tuple[Document, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1]
        results = []
        for idx in order:
            if scores[idx] <= 0:
                break
            chunk = self.chunks[idx]
            if matches_filter(chunk.metadata, filters):
                results.append((chunk, float(scores[idx])))
                if len(results) >= k:
                    break
        return results


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    from app.rag.chunking import load_chunks

    return BM25Index(load_chunks())


def bm25_search(question: str, chunks: list | None = None, top_k: int = 5, filters: dict | None = None) -> list[Document]:
    """Sans `chunks` : recherche sur tout le corpus. Avec `chunks` : re-tri local (ancien comportement)."""
    if chunks is not None:
        if not chunks:
            return []
        local = BM25Okapi([tokenize(c.page_content) for c in chunks])
        scores = local.get_scores(tokenize(question))
        ranked = sorted(zip(scores, range(len(chunks))), key=lambda x: x[0], reverse=True)
        return [chunks[i] for _, i in ranked[:top_k]]
    return [doc for doc, _ in get_bm25_index().search(question, k=top_k, filters=filters)]


def chunk_key(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or str(hash(doc.page_content))


def reciprocal_rank_fusion(result_lists: list[list[Document]], k: int = config.RRF_K,
                           top_n: int | None = None) -> list[Document]:
    """score(d) = Σ 1 / (k + rang_i(d)) — robuste, sans besoin de normaliser les scores."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = chunk_key(doc)
            docs.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores, key=scores.get, reverse=True)
    if top_n:
        ranked = ranked[:top_n]
    return [docs[key] for key in ranked]
