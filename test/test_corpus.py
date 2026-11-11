"""Tests unitaires hors ligne (aucune clé API, aucun modèle) :
    python -m pytest test/test_corpus.py -q
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app.rag.chunking import build_chunks
from app.rag.corpus import get_corpus, short_act_name
from app.rag.eval_dataset import EVAL_QUESTIONS
from app.rag.hybrid_search import BM25Index, matches_filter, reciprocal_rank_fusion, tokenize
from app.rag.retrieval_metrics import recall_at_k, reciprocal_rank
from app.rag.retriever import build_filter


@pytest.fixture(scope="module")
def corpus():
    return get_corpus()


@pytest.fixture(scope="module")
def chunks(corpus):
    return build_chunks(corpus)


@pytest.mark.parametrize("ref,celex", [
    ("RGPD", "32016R0679"), ("gdpr", "32016R0679"), ("2016/679", "32016R0679"),
    ("AI Act", "32024R1689"), ("NIS2", "32022L2555"), ("article 23 de NIS2", "32022L2555"),
    ("directive 2002/58/CE", "32002L0058"), ("93/13", "31993L0013"), ("DMA", "32022R1925"),
    ("32022R2065", "32022R2065"),
])
def test_resolve_act(corpus, ref, celex):
    assert corpus.resolve_act(ref) == celex


def test_get_article(corpus):
    art = corpus.get_article("RGPD", "article 33")
    assert art is not None and "72 heures" in art.text


def test_short_name():
    assert short_act_name("Règlement (CE) n o  765/2008 du Parlement européen") == "Règlement (CE) n° 765/2008"
    assert short_act_name("Directive 93/13/CEE du Conseil, du 5 avril 1993") == "Directive 93/13/CEE"


def test_chunks_have_header_and_size(chunks):
    from app import config
    assert len(chunks) > 3941
    for c in chunks[:500]:
        assert " — Article " in c.page_content.split("\n", 1)[0]
        assert "(?=" not in c.page_content          # pas de regex recollée dans le texte
    assert max(len(c.page_content.split("\n", 1)[1]) for c in chunks) <= config.CHUNK_SIZE + 50
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_eval_gold_articles_exist(corpus):
    for q in EVAL_QUESTIONS:
        for aid in q["gold"]:
            assert aid in corpus.articles, f"{q['id']} : {aid} absent du corpus"


def test_bm25_finds_exact_terms(chunks):
    index = BM25Index(chunks)
    hits = index.search("notification violation données autorité de contrôle 72 heures RGPD", k=5)
    assert "32016R0679_art_33" in [d.metadata["article_id"] for d, _ in hits]


def test_tokenize_keeps_act_numbers():
    assert "2016/679" in tokenize("le règlement (UE) 2016/679")
    assert "effacement" in tokenize("Droit à l'effacement")


def test_filters():
    f = build_filter(celex="32016R0679", year_min=2015)
    assert matches_filter({"celex": "32016R0679", "year": 2016}, f)
    assert not matches_filter({"celex": "32016R0679", "year": 2010}, f)
    assert build_filter() is None


def test_rrf_and_metrics():
    from langchain_core.documents import Document
    a, b, c = (Document(page_content=x, metadata={"chunk_id": x}) for x in "abc")
    fused = reciprocal_rank_fusion([[a, b], [b, c]])
    assert fused[0].metadata["chunk_id"] == "b"
    assert recall_at_k(["x", "y"], ["y", "z"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["y"]) == 0.5
