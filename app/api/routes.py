"""API FastAPI du RAG juridique (corpus EUR-Lex : protection des données & numérique)."""
import asyncio
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse
from langfuse import get_client, observe
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import config
from app.dependencies import get_vectordb
from app.rag.corpus import get_corpus
from app.rag.memory import clear_memory, condense_question, get_chat_history_as_string, save_to_memory
from app.rag.pipeline import MODES, generate_answer, retrieve, sources_payload
from app.rag.retriever import build_filter

try:
    from app.rag.evaluate import evaluate_rag  # noqa: F401
    RAGAS_ENABLED = True
except ImportError:
    RAGAS_ENABLED = False

router = APIRouter()
langfuse = get_client()

_agent = None
_supervisor = None
supervisor_histories: dict[str, list] = {}


def get_agent():
    global _agent
    if _agent is None:
        from app.agent.agent import build_agent
        _agent = build_agent()
    return _agent


def get_supervisor():
    global _supervisor
    if _supervisor is None:
        from app.agent.supervisor import build_supervisor_graph
        _supervisor = build_supervisor_graph()
    return _supervisor


def _filters(celex, acte, doc_type, year_min, year_max):
    if acte and not celex:
        celex = get_corpus().resolve_act(acte)
        if not celex:
            raise HTTPException(404, f"Acte « {acte} » introuvable dans le corpus")
    return build_filter(celex=celex, doc_type=doc_type, year_min=year_min, year_max=year_max)


def _check_mode(mode: str):
    if mode not in MODES:
        raise HTTPException(422, f"mode doit être l'un de {MODES}")


async def _ragas(question, answer, chunks):
    if not RAGAS_ENABLED or not chunks:
        return {"faithfulness": None, "answer_relevancy": None}
    from app.rag.evaluate import evaluate_rag
    return await asyncio.get_event_loop().run_in_executor(None, evaluate_rag, question, answer, chunks)


# ---------------------------------------------------------------- santé & corpus
@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.get("/health/ready")
async def health_ready():
    checks = {"vectorstore": False, "chunks_indexed": 0, "groq_key": bool(os.getenv("GROQ_API_KEY")),
              "cohere_key": bool(os.getenv("COHERE_API_KEY"))}
    try:
        checks["chunks_indexed"] = get_vectordb()._collection.count()
        checks["vectorstore"] = checks["chunks_indexed"] > 0
    except Exception as e:
        checks["vectorstore_error"] = str(e)
    ready = checks["vectorstore"] and checks["groq_key"]
    return JSONResponse(status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
                        content={"ready": ready, "checks": checks})


@router.get("/corpus/stats")
async def corpus_stats():
    corpus = get_corpus()
    try:
        n_chunks = get_vectordb()._collection.count()
    except Exception:
        n_chunks = None
    types = {}
    for d in corpus.documents.values():
        types[d.doc_type] = types.get(d.doc_type, 0) + 1
    return {"documents": len(corpus.documents), "articles": len(corpus.articles), "chunks_indexes": n_chunks,
            "types": types, "embedding_model": config.EMBEDDING_MODEL, "llm": config.LLM_MODEL,
            "top_k": config.TOP_K}


@router.get("/documents")
async def list_documents(q: str | None = None):
    corpus = get_corpus()
    docs = corpus.search_documents(q, limit=100) if q else sorted(corpus.documents.values(), key=lambda d: d.date, reverse=True)
    return [{"celex": d.celex, "label": d.label, "type": d.doc_type, "date": d.date,
             "n_articles": d.n_articles, "title": d.title, "url": d.url} for d in docs]


@router.get("/article/{acte}/{numero}")
async def article(acte: str, numero: str):
    corpus = get_corpus()
    art = corpus.get_article(acte, numero)
    if art is None:
        raise HTTPException(404, f"Article {numero} de « {acte} » introuvable")
    doc = corpus.documents.get(art.celex)
    return {"id": art.id, "acte": corpus.doc_label(art.celex), "celex": art.celex, "article": art.num,
            "titre": art.title, "texte": art.text, "url": doc.url if doc else None}


@router.post("/ingest")
async def ingest(background_tasks: BackgroundTasks, limit: int | None = None):
    """(Ré)indexe tout le corpus en tâche de fond (≈ 20–40 min sur CPU)."""
    from app.rag.chunking import ingest_corpus
    background_tasks.add_task(ingest_corpus, limit=limit)
    return {"message": "Indexation lancée en tâche de fond. Suivre les logs du serveur."}


# ---------------------------------------------------------------- RAG
@router.post("/retrieve")
async def retrieve_only(question: str, mode: str = "vector", k: int = Query(config.TOP_K, ge=1, le=30),
                        celex: str | None = None, acte: str | None = None, doc_type: str | None = None,
                        year_min: int | None = None, year_max: int | None = None):
    """Recherche seule (sans LLM) : utile pour déboguer et pour l'évaluation Recall@k / MRR."""
    _check_mode(mode)
    chunks = retrieve(question, mode=mode, k=k, filters=_filters(celex, acte, doc_type, year_min, year_max))
    return {"question": question, "mode": mode, "sources": sources_payload(chunks)}


@router.post("/ask")
async def ask(question: str, mode: str = "vector", k: int = Query(config.TOP_K, ge=1, le=30),
              celex: str | None = None, acte: str | None = None, doc_type: str | None = None,
              year_min: int | None = None, year_max: int | None = None):
    """RAG vectoriel. mode = vector (baseline) | hybrid | advanced. Filtres optionnels :
    acte ('RGPD', '2016/679'…) ou celex, doc_type (REG/DIR/DEC), year_min, year_max."""
    _check_mode(mode)
    chunks = retrieve(question, mode=mode, k=k, filters=_filters(celex, acte, doc_type, year_min, year_max))
    answer = generate_answer(question, chunks)
    return {"question": question, "mode": mode, "answer": answer, "sources": sources_payload(chunks)}


@router.post("/ask/advanced")
async def ask_advanced(question: str, k: int = Query(config.TOP_K, ge=1, le=30)):
    """MultiQuery + HyDE + BM25 + RRF + reranking Cohere."""
    from app.rag.reranker import rerank
    from app.rag.retriever import advanced_retrieval

    candidates = advanced_retrieval(question, get_vectordb(), k=k)
    chunks = rerank(question, candidates, top_k=k)
    answer = generate_answer(question, chunks)
    return {"question": question, "answer": answer, "chunks_before_rerank": len(candidates),
            "chunks_after_rerank": len(chunks), "sources": sources_payload(chunks)}


@router.post("/ask/filtered")
async def ask_filtered(question: str, acte: str | None = None, doc_type: str | None = None,
                       year_min: int | None = None, year_max: int | None = None, mode: str = "advanced"):
    """Compatibilité avec l'ancienne route (company/year -> acte/type/années)."""
    return await ask(question, mode=mode, k=config.TOP_K, celex=None, acte=acte, doc_type=doc_type,
                     year_min=year_min, year_max=year_max)


@router.post("/evaluate")
async def evaluate(question: str, mode: str = "vector"):
    _check_mode(mode)
    chunks = retrieve(question, mode=mode)
    answer = generate_answer(question, chunks)
    return {"question": question, "mode": mode, "answer": answer, "scores": await _ragas(question, answer, chunks),
            "sources": sources_payload(chunks)}


@router.post("/extract")
async def extract(question: str, acte: str | None = None, mode: str = "hybrid"):
    """Fiche structurée (JSON) de la disposition : destinataires, obligations, délais, sanctions…"""
    from app.rag.extraction import extract_legal_provision
    from app.rag.pipeline import format_context

    _check_mode(mode)
    chunks = retrieve(question, mode=mode, k=6, filters=_filters(None, acte, None, None, None))
    data = extract_legal_provision(format_context(chunks))
    return {"question": question, "extracted_data": data.model_dump(), "sources": sources_payload(chunks)}


@router.post("/chat/memory")
@observe()
async def chat(question: str, session_id: str = "default", mode: str = "advanced",
               k: int = Query(config.TOP_K, ge=1, le=30), celex: str | None = None, acte: str | None = None,
               evaluate: bool = False):
    """RAG conversationnel. Une relance (« Et pour l'AI Act ? ») est d'abord reformulée par le
    LLM en question autonome, qui sert à la recherche ; la réponse tient compte de l'historique."""
    _check_mode(mode)
    filters = _filters(celex, acte, None, None, None)
    chat_history = get_chat_history_as_string(session_id)
    search_query = condense_question(question, chat_history)

    with langfuse.start_as_current_observation(as_type="span", name="retrieval") as span:
        chunks = retrieve(search_query, mode=mode, k=k, filters=filters)
        span.update(output={"chunks": len(chunks)})
    with langfuse.start_as_current_observation(as_type="span", name="generation") as span:
        answer = generate_answer(question, chunks, chat_history=chat_history)
        span.update(output={"answer": answer})

    save_to_memory(session_id, question, answer)

    scores = {"faithfulness": None, "answer_relevancy": None}
    if evaluate:
        scores = await _ragas(question, answer, chunks)
        for name in ("faithfulness", "answer_relevancy"):
            if scores.get(name) is not None:
                langfuse.score_current_trace(name=name, value=scores[name])

    return {"session_id": session_id, "question": question, "search_query": search_query, "mode": mode,
            "answer": answer, "scores": scores, "sources": sources_payload(chunks)}


@router.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    return {"cleared": clear_memory(session_id), "session_id": session_id}


# ---------------------------------------------------------------- agents
@router.post("/agent/graph/ask")
async def agent_graph_ask(question: str):
    from app.agent.graph import build_router_graph
    result = build_router_graph(get_vectordb()).invoke({"question": question})
    return {"question": question, "question_type": result["question_type"], "answer": result["answer"],
            "sources": sources_payload(result["chunks"] or [])}


@router.post("/agent/ask")
@observe()
async def agent_ask(question: str, session_id: str = "default", evaluate: bool = False):
    from langfuse.langchain import CallbackHandler

    from app.agent.agent import run_agent
    from app.agent.tools import get_last_chunks

    try:
        answer = run_agent(get_agent(), question, session_id=session_id, callbacks=[CallbackHandler()])
    finally:
        langfuse.flush()
    chunks = get_last_chunks()
    scores = await _ragas(question, answer, chunks) if evaluate else {"faithfulness": None, "answer_relevancy": None}
    return {"question": question, "answer": answer, "chunks_used": len(chunks), "scores": scores,
            "sources": sources_payload([c for c in chunks if c.metadata.get("article_id")])}


@router.post("/agent/graph/supervisor")
@observe()
async def supervisor_ask(question: str, session_id: str = "default"):
    from langfuse.langchain import CallbackHandler

    from app.agent.supervisor import initial_state

    try:
        result = get_supervisor().invoke(initial_state(question, supervisor_histories.get(session_id)),
                                         config={"callbacks": [CallbackHandler()]})
        supervisor_histories[session_id] = result.get("history", [])
        return {"question": question, "answer": result["final_answer"], "iterations": result["iterations"],
                "plan": result["plan"], "session_id": session_id}
    finally:
        langfuse.flush()


@router.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
