"""Évaluation RAGAS d'une réponse (juge : LLM Groq).

    faithfulness      : la réponse est-elle fidèle aux articles récupérés (pas d'hallucination) ?
    answer_relevancy  : la réponse traite-t-elle bien la question ?
    context_recall    : (si une réponse de référence est fournie) le contexte contient-il
                        les éléments nécessaires à la réponse attendue ?

Note : answer_relevancy tourne avec strictness=1 car l'API Groq n'accepte pas n>1.
"""
import math
import warnings

from loguru import logger

from app import config

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _safe(val):
    try:
        v = float(val[0]) if isinstance(val, list) else float(val)
        return round(v, 3) if not (math.isnan(v) or math.isinf(v)) else None
    except Exception:
        return None


def evaluate_rag(question: str, answer: str, contexts: list, reference: str | None = None) -> dict:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, LLMContextRecall, faithfulness

    from app.llm import get_llm
    from app.rag.embeddings import get_embeddings

    # raisonnement « low » + 8 000 tokens : sinon le juge gpt-oss épuise son budget à réfléchir
    # (LLMDidNotFinishException) sur les contextes juridiques longs.
    llm = LangchainLLMWrapper(get_llm(config.JUDGE_LLM_MODEL, reasoning_effort="low", max_tokens=8000))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    texts = [c.page_content if hasattr(c, "page_content") else str(c) for c in contexts]
    sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=texts,
                              reference=reference or None)
    metrics = [faithfulness, AnswerRelevancy(strictness=1)]
    if reference:
        metrics.append(LLMContextRecall())

    try:
        result = evaluate(dataset=EvaluationDataset(samples=[sample]), metrics=metrics,
                          llm=llm, embeddings=embeddings, show_progress=False)
    except Exception as e:
        logger.warning(f"RAGAS a échoué : {e}")
        return {"faithfulness": None, "answer_relevancy": None, "context_recall": None}

    scores = result.scores[0] if result.scores else {}
    return {
        "faithfulness": _safe(scores.get("faithfulness")),
        "answer_relevancy": _safe(scores.get("answer_relevancy")),
        "context_recall": _safe(scores.get("context_recall")) if reference else None,
    }
