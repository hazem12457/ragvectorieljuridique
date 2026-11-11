"""Fabrique de LLM (Groq) et comptage des tokens.

Les clients sont créés à la demande (et mis en cache) : importer un module
n'exige donc pas que GROQ_API_KEY soit définie.

Les modèles gpt-oss sont des modèles « à raisonnement » : ils consomment des tokens
à réfléchir avant de répondre. `reasoning_effort="low"` réduit coût et latence pour
les tâches simples (reformulation, HyDE, classification, juge RAGAS).
"""
from functools import lru_cache

from app import config
from app.monitoring.metrics import TOKEN_USAGE


@lru_cache(maxsize=16)
def get_llm(model: str | None = None, temperature: float = 0.0, reasoning_effort: str | None = None,
            max_tokens: int | None = None):
    from langchain_groq import ChatGroq

    kwargs = {"model": model or config.LLM_MODEL, "temperature": temperature}
    if reasoning_effort and "gpt-oss" in kwargs["model"]:
        kwargs["reasoning_effort"] = reasoning_effort
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatGroq(**kwargs)


def get_fast_llm():
    return get_llm(config.FAST_LLM_MODEL, reasoning_effort="low")


def record_tokens(response, model: str, task: str) -> None:
    usage = getattr(response, "usage_metadata", None)
    if usage and usage.get("total_tokens"):
        TOKEN_USAGE.labels(model=model, task=task).inc(usage["total_tokens"])
