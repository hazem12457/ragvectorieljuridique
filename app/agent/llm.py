"""Conservé pour compatibilité : utiliser de préférence app.llm.get_llm()."""
from app import config
from app.llm import get_llm


def groq_llm():
    return get_llm(config.LLM_MODEL)
