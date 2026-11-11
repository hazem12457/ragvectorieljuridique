"""Mémoire de conversation par session (en mémoire du processus).

Correction par rapport à la version finance : la question était enregistrée APRÈS la
réponse, et via `add_messages(str)` (qui attend une liste de messages) -> historique
dans le mauvais ordre. On enregistre maintenant question puis réponse.
"""
from langchain_core.chat_history import InMemoryChatMessageHistory

from app.monitoring.metrics import SESSIONS_CREATED

conversations: dict[str, InMemoryChatMessageHistory] = {}


def get_or_create_memory(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in conversations:
        conversations[session_id] = InMemoryChatMessageHistory()
        SESSIONS_CREATED.inc()
    return conversations[session_id]


def get_chat_history_as_string(session_id: str, last_n: int = 6) -> str:
    messages = get_or_create_memory(session_id).messages[-last_n:]
    lines = []
    for msg in messages:
        role = "Utilisateur" if msg.type == "human" else "Assistant"
        lines.append(f"{role} : {msg.content}")
    return "\n".join(lines)


def get_last_user_questions(session_id: str, n: int = 2) -> list[str]:
    msgs = [m.content for m in get_or_create_memory(session_id).messages if m.type == "human"]
    return msgs[-n:]


def condense_question(question: str, chat_history: str) -> str:
    """« Et pour l'AI Act ? » -> « Quel est le plafond des amendes prévues par l'AI Act ? ».
    Sans historique (ou en cas d'erreur), la question est renvoyée telle quelle."""
    if not chat_history:
        return question
    from loguru import logger

    from app.llm import get_fast_llm
    from app.rag.prompts import CONDENSE_PROMPT

    try:
        result = (CONDENSE_PROMPT | get_fast_llm()).invoke({"chat_history": chat_history, "question": question})
        standalone = result.content.strip().splitlines()[0].strip() if result.content.strip() else ""
        return standalone or question
    except Exception as e:
        logger.warning(f"Reformulation de la relance échouée : {e}")
        return question


def save_to_memory(session_id: str, question: str, answer: str) -> None:
    memory = get_or_create_memory(session_id)
    memory.add_user_message(question)
    memory.add_ai_message(answer)


def clear_memory(session_id: str) -> bool:
    return conversations.pop(session_id, None) is not None
