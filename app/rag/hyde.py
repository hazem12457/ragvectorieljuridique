"""HyDE : on fait rédiger au LLM un « faux article de loi » qui répondrait à la question,
puis on cherche avec ce texte. Un passage rédigé en style législatif ressemble bien plus
aux articles du corpus que la question elle-même (« Dans quel délai… ? »)."""
from loguru import logger

from app import config
from app.llm import get_fast_llm, record_tokens
from app.rag.prompts import HYDE_PROMPT


def hyde_answer(question: str) -> str:
    try:
        result = (HYDE_PROMPT | get_fast_llm()).invoke({"question": question})
        record_tokens(result, config.FAST_LLM_MODEL, "hyde")
        return result.content or question
    except Exception as e:
        logger.warning(f"HyDE échoué, repli sur la question brute. Erreur : {e}")
        return question
