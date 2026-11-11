"""MultiQuery : 3 reformulations de la question (vocabulaire juridique, nom de l'acte)
pour couvrir plus de terrain sémantique. La question originale est toujours conservée."""
from loguru import logger

from app import config
from app.llm import get_fast_llm, record_tokens
from app.rag.prompts import MULTI_QUERY_PROMPT


def generate_queries(question: str) -> list[str]:
    try:
        result = (MULTI_QUERY_PROMPT | get_fast_llm()).invoke({"question": question})
        record_tokens(result, config.FAST_LLM_MODEL, "multi_query")
        variants = [q.strip(" -•\t") for q in result.content.strip().split("\n") if q.strip()]
        return [question] + variants[:3]
    except Exception as e:
        logger.warning(f"MultiQuery échoué, repli sur la question seule. Erreur : {e}")
        return [question]
