from app.llm import get_fast_llm
from app.rag.prompts import CLASSIFY_PROMPT, QUESTION_TYPES


def classify_question(question: str) -> str:
    """factuelle | relationnelle | multi_hop (mêmes types que le jeu d'évaluation)."""
    try:
        result = (CLASSIFY_PROMPT | get_fast_llm()).invoke({"question": question})
        label = result.content.strip().lower().replace("-", "_").strip(" .")
        return label if label in QUESTION_TYPES else "factuelle"
    except Exception:
        return "factuelle"
