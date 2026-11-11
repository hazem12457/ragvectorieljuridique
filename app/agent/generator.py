"""Génération adaptée au type de question (utilisée par le routeur LangGraph).

Base = ANSWER_TEMPLATE commun, complété par une consigne propre au type de question.
(Pour la comparaison RAG vectoriel / Graph RAG, on utilise pipeline.generate_answer
avec le prompt commun, sans consigne spécifique.)
"""
from langchain_core.prompts import ChatPromptTemplate

from app.rag.pipeline import generate_answer as base_generate
from app.rag.prompts import ANSWER_TEMPLATE, DOMAIN, TYPE_INSTRUCTIONS

PROMPTS = {
    qtype: ChatPromptTemplate.from_template(
        ANSWER_TEMPLATE.replace("Réponse :", f"Consigne supplémentaire : {instruction}\n\nRéponse :")
    ).partial(domain=DOMAIN)
    for qtype, instruction in TYPE_INSTRUCTIONS.items()
}


def generate_answer(question: str, chunks: list, question_type: str) -> str:
    return base_generate(question, chunks, prompt=PROMPTS.get(question_type, PROMPTS["factuelle"]))
