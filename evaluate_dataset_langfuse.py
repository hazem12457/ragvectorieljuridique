"""Expérience LangFuse : l'agent juridique est évalué sur le jeu de questions (dataset LangFuse).
Nécessite LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST et GROQ_API_KEY.
    python evaluate_dataset_langfuse.py
"""
import time

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client

from app.agent.agent import build_agent, run_agent
from app.agent.tools import get_last_chunks
from app.rag.eval_dataset import EVAL_QUESTIONS
from app.rag.evaluate import evaluate_rag

langfuse = get_client()
DATASET_NAME = "rag-juridique-eval-v1"


def create_dataset():
    try:
        langfuse.get_dataset(DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' déjà présent")
    except Exception:
        langfuse.create_dataset(name=DATASET_NAME)
        for item in EVAL_QUESTIONS:
            langfuse.create_dataset_item(
                dataset_name=DATASET_NAME,
                input={"question": item["question"]},
                expected_output={"answer": item["reference"], "gold_articles": item["gold"]},
                metadata={"id": item["id"], "type": item["type"]},
            )
        print(f"Dataset '{DATASET_NAME}' créé ({len(EVAL_QUESTIONS)} questions)")


def make_task(agent):
    def task(*, item, **kwargs):
        question = item.input["question"]
        answer = run_agent(agent, question, session_id=f"eval-{time.time()}")
        chunks = get_last_chunks()
        if chunks:
            scores = evaluate_rag(question, answer, chunks, reference=item.expected_output.get("answer"))
            for name, value in scores.items():
                if value is not None:
                    langfuse.score_current_trace(name=name, value=value)
            print(f"  {question[:70]}… -> {scores}")
        time.sleep(15)  # quota Groq gratuit
        return answer
    return task


def run_experiment(name: str):
    dataset = langfuse.get_dataset(DATASET_NAME)
    dataset.run_experiment(name=name, task=make_task(build_agent()), max_concurrency=1,
                           metadata={"corpus": "EUR-Lex numérique (100 actes)", "agent": "ReAct 3 outils"})
    langfuse.flush()
    print(f"Terminé : LangFuse → Datasets → {DATASET_NAME} → Runs")


if __name__ == "__main__":
    create_dataset()
    run_experiment("agent-juridique-v1")
