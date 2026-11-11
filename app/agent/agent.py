"""Agent ReAct (LangGraph) : le LLM décide lui-même quels outils juridiques appeler."""
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
# create_react_agent est déprécié dans les versions récentes de LangGraph :
# migrer vers langchain.agents.create_agent lors d'une mise à jour.
from langgraph.prebuilt import create_react_agent

from app import config
from app.agent.tools import find_legal_acts, get_article, search_legal_corpus
from app.llm import get_llm
from app.rag.prompts import AGENT_SYSTEM_PROMPT


def build_agent():
    conn = sqlite3.connect(config.AGENT_DB_PATH, check_same_thread=False)
    return create_react_agent(
        model=get_llm(config.LLM_MODEL),
        tools=[search_legal_corpus, get_article, find_legal_acts],
        prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=SqliteSaver(conn),
    )


def run_agent(agent, question: str, session_id: str = "dev", callbacks: list | None = None) -> str:
    config_ = {"configurable": {"thread_id": session_id}}
    if callbacks:
        config_["callbacks"] = callbacks
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config_)
    return result["messages"][-1].content


if __name__ == "__main__":
    agent = build_agent()
    for q in [
        "Que prévoit l'article 33 du RGPD ?",
        "Quels textes du corpus portent sur la cybersécurité ?",
        "Quelles pratiques d'IA sont interdites et quelles amendes sont encourues ?",
    ]:
        print(f"\n{'=' * 60}\nQUESTION : {q}\nRÉPONSE : {run_agent(agent, q)}")
