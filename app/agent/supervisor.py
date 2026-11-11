"""Système multi-agents (superviseur LangGraph).

Agents finance (retrieval / market / calculation) remplacés par :
    retrieval : recherche dans les articles (search_legal_corpus)
    article   : texte exact d'un article cité (get_article)
    catalogue : actes du corpus sur un thème (find_legal_acts)
"""
import re
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app import config
from app.agent.tools import find_legal_acts, get_article, get_last_chunks, search_legal_corpus
from app.llm import get_fast_llm, get_llm, record_tokens
from app.monitoring.metrics import AGENT_ITERATIONS
from app.rag.prompts import ARTICLE_REF_PROMPT, SUPERVISOR_PROMPT, SYNTHESIZER_PROMPT

AGENTS = ("retrieval", "article", "catalogue")


class SupervisorState(TypedDict):
    question: str
    plan: Optional[str]
    agents_to_call: Optional[list[str]]
    retrieval_output: Optional[str]
    article_output: Optional[str]
    catalogue_output: Optional[str]
    retrieval_chunks: Optional[list]
    final_answer: Optional[str]
    history: Optional[list[dict]]
    iterations: int
    max_iterations: int


def initial_state(question: str, history: list | None = None, max_iterations: int = 3) -> SupervisorState:
    return {
        "question": question, "plan": None, "agents_to_call": None,
        "retrieval_output": None, "article_output": None, "catalogue_output": None,
        "retrieval_chunks": None, "final_answer": None, "history": history or [],
        "iterations": 0, "max_iterations": max_iterations,
    }


def _outputs_context(state: SupervisorState) -> str:
    parts = []
    if state.get("retrieval_output"):
        parts.append(f"Agent retrieval :\n{state['retrieval_output']}")
    if state.get("article_output"):
        parts.append(f"Agent article :\n{state['article_output']}")
    if state.get("catalogue_output"):
        parts.append(f"Agent catalogue :\n{state['catalogue_output']}")
    return "\n\n".join(parts)


def supervisor_node(state: SupervisorState) -> dict:
    context = f"Question : {state['question']}\nItérations : {state['iterations']}\n\n"
    for turn in (state.get("history") or [])[-3:]:
        context += f"Question précédente : {turn['question']}\nRéponse : {turn['final_answer']}\n"
    context += _outputs_context(state)

    llm = get_llm(config.LLM_MODEL)
    response = llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=context)])
    record_tokens(response, config.LLM_MODEL, "supervisor_routing")

    raw = response.content.strip()
    next_agent = "finish"
    for line in raw.splitlines():
        if line.strip().upper().startswith("NEXT:"):
            next_agent = line.split(":", 1)[1].strip().lower()
    return {"plan": raw, "agents_to_call": [next_agent], "iterations": state["iterations"] + 1}


def retrieval_agent(state: SupervisorState) -> dict:
    result = search_legal_corpus.invoke(state["question"])
    return {"retrieval_output": result, "retrieval_chunks": get_last_chunks()}


def article_agent(state: SupervisorState) -> dict:
    llm = get_fast_llm()
    response = llm.invoke([SystemMessage(content=ARTICLE_REF_PROMPT), HumanMessage(content=state["question"])])
    record_tokens(response, config.FAST_LLM_MODEL, "article_ref_extraction")
    acte = re.search(r"ACTE:\s*(.+)", response.content)
    num = re.search(r"ARTICLE:\s*([\w ]+)", response.content)
    if not acte or acte.group(1).strip().upper() == "NONE" or not num:
        return {"article_output": "Aucune référence d'article identifiable dans la question."}
    return {"article_output": get_article.invoke({"acte": acte.group(1).strip(), "numero": num.group(1).strip()})}


def catalogue_agent(state: SupervisorState) -> dict:
    return {"catalogue_output": find_legal_acts.invoke(state["question"])}


def synthesizer_node(state: SupervisorState) -> dict:
    context = f"Question : {state['question']}\n\n{_outputs_context(state)}"
    history = state.get("history") or []
    if history:
        context += "\n\nConversation précédente :\n" + "\n".join(
            f"Q : {t['question']}\nR : {t['final_answer']}" for t in history[-3:])

    llm = get_llm(config.FAST_LLM_MODEL)
    response = llm.invoke([SystemMessage(content=SYNTHESIZER_PROMPT), HumanMessage(content=context)])
    record_tokens(response, config.FAST_LLM_MODEL, "synthesizer")
    answer = response.content.strip()
    AGENT_ITERATIONS.observe(state["iterations"])
    return {"final_answer": answer, "history": history + [{"question": state["question"], "final_answer": answer}]}


def route_after_supervisor(state: SupervisorState) -> str:
    nxt = (state["agents_to_call"] or ["finish"])[0]
    if nxt == "finish" or state["iterations"] >= state["max_iterations"]:
        return "synthesizer"
    return nxt if nxt in AGENTS else "synthesizer"


def build_supervisor_graph():
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("article", article_agent)
    graph.add_node("catalogue", catalogue_agent)
    graph.add_node("synthesizer", synthesizer_node)
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor,
                                {a: a for a in AGENTS} | {"synthesizer": "synthesizer"})
    for a in AGENTS:
        graph.add_edge(a, "supervisor")
    graph.add_edge("synthesizer", END)
    return graph.compile()
