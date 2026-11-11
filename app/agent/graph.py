"""Routeur LangGraph : classe la question puis choisit la stratégie de recherche.

    factuelle      -> recherche hybride, TOP_K chunks (une disposition précise)
    relationnelle  -> recherche avancée + reranking (liens entre actes / articles)
    multi_hop      -> recherche avancée + reranking avec 2×TOP_K chunks (plusieurs articles à combiner)
"""
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from app import config
from app.agent.classifier import classify_question
from app.agent.generator import generate_answer
from app.rag.reranker import rerank
from app.rag.retriever import advanced_retrieval, hybrid_retrieval


class RouterState(TypedDict):
    question: str
    question_type: Optional[str]
    answer: Optional[str]
    chunks: Optional[list]


def build_router_graph(vectordb):
    def classify_node(state: RouterState) -> dict:
        qtype = classify_question(state["question"])
        logger.info(f"[classify] {qtype} : {state['question']}")
        return {"question_type": qtype}

    def route_question(state: RouterState) -> str:
        return f"{state['question_type']}_node"

    def factuelle_node(state: RouterState) -> dict:
        return {"chunks": hybrid_retrieval(state["question"], vectordb, k=config.TOP_K)}

    def relationnelle_node(state: RouterState) -> dict:
        candidates = advanced_retrieval(state["question"], vectordb, k=config.TOP_K)
        return {"chunks": rerank(state["question"], candidates, top_k=config.TOP_K)}

    def multi_hop_node(state: RouterState) -> dict:
        candidates = advanced_retrieval(state["question"], vectordb, k=2 * config.TOP_K)
        return {"chunks": rerank(state["question"], candidates, top_k=2 * config.TOP_K)}

    def generate_node(state: RouterState) -> dict:
        return {"answer": generate_answer(state["question"], state["chunks"], state["question_type"])}

    graph = StateGraph(RouterState)
    graph.add_node("classify_node", classify_node)
    graph.add_node("factuelle_node", factuelle_node)
    graph.add_node("relationnelle_node", relationnelle_node)
    graph.add_node("multi_hop_node", multi_hop_node)
    graph.add_node("generate_node", generate_node)

    graph.set_entry_point("classify_node")
    graph.add_conditional_edges("classify_node", route_question, {
        "factuelle_node": "factuelle_node",
        "relationnelle_node": "relationnelle_node",
        "multi_hop_node": "multi_hop_node",
    })
    for node in ("factuelle_node", "relationnelle_node", "multi_hop_node"):
        graph.add_edge(node, "generate_node")
    graph.add_edge("generate_node", END)
    return graph.compile()


if __name__ == "__main__":
    from app.dependencies import get_vectordb

    router = build_router_graph(get_vectordb())
    for q in [
        "Dans quel délai notifier une violation de données selon le RGPD ?",
        "Quelle directive le RGPD abroge-t-il ?",
        "Comparez les délais de notification d'incident du RGPD et de NIS2.",
    ]:
        r = router.invoke({"question": q})
        print(f"\n{'=' * 60}\n[{r['question_type']}] {q}\n{r['answer']}")
