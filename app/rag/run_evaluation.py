"""Évaluation du RAG vectoriel sur le jeu de questions juridiques.

1) Recherche (sans LLM, rapide, gratuit) : Recall@k, Hit@k, MRR au niveau article,
   pour chaque mode (vector / hybrid / advanced), global et par type de question.
2) Optionnel (--answers) : génération des réponses + scores RAGAS
   (faithfulness, answer_relevancy, context_recall). Consomme du quota Groq.

Exemples :
    python -m app.rag.run_evaluation                         # vector + hybrid, recherche seule
    python -m app.rag.run_evaluation --modes vector hybrid advanced
    python -m app.rag.run_evaluation --modes vector --answers --sleep 10
Résultats : results/eval_<horodatage>.json  (+ résumé dans la console)
"""
import argparse
import json
import time
from datetime import datetime

from app import config
from app.rag.eval_dataset import EVAL_QUESTIONS
from app.rag.pipeline import article_ids, format_context, generate_answer, retrieve
from app.rag.retrieval_metrics import hit_at_k, mean, recall_at_k, reciprocal_rank


def evaluate_mode(mode: str, k: int, questions: list[dict], with_answers: bool, sleep: float,
                  on_progress=None, retrieve_fn=None, extra_fn=None) -> dict:
    """`retrieve_fn(question, k) -> chunks` : fonction de recherche externe (ex. Graph RAG, autre projet) ;
    par défaut, `pipeline.retrieve` avec `mode`. `extra_fn(item, chunks) -> dict` : colonnes en plus."""
    rows = []
    for i, item in enumerate(questions, start=1):
        t0 = time.time()
        if retrieve_fn is not None:
            chunks = retrieve_fn(item["question"], k)
        else:
            chunks = retrieve(item["question"], mode=mode, k=k)
        latency = time.time() - t0
        found = article_ids(chunks)
        context_chars = len(format_context(chunks))
        row = {
            "id": item["id"], "type": item["type"], "question": item["question"],
            "gold": item["gold"], "retrieved_articles": found,
            "recall": recall_at_k(found, item["gold"]),
            "hit": hit_at_k(found, item["gold"]),
            "rr": reciprocal_rank(found, item["gold"]),
            "retrieval_seconds": round(latency, 2),
            "n_articles": len(found),
            "context_chars": context_chars,
            "context_tokens_approx": round(context_chars / 4),   # ≈ 4 caractères par token
        }
        if extra_fn is not None:
            row.update(extra_fn(item, chunks))
        if with_answers:
            from app.rag.evaluate import evaluate_rag

            row["answer"] = generate_answer(item["question"], chunks)
            row.update(evaluate_rag(item["question"], row["answer"], chunks, reference=item.get("reference")))
            if sleep:
                time.sleep(sleep)   # quota Groq gratuit
        rows.append(row)
        extra = ""
        if with_answers:
            extra = (f"  faith={row.get('faithfulness')}  relev={row.get('answer_relevancy')}"
                     f"  ctx_recall={row.get('context_recall')}")
        print(f"  [{mode}] {i:>2}/{len(questions)} {item['id']}  recall={row['recall']:.2f}  rr={row['rr']:.2f}{extra}")
        if on_progress:
            on_progress({"mode": mode, "k": k, "rows": rows, "summary": summarize(rows)})
    return {"mode": mode, "k": k, "rows": rows, "summary": summarize(rows)}


def summarize(rows: list[dict]) -> dict:
    def agg(subset):
        out = {
            "n": len(subset),
            "recall@k": mean([r["recall"] for r in subset]),
            "hit@k": mean([r["hit"] for r in subset]),
            "mrr": mean([r["rr"] for r in subset]),
            "retrieval_seconds": mean([r["retrieval_seconds"] for r in subset]),
            "n_articles": mean([r.get("n_articles") for r in subset]),
            "context_tokens_approx": mean([r.get("context_tokens_approx") for r in subset]),
        }
        for key in ("faithfulness", "answer_relevancy", "context_recall"):
            if any(key in r for r in subset):
                out[key] = mean([r.get(key) for r in subset])
        return out

    summary = {"global": agg(rows)}
    for qtype in sorted({r["type"] for r in rows}):
        summary[qtype] = agg([r for r in rows if r["type"] == qtype])
    return summary


def print_table(results: list[dict]) -> None:
    cols = ["recall@k", "hit@k", "mrr", "retrieval_seconds", "context_tokens_approx",
            "faithfulness", "answer_relevancy", "context_recall"]
    width = 30 + 13 * len(cols)
    print("\n" + "=" * width)
    print(f"{'mode':<12}{'sous-ensemble':<15}{'n':>3}  " + "".join(f"{c[:12]:>13}" for c in cols))
    print("-" * width)
    for res in results:
        for subset, s in res["summary"].items():
            vals = "".join(f"{(s.get(c) if s.get(c) is not None else '-'):>13}" for c in cols)
            print(f"{res['mode']:<12}{subset:<15}{s['n']:>3}  {vals}")
    print("=" * width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["vector", "hybrid"], choices=["vector", "hybrid", "advanced"])
    parser.add_argument("--k", type=int, default=config.TOP_K, help="nb de chunks récupérés (contexte du LLM)")
    parser.add_argument("--answers", action="store_true", help="générer les réponses + scores RAGAS (Groq)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--types", nargs="+", default=None, help="filtrer : factuelle relationnelle multi_hop")
    parser.add_argument("--sleep", type=float, default=0.0, help="pause entre questions (quota Groq)")
    args = parser.parse_args()

    questions = [q for q in EVAL_QUESTIONS if not args.types or q["type"] in args.types][: args.limit]
    print(f"{len(questions)} questions · k={args.k} · modes={args.modes} · réponses={'oui' if args.answers else 'non'}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"

    def save(results):
        payload = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "embedding_model": config.EMBEDDING_MODEL, "llm": config.LLM_MODEL,
            "judge": config.JUDGE_LLM_MODEL if args.answers else None, "k": args.k,
            "budget": f"{args.k} articles distincts", "reranker": (
                config.LOCAL_RERANK_MODEL if config.RERANKER == "local" else config.RERANK_MODEL),
            "chunk_size": config.CHUNK_SIZE, "results": results,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    results = []
    for m in args.modes:
        # sauvegarde après chaque question : rien n'est perdu si l'évaluation est interrompue
        results.append(evaluate_mode(m, args.k, questions, args.answers, args.sleep,
                                     on_progress=lambda partial: save(results + [partial])))
        save(results)
    print_table(results)
    print(f"Résultats enregistrés : {out}")


if __name__ == "__main__":
    main()
