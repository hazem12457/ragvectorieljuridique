"""Métriques de recherche, calculées au niveau de l'ARTICLE (pas du chunk).

Un article long est découpé en plusieurs chunks : on dédoublonne la liste des chunks
récupérés en liste d'articles (ordre conservé) avant de comparer à la vérité terrain.

    Recall@k = |articles attendus ∩ articles trouvés dans les k premiers chunks| / |articles attendus|
    Hit@k    = 1 si au moins un article attendu est trouvé, sinon 0
    MRR      = 1 / rang du premier article attendu (0 si absent)
"""


def recall_at_k(retrieved: list[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved) & set(gold)) / len(set(gold))


def hit_at_k(retrieved: list[str], gold: list[str]) -> float:
    return 1.0 if set(retrieved) & set(gold) else 0.0


def reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    gold = set(gold)
    for rank, aid in enumerate(retrieved, start=1):
        if aid in gold:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None
