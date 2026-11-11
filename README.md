# RAG vectoriel juridique — droit européen des données et du numérique

Projet S3 ESPRIT — module *Advanced Learning for Text and Graph Data*.
Volet : **protection des données et numérique** (RGPD, AI Act, DSA, DMA, NIS2, Data Act, DORA, CRA, ePrivacy…).

Ce dépôt contient le **RAG vectoriel** du projet : un assistant de questions-réponses qui répond à partir des
**articles officiels** (EUR-Lex, en français) de 100 actes européens et cite systématiquement l'acte et l'article.
Il sert de **baseline** à comparer avec le **Graph RAG** (Neo4j), qui constitue l'étape suivante.

Le code est issu d'un RAG « finance » (rapports annuels en PDF, Tesla 10-K) entièrement réadapté au domaine juridique :
corpus, découpage, embeddings, prompts, outils des agents, extraction structurée, évaluation et interface.

---

## Sommaire
1. [Ce qui a changé par rapport à la version finance](#1-ce-qui-a-changé-par-rapport-à-la-version-finance)
2. [Architecture](#2-architecture)
3. [Le corpus](#3-le-corpus)
4. [Le pipeline, étape par étape](#4-le-pipeline-étape-par-étape)
5. [Agents](#5-agents)
6. [Évaluation](#6-évaluation)
7. [Installation et lancement](#7-installation-et-lancement)
8. [API](#8-api)
9. [Configuration](#9-configuration)
10. [Structure du projet](#10-structure-du-projet)
11. [Limites connues](#11-limites-connues)
12. [Lien avec le Graph RAG (étape suivante)](#12-lien-avec-le-graph-rag-étape-suivante)

---

## 1. Ce qui a changé par rapport à la version finance

| Élément | Version finance | Version juridique |
|---|---|---|
| Données | 1 PDF uploadé (Tesla 10-K) | 100 actes EUR-Lex, 3 941 articles (`data/eurlex/`) |
| Ingestion | `/upload` d'un PDF, efface la base à chaque upload | `python -m app.rag.chunking` : indexe tout le corpus |
| Unité de découpage | page PDF → chunks de 500 caractères | **article** → 1 chunk, ou sous-chunks de 1 500 car. pour les articles longs |
| Contexte d'un chunk | aucun | en-tête `Acte — Article n — Titre` en tête de chaque chunk |
| Embeddings | `all-MiniLM-L6-v2` (anglais) | **`intfloat/multilingual-e5-base`** (multilingue, préfixes `query:`/`passage:`) |
| Métadonnées / filtres | `company`, `year` | `celex`, `doc_type` (REG/DIR/DEC), `year`, `article_id`, `article_num`, `eurovoc` |
| BM25 | re-tri des chunks déjà trouvés | **vrai second moteur sur tout le corpus** + fusion RRF |
| Prompts | analyste financier (anglais) | juriste spécialisé en droit européen (français), citations [n] obligatoires |
| Types de questions | factual / comparative / analytical | **factuelle / relationnelle / multi_hop** (= types du jeu d'évaluation) |
| Outils de l'agent | recherche, calculatrice, cours de bourse (yfinance) | `search_legal_corpus`, `get_article`, `find_legal_acts` |
| Extraction (Instructor) | `FinancialMetrics` (revenue, net income…) | `LegalProvision` (destinataires, obligations, droits, délais, sanctions, renvois) |
| Évaluation | 10 questions Tesla, RAGAS | 24 questions juridiques vérifiées, **Recall@k, Hit@k, MRR** + RAGAS |
| Interface | upload PDF, anglais | sélection du mode, du k et de l'acte, sources cliquables vers EUR-Lex, français |

Bugs corrigés au passage :
- l'historique de conversation enregistrait la réponse **avant** la question, et la question via `add_messages(str)` (qui attend une liste) ;
- `/ask/advanced` générait la réponse à partir des chunks **non rerankés** ;
- la route `/ask/filetred` (faute de frappe) est devenue `/ask/filtered` ;
- les chemins relatifs (`./app/vectorstore`) dépendaient du dossier de lancement : ils sont maintenant absolus (`app/config.py`) ;
- les clients LLM étaient créés à l'import (crash sans `GROQ_API_KEY`) : ils sont maintenant créés à la demande ;
- RAGAS ne tournait plus à chaque message du chat (coûteux et lent) : il faut le demander avec `evaluate=true`.

---

## 2. Architecture

```
                    ┌──────────────────────── Indexation (une fois) ────────────────────────┐
 data/eurlex/       │  articles.jsonl ─► découpage par article ─► en-tête « Acte — Art. — Titre »│
 (collect_eurlex)   │                  ─► e5 multilingue (passage:) ─► ChromaDB (cosinus)      │
                    │                  ─► data/index/chunks.jsonl ─► index BM25 (en mémoire)   │
                    └───────────────────────────────────────────────────────────────────────┘

 Question ─► Streamlit ─► FastAPI ─► recherche (3 modes) ─► k extraits numérotés ─► LLM Groq ─► réponse citée
                                       │
                                       ├─ vector   : similarité cosinus (baseline)
                                       ├─ hybrid   : vectoriel + BM25, fusion RRF
                                       └─ advanced : MultiQuery (×4) + HyDE + BM25 ─► RRF ─► rerank cross-encoder
```

Deux autres sorties partent du même moteur de recherche :
- **`/extract`** : fiche JSON structurée de la disposition (Instructor + Groq) ;
- **agents** : agent ReAct, routeur LangGraph et superviseur multi-agents.

---

## 3. Le corpus

Produit par `scripts/collect_eurlex.py` (SPARQL Cellar pour les métadonnées et relations, API REST Cellar pour les
textes en français ; le site eur-lex.europa.eu bloque les scripts, on passe par
`http://publications.europa.eu/resource/celex/<CELEX>`).

| Fichier | Contenu |
|---|---|
| `data/eurlex/articles.jsonl` | 3 941 articles : `id` (`<celex>_art_<num>`), `celex`, `doc_title`, `doc_type`, `date`, `article_num`, `article_title`, `text`, `eurovoc` |
| `data/eurlex/documents.csv` | 100 actes : `celex`, `type` (REG/DIR/DEC), `date`, `title`, `eurovoc` (séparés par `\|`), `n_articles` |
| `data/eurlex/relations.csv` | liens acte → acte (CITES, AMENDS, REPEALS, BASED_ON) — **non utilisés par le RAG vectoriel**, réservés au graphe |
| `data/eurlex/article_refs.csv` | renvois article → article — idem, réservés au graphe |

Statistiques : 3 143 articles de règlements, 782 de directives, 16 de décisions. Longueur médiane d'un article
≈ 1 450 caractères, 10 % dépassent 4 400 caractères, le plus long ≈ 219 000 caractères.

`app/rag/corpus.py` charge le corpus et fournit :
- un **nom court** pour chaque acte (`Règlement (UE) 2016/679`) et des **alias** usuels (`RGPD`, `GDPR`, `AI Act`, `NIS2`, `DSA`…), utilisés à l'affichage, dans le texte indexé et pour résoudre les références ;
- `resolve_act("article 23 de NIS2")` → `32022L2555` (par sigle, numéro `2016/679`, `93/13`, ou CELEX) ;
- `get_article("RGPD", "33")` → texte exact de l'article.

---

## 4. Le pipeline, étape par étape

### 4.1 Découpage (`app/rag/chunking.py`)

**L'article est l'unité naturelle d'un texte de loi**, c'est donc lui qui sert de base, et non une page ou un nombre fixe de caractères.
- Un article de 1 500 caractères au plus donne **un seul chunk** (c'est le cas d'environ la moitié des articles).
- Un article plus long est sous-découpé à environ 1 500 caractères avec 200 caractères de chevauchement, en coupant de préférence
  avant un paragraphe numéroté (`2. Le …`), puis avant un point (`a)`, `b)`), une définition (`1)`), une fin de phrase.
- **Chaque chunk commence par un en-tête** : `Règlement (UE) 2016/679 (RGPD) — Article 83 — Conditions générales pour imposer des amendes administratives`.
  Sans lui, un sous-chunk comme « 2. Le paragraphe 1 ne s'applique pas… » serait impossible à rattacher, aussi bien pour la recherche que pour le LLM.
  L'en-tête contient aussi le sigle (`RGPD`), ce qui permet de trouver l'article avec une question du type « … selon le RGPD ».

Résultat : **9 290 chunks** pour 3 941 articles (2 025 articles tiennent en un seul chunk), chacun sous la limite de
512 tokens du modèle e5.

Chaque chunk porte les métadonnées `chunk_id`, `article_id`, `celex`, `doc_label`, `doc_type`, `year`, `date`,
`article_num`, `article_title`, `eurovoc`, `chunk_index` et `n_chunks`. Elles sont sauvegardées dans
`data/index/chunks.jsonl`, qui sert aussi au BM25 et servira au Graph RAG pour relier les chunks aux nœuds `Article`.

### 4.2 Embeddings (`app/rag/embeddings.py`)

`intfloat/multilingual-e5-base` (768 dimensions), exécuté en local sur CPU avec sentence-transformers.
- **Pourquoi pas `all-MiniLM-L6-v2` ?** Il est entraîné uniquement sur de l'anglais, alors que le corpus est en français juridique.
- e5 attend les préfixes `query: ` pour les questions et `passage: ` pour les documents ; ils sont ajoutés automatiquement.
- Les vecteurs sont normalisés et la collection Chroma utilise la distance cosinus.
- Le modèle se change avec la variable `EMBEDDING_MODEL` (par exemple `BAAI/bge-m3`, meilleur mais environ 3 fois plus lent sur CPU) ; il faut alors réindexer.

### 4.3 Index vectoriel

ChromaDB persistant dans `app/vectorstore_juridique/`, collection `eurlex_numerique`. L'indexation complète prend
**environ 20 minutes sur CPU** (Ryzen 7, environ 8 chunks/s).

### 4.4 Recherche : trois modes (`app/rag/retriever.py`, `app/rag/pipeline.py`)

| Mode | Fonctionnement | LLM nécessaire ? |
|---|---|---|
| `vector` | les chunks les plus proches (cosinus) | non |
| `hybrid` | 20 candidats vectoriels + 20 candidats BM25, fusionnés par **Reciprocal Rank Fusion** | non |
| `advanced` | question + 3 reformulations (MultiQuery) + un article hypothétique (HyDE), chacune en vectoriel (10 résultats), + BM25 → RRF → 20 candidats → **reranking cross-encoder** `BAAI/bge-reranker-v2-m3` (local, multilingue) | oui (Groq, pour MultiQuery et HyDE) |

**Règles communes, figées pour la comparaison avec le Graph RAG (7 novembre 2026)** :
- **Budget : k = 5 articles distincts** dans le contexte, chacun représenté par son chunk le mieux classé
  (`pipeline.top_articles`). Avant, le contexte contenait 5 *chunks*, souvent plusieurs morceaux du même article
  (3 fois l'article 33 du RGPD, par exemple), ce qui réduisait le nombre d'articles différents.
- **Même reranker** pour tous les modes qui rerankent (advanced et graph) : cross-encoder local `BAAI/bge-reranker-v2-m3`
  (≈ 5 s pour 20 passages sur CPU). Cohere `rerank-v3.5` reste disponible avec `RERANKER=cohere`. Les modes `vector` et
  `hybrid` ne rerankent pas : ils servent de références.
- **Candidats du mode advanced mis en cache** (`data/cache/advanced_candidates.json`) : MultiQuery et HyDE sont générés par
  le LLM et varient d'une exécution à l'autre. Le cache fige les 20 candidats de chaque question ; le Graph RAG relit les
  mêmes, et une réévaluation ne consomme plus de quota Groq. Supprimer le fichier pour régénérer.
- **Même LLM** (`gpt-oss-120b`, température 0) et **même prompt** pour tous les modes. Toute modification du prompt impose
  de réévaluer tous les modes.

- **BM25** (`app/rag/hybrid_search.py`) : tokenisation française (minuscules, sans accents, mots vides retirés), en conservant les
  numéros d'actes (`2016/679`). Il capte ce que l'embedding rate : « article 83 », « NIS2 », « 72 heures », « contrôleur d'accès ».
- **RRF** : `score(d) = Σ 1/(60 + rang)`. Cette fusion est robuste et ne demande pas de normaliser des scores hétérogènes (cosinus contre BM25).
- **MultiQuery** : les reformulations reprennent le vocabulaire des textes (« responsable du traitement », « entité essentielle »).
- **HyDE** : le LLM rédige un faux article dans le style législatif ; ce passage ressemble davantage aux articles du corpus que la question elle-même.
- **Filtres** (tous les modes) : `celex` ou `acte` (« RGPD »), `doc_type`, `year_min`, `year_max`.

### 4.5 Génération (`app/rag/prompts.py` → `ANSWER_PROMPT`)

Les k extraits (k = 5 par défaut) sont numérotés `[1]…[k]` et passés au LLM (`openai/gpt-oss-120b` via Groq) avec un prompt de
juriste qui impose de :
1. citer chaque affirmation `[n]` en nommant l'acte et l'article ;
2. reprendre fidèlement conditions, exceptions, délais, seuils et montants ;
3. dire explicitement « Les articles fournis ne permettent pas de répondre » plutôt que d'inventer ;
4. désigner un acte cité uniquement par sa référence telle qu'elle figure dans les extraits (« directive (UE) 2020/1828 »),
   sans ajouter d'intitulé ou d'objet tiré de ses propres connaissances ;
5. réserver les guillemets « » aux citations exactes des extraits ;
6. répondre en français, sans conseil juridique personnalisé.

Les règles 4 et 5 ont été ajoutées le 7 novembre 2026 après une hallucination constatée (voir [§11](#11-limites-connues)).

**Ce prompt est partagé avec le futur Graph RAG**, comme le reste de la chaîne de génération (`format_context`, `generate_answer`).

### 4.6 Mémoire de conversation (`app/rag/memory.py`)

Une session = un identifiant. Les 6 derniers messages sont injectés dans le prompt. Pour la **recherche**, une relance est
d'abord **reformulée en question autonome** par le LLM rapide (`CONDENSE_PROMPT`) : après « Quel est le plafond des amendes
du RGPD ? », la relance « Et pour l'AI Act ? » devient « Quel est le plafond des amendes prévues par l'AI Act ? ».
La première version mettait simplement les questions bout à bout : la recherche restait alors bloquée sur le RGPD.
La question reformulée est renvoyée dans le champ `search_query` de `/chat/memory`.

### 4.7 Extraction structurée (`app/rag/extraction.py`, route `/extract`)

Instructor + Groq en `Mode.TOOLS` : la sortie est contrainte au schéma Pydantic `LegalProvision`, qui contient
`acte`, `article`, `objet`, `destinataires`, `obligations`, `droits`, `delais`, `sanctions`, `renvois`, `source_quote`
et `not_found_in_context`. Tout champ absent du contexte reste vide : le modèle n'invente pas.

---

## 5. Agents

| Composant | Fichier | Rôle |
|---|---|---|
| Outils | `app/agent/tools.py` | `search_legal_corpus` (hybride + rerank), `get_article` (texte exact d'un article), `find_legal_acts` (catalogue par thème ou sigle) |
| Agent ReAct | `app/agent/agent.py` | le LLM choisit lui-même les outils ; mémoire SQLite (`agent-memory-juridique.db`) |
| Routeur | `app/agent/graph.py` | classe la question (factuelle / relationnelle / multi_hop) puis choisit la stratégie : hybride k=5, avancé k=5, avancé k=10 |
| Superviseur | `app/agent/supervisor.py` | orchestre 3 agents (`retrieval`, `article`, `catalogue`) puis un synthétiseur |

---

## 6. Évaluation

### Jeu de questions (`app/rag/eval_dataset.py`)

24 questions en français, chacune avec :
- `type` : `factuelle` (14), `relationnelle` (4) ou `multi_hop` (6) ;
- `gold` : identifiants des articles qui contiennent la réponse, **tous vérifiés dans le corpus** (un test le contrôle) ;
- `reference` : la réponse attendue, rédigée à partir du texte officiel (délais, montants et seuils vérifiés).

Ce jeu est à compléter jusqu'à **30 à 40 questions** lors de l'étape « Évaluation », surtout avec des questions relationnelles et
multi-hop, là où le graphe doit faire la différence.

### Métriques

Les métriques de recherche se calculent **au niveau de l'article** : la liste des chunks est dédoublonnée en liste d'articles.
- **Recall@k** = part des articles attendus retrouvés parmi les k chunks ;
- **Hit@k** = au moins un article attendu trouvé ;
- **MRR** = 1 / rang du premier article attendu ;
- **RAGAS** (option `--answers`) : `faithfulness`, `answer_relevancy` et `context_recall` (ce dernier grâce à la réponse de référence).

### Commandes

```bash
# Recherche seule (sans LLM, gratuit) : vector et hybrid
python -m app.rag.run_evaluation --modes vector hybrid

# Avec le mode avancé (Groq, et Cohere s'il y a une clé)
python -m app.rag.run_evaluation --modes vector hybrid advanced

# Réponses + RAGAS (consomme du quota Groq ; --sleep pour le quota gratuit)
python -m app.rag.run_evaluation --modes vector --answers --sleep 10
```
Les résultats sont enregistrés dans `results/eval_<date>.json`, avec le détail par question et les agrégats globaux et par type.

### Résultats de référence (règles figées : 5 articles distincts, reranker cross-encoder, 24 questions)

Mesuré le 7 novembre 2026 (`results/eval_20261107_210814.json`). Recherche seule. **C'est la baseline de la comparaison avec
le Graph RAG** (mode `advanced`).

| Mode | Sous-ensemble | n | Recall@5 | Hit@5 | MRR | Latence (s) | Contexte (≈ tokens) |
|---|---|---|---|---|---|---|---|
| vector | **global** | 24 | 0,896 | 0,917 | 0,719 | 0,4 | 1 395 |
| vector | factuelle | 14 | 0,929 | 0,929 | 0,726 | | |
| vector | relationnelle | 4 | 0,750 | 0,750 | 0,562 | | |
| vector | multi_hop | 6 | 0,917 | 1,000 | 0,806 | | |
| hybrid | **global** | 24 | 0,903 | 0,958 | 0,788 | 0,1 | 1 323 |
| hybrid | factuelle | 14 | 1,000 | 1,000 | 0,744 | | |
| hybrid | relationnelle | 4 | 0,750 | 0,750 | 0,625 | | |
| hybrid | multi_hop | 6 | 0,778 | 1,000 | 1,000 | | |
| advanced | **global** | 24 | **0,917** | **0,958** | **0,896** | 10,9 | 1 460 |
| advanced | factuelle | 14 | 1,000 | 1,000 | 0,929 | | |
| advanced | relationnelle | 4 | 0,750 | 0,750 | 0,750 | | |
| advanced | multi_hop | 6 | 0,833 | 1,000 | 0,917 | | |

Ce qui a changé par rapport à la mesure du 4 novembre : budget en articles distincts (au lieu de chunks), et un vrai
reranking dans le mode `advanced` (le 4 novembre, sans clé Cohere, il n'y en avait pas). Le reranker améliore surtout le
**classement** : MRR global 0,833 → 0,896, factuelles 0,821 → 0,929. La latence du mode `advanced` (≈ 11 s) vient des
appels Groq de MultiQuery / HyDE et du cross-encoder sur CPU ; avec le cache des candidats, une réévaluation ne refait
que le reranking. Les questions qui échouent encore en `advanced` : **R02** (aucun article trouvé), **M04** (manque
l'article 14 du CRA), **M05** (manque un des deux actes).

#### Mesure précédente (4 novembre 2026, 5 chunks, sans reranking)

`results/eval_20261104_223224.json`. Conservée pour mémoire ; elle n'est plus comparable (autre budget).

| Mode | Sous-ensemble | n | Recall@5 | Hit@5 | MRR |
|---|---|---|---|---|---|
| vector | **global** | 24 | 0,868 | 0,917 | 0,719 |
| vector | factuelle | 14 | 0,929 | 0,929 | 0,726 |
| vector | relationnelle | 4 | 0,750 | 0,750 | 0,562 |
| vector | multi_hop | 6 | 0,806 | 1,000 | 0,806 |
| hybrid | **global** | 24 | **0,889** | **0,958** | **0,788** |
| hybrid | factuelle | 14 | 1,000 | 1,000 | 0,744 |
| hybrid | relationnelle | 4 | 0,750 | 0,750 | 0,625 |
| hybrid | multi_hop | 6 | 0,722 | 1,000 | 1,000 |
| advanced | **global** | 24 | **0,924** | **0,958** | **0,833** |
| advanced | factuelle | 14 | 1,000 | 1,000 | 0,821 |
| advanced | relationnelle | 4 | 0,750 | 0,750 | 0,625 |
| advanced | multi_hop | 6 | 0,861 | 1,000 | 1,000 |

**Scores RAGAS** : la chaîne fonctionne (test sur 3 questions factuelles en mode `vector` : faithfulness 1,0,
answer_relevancy 0,85 en moyenne, context_recall 1,0), mais l'évaluation complète est **reportée**. Elle coûte environ
45 s par question avec le juge LLM ; on la lancera pour la comparaison finale baseline vectorielle contre Graph RAG :
`python -m app.rag.run_evaluation --modes vector --answers --sleep 3`.

Lecture des résultats (mesure du 4 novembre) :
- **Chaque technique apporte quelque chose.** Le BM25 (hybrid) fait passer le Recall@5 de 0,868 à 0,889 ; MultiQuery et HyDE
  (advanced) le portent à 0,924, surtout grâce aux questions multi-hop (0,722 → 0,861).
- **Les questions factuelles sont bien couvertes.** L'hybride les retrouve toutes (Recall@5 = 1). Le BM25 récupère par exemple
  F06 (« format structuré… lisible par machine »), que le vectoriel seul rate.
- **Les questions relationnelles sont le point faible attendu.** Sur R02 (« comment le DSA modifie-t-il la directive
  2000/31/CE ? »), **les trois modes** ne renvoient que des articles de la directive 2000/31/CE elle-même, jamais l'article 89 du DSA
  qui opère la modification. Ce lien existe dans `relations.csv` (AMENDS) : c'est exactement le cas que le Graph RAG doit résoudre.
- **Les questions multi-hop** trouvent toujours au moins un bon article (Hit = 1), mais pas tous : M01 rate l'article 33 du RGPD,
  M04 l'article 14 du CRA, M06 un des trois articles du Data Act. Avec 5 extraits pour 2 ou 3 actes, la place manque ;
  l'expansion dans le graphe (articles voisins, renvois) doit aider ici aussi.

---

## 7. Installation et lancement

Prérequis : Anaconda (ou Python 3.11), une clé **Groq** gratuite (https://console.groq.com) et, en option, une clé Cohere.

```powershell
# 1. Environnement
conda create -n rag-juridique python=3.11 -y
conda activate rag-juridique
pip install torch --index-url https://download.pytorch.org/whl/cpu   # torch CPU (≈ 200 Mo au lieu de 2,5 Go)
pip install -r requirements.txt

# 2. Clés
copy .env.example .env      # puis renseigner GROQ_API_KEY (et COHERE_API_KEY en option)

# 3. Indexation du corpus (≈ 20 min sur CPU, une seule fois ; télécharge e5-base ≈ 1 Go au premier lancement)
python -m app.rag.chunking
#    test rapide : python -m app.rag.chunking --limit 200

# 4. Tests hors ligne
python -m pytest test/test_corpus.py -q

# 5. API (http://localhost:8001/docs)
uvicorn app.main:app --port 8001

# 6. Interface (http://localhost:8501)
streamlit run streamlit_app.py
```

> **Windows, Smart App Control :** sur ce poste, la DLL de pandas 3.x est bloquée (« Une stratégie de contrôle
> d'application a bloqué ce fichier »). C'est pourquoi `requirements.txt` impose `pandas<3`.
>
> Smart App Control bloque aussi **parfois** `torch_global_deps.dll` au premier chargement (WinError 4551). Symptôme :
> le log affiche `Warmup ignoré : [WinError 4551]…` et la barre latérale de Streamlit indique 0 chunk. Il suffit de
> relancer `uvicorn` ; vérifier ensuite que le log affiche bien `Warmup terminé : 9290 chunks indexés`.
>
> **Streamlit hors terminal interactif** (script, tâche de fond) : au premier lancement, Streamlit attend une adresse
> e-mail et s'arrête. Lancer avec `streamlit run streamlit_app.py --server.headless true`.
>
> Les routes `/ask`, `/retrieve`, `/chat/memory`… prennent leurs paramètres **dans l'URL** (query string), pas dans un
> corps JSON : `POST /ask?question=...&mode=hybrid&k=5`.

Avec Docker : `docker compose up --build` (backend, Streamlit, Prometheus, Grafana). Il faut indexer avant le lancement
(le dossier `app/vectorstore_juridique` est monté dans le conteneur).

Relancer la collecte du corpus : `python scripts/collect_eurlex.py --out data/eurlex --max-docs 100`.

---

## 8. API

| Route | Méthode | Description |
|---|---|---|
| `/health`, `/health/ready` | GET | santé ; *ready* vérifie l'index et la clé Groq |
| `/corpus/stats` | GET | nombre d'actes, d'articles et de chunks indexés, modèles utilisés |
| `/documents?q=` | GET | liste ou recherche des actes du corpus |
| `/article/{acte}/{numero}` | GET | texte exact d'un article, ex. `/article/RGPD/17` |
| `/ingest` | POST | réindexation en tâche de fond |
| `/retrieve` | POST | **recherche seule** (sans LLM) : `question`, `mode`, `k`, filtres |
| `/ask` | POST | RAG : `question`, `mode` (`vector`, `hybrid` ou `advanced`), `k`, `acte`/`celex`, `doc_type`, `year_min`, `year_max` |
| `/ask/advanced` | POST | pipeline avancé complet (nombre de chunks avant et après rerank) |
| `/ask/filtered` | POST | filtres par acte, type et années |
| `/chat/memory` | POST | RAG conversationnel (`session_id`, `mode`, `k`, `acte`, `evaluate`) |
| `/chat/{session_id}` | DELETE | efface une conversation |
| `/evaluate` | POST | réponse + scores RAGAS |
| `/extract` | POST | fiche JSON `LegalProvision` |
| `/agent/ask` | POST | agent ReAct |
| `/agent/graph/ask` | POST | routeur LangGraph (type de question puis stratégie) |
| `/agent/graph/supervisor` | POST | superviseur multi-agents |
| `/metrics` | GET | métriques Prometheus |

Toutes les réponses RAG renvoient `sources` : `[ref]`, acte, article, titre, CELEX, lien EUR-Lex et extrait.

---

## 9. Configuration

Fichier `app/config.py` ; chaque valeur peut être surchargée par variable d'environnement ou dans `.env`.

| Variable | Défaut | Rôle |
|---|---|---|
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | modèle d'embeddings (réindexer si on le change) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1500 / 200 | sous-découpage des articles longs (en caractères) |
| `TOP_K` | 5 | nb d'**articles distincts** dans le contexte du LLM (**identique pour les 2 RAG**) |
| `CANDIDATES_K` | 20 | candidats avant fusion et rerank |
| `LLM_MODEL` | `openai/gpt-oss-120b` | génération des réponses (Groq) |
| `FAST_LLM_MODEL` | `openai/gpt-oss-20b` | MultiQuery, HyDE, classification |
| `JUDGE_LLM_MODEL` | `openai/gpt-oss-20b` | juge RAGAS |
| `RERANKER` | `local` | `local` (cross-encoder sur CPU) ou `cohere` |
| `LOCAL_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | cross-encoder local (≈ 2,3 Go, téléchargé au premier usage) |
| `RERANK_MODEL` | `rerank-v3.5` | modèle Cohere si `RERANKER=cohere` |
| `SEED_CACHE` / `SEED_CACHE_PATH` | `1` / `data/cache/advanced_candidates.json` | cache des candidats du mode advanced |
| `CORPUS_DIR`, `INDEX_DIR`, `VECTORSTORE_DIR` | `data/eurlex`, `data/index`, `app/vectorstore_juridique` | chemins |

---

## 10. Structure du projet

```
app/
  config.py              configuration centrale
  llm.py                 fabrique ChatGroq (création à la demande) + comptage des tokens
  dependencies.py        collection Chroma partagée
  main.py                FastAPI (warmup : embeddings, Chroma, BM25)
  api/routes.py          routes
  rag/
    corpus.py            chargement du corpus, noms courts, alias, résolution « article 17 du RGPD »
    chunking.py          découpage par article + indexation (CLI)
    embeddings.py        e5 multilingue avec préfixes query/passage
    hybrid_search.py     BM25 corpus + filtres + RRF
    multiquery.py, hyde.py, reranker.py
    retriever.py         vector / hybrid / advanced + filtres
    pipeline.py          retrieve() + format_context() + generate_answer()  ← réutilisé par le Graph RAG
    prompts.py           TOUS les prompts (français)
    memory.py            mémoire de conversation
    extraction.py        LegalProvision (Instructor)
    evaluate.py          RAGAS
    eval_dataset.py      questions + articles attendus + réponses de référence
    retrieval_metrics.py Recall@k, Hit@k, MRR
    run_evaluation.py    script d'évaluation (CLI)
  agent/                 outils, agent ReAct, routeur, superviseur
data/eurlex/             corpus (copié depuis la collecte)
data/index/chunks.jsonl  chunks + métadonnées (généré)
scripts/collect_eurlex.py
streamlit_app.py         interface
test/test_corpus.py      tests unitaires hors ligne
results/                 résultats d'évaluation (générés)
```

Les fichiers de la version finance (PDF Tesla, ancien index MiniLM, base de l'ancien agent, anciens résultats) ont été supprimés.

---

## 11. Limites connues

- **Relations entre actes non exploitées.** Le RAG vectoriel ne voit ni `relations.csv` ni `article_refs.csv`. Une question
  comme « quels actes modifient le RGPD ? » ne trouve la réponse que si elle figure en toutes lettres dans un article. C'est
  précisément ce que doit apporter le Graph RAG.
- **`relations.csv` ne contient que les relations internes au corpus.** `collect_eurlex.fetch_relations` ne garde une relation
  que si l'acte cible fait partie des 100 actes collectés. Résultat : 619 CITES, 18 AMENDS, 2 BASED_ON et **aucun REPEALS**,
  car les actes abrogés (directive 95/46/CE par le RGPD, directive (UE) 2016/1148 par NIS2) ne sont pas dans le corpus. Le
  projet Graph RAG récupère ces relations vers l'extérieur (732 relations, 668 actes hors corpus, avec leur titre officiel).
- **Hallucinations ponctuelles sur les actes cités.** Exemple constaté le 7 novembre 2026 : à la question « Quelle directive le
  DMA modifie-t-il ? », la réponse citait correctement les articles 51 et 52 du DMA (directives (UE) 2019/1937 et 2020/1828),
  mais ajoutait pour la 2020/1828 le titre « Directive sur les pratiques commerciales déloyales », entre guillemets. Ce titre
  ne figure dans aucun extrait, et il est faux : la directive 2020/1828 porte sur les **actions représentatives** des
  consommateurs ; les pratiques commerciales déloyales relèvent de la directive 2005/29/CE. Le LLM a complété avec ses
  propres connaissances. Les règles 4 et 5 du prompt (§4.5) réduisent le problème sans le supprimer : après correction, le
  mode `vector` ne donnait plus que les numéros, mais le mode `hybrid` ajoutait encore un intitulé (cette fois exact) entre
  guillemets. Pistes : vérifier automatiquement que chaque passage entre « » figure dans le contexte, et mesurer la
  `faithfulness` RAGAS lors de l'évaluation finale.
- **Renvois internes.** Un article qui dit « les conditions visées à l'article 6 » n'entraîne pas la récupération de l'article 6.
- **Le bon article, mais pas le bon morceau.** Un article long est découpé en plusieurs chunks : la recherche peut trouver
  l'article 99 de l'AI Act sans ramener le paragraphe qui fixe le plafond (35 M EUR / 7 %). Le LLM le signale alors, sans
  inventer. Le Recall@k, mesuré par article, ne voit pas ce cas.
- **Modèles à raisonnement.** `gpt-oss` réfléchit avant de répondre. Les tâches simples (MultiQuery, HyDE, classification,
  reformulation, juge RAGAS) tournent avec `reasoning_effort="low"` ; sinon le juge RAGAS épuisait ses tokens
  (`LLMDidNotFinishException`).
- **Considérants et annexes non indexés.** Seuls les articles du dispositif le sont (les annexes de l'AI Act, par exemple, en sont absentes).
- **Questions multi-actes avec k = 5.** Quand la réponse couvre 3 articles de 3 actes, 5 extraits peuvent ne pas suffire ;
  le routeur passe alors à k = 10 pour les questions multi-hop.
- **answer_relevancy** tourne avec `strictness=1`, car l'API Groq n'accepte pas `n>1`.
- **Quota Groq gratuit** : utiliser `--sleep` pour les évaluations avec réponses.

---

## 12. Lien avec le Graph RAG (étape suivante)

Le Graph RAG est un **projet séparé**, dans le dossier voisin `..\..\GraphRAG-juridique` (Neo4j). Ce dépôt reste le RAG
vectoriel seul : aucun code ni aucune donnée propres au graphe n'y sont ajoutés. Le Graph RAG lit les données de ce
projet par leur chemin (variable `VECTOR_RAG_DIR`) et importe son code de génération et d'évaluation.

Pour une comparaison équitable (consigne du projet), le Graph RAG réutilisera :
- **le même modèle d'embeddings** (`app/rag/embeddings.get_embeddings()`), qui servira aussi aux embeddings des nœuds `Article` dans Neo4j ;
- **les mêmes chunks** (`data/index/chunks.jsonl`, reliés aux articles par `article_id`) ;
- **le même prompt et le même LLM** (`app/rag/pipeline.generate_answer`) ;
- **le même nombre d'extraits** (`TOP_K`) ;
- **le même jeu de questions et les mêmes métriques** (`run_evaluation.py`, où il suffira d'ajouter un mode `graph`).

Schéma prévu :
```
(:Document {celex, title, date, type})-[:HAS_ARTICLE]->(:Article {id, num, title, text, embedding})
(:Document)-[:CITES|AMENDS|REPEALS|BASED_ON]->(:Document)
(:Article)-[:REFERS_TO]->(:Article)
(:Document)-[:HAS_SUBJECT]->(:EuroVoc {label})
```
