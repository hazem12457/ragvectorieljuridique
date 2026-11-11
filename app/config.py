"""Configuration centrale du RAG juridique.

Toutes les valeurs peuvent être surchargées par variable d'environnement (ou dans `.env`).
Les chemins sont absolus (calculés depuis la racine du projet) : le code fonctionne
quel que soit le dossier depuis lequel on le lance.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# --- Corpus EUR-Lex (produit par scripts/collect_eurlex.py) ---
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", ROOT / "data" / "eurlex"))
ARTICLES_PATH = CORPUS_DIR / "articles.jsonl"
DOCUMENTS_PATH = CORPUS_DIR / "documents.csv"
RELATIONS_PATH = CORPUS_DIR / "relations.csv"
ARTICLE_REFS_PATH = CORPUS_DIR / "article_refs.csv"

# --- Index ---
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"          # chunks + métadonnées (sert aussi au BM25 et au Graph RAG)
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", str(ROOT / "app" / "vectorstore_juridique"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "eurlex_numerique")

# --- Embeddings (multilingue : le corpus est en français) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# --- Découpage ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))        # en caractères (≈ 400 tokens avec l'en-tête, < 512 max d'e5)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Recherche ---
TOP_K = int(os.getenv("TOP_K", "5"))                  # nb de chunks dans le contexte du LLM (identique pour les 2 RAG)
CANDIDATES_K = int(os.getenv("CANDIDATES_K", "20"))   # nb de candidats avant fusion / reranking
RRF_K = int(os.getenv("RRF_K", "60"))                 # constante de la Reciprocal Rank Fusion

# --- LLM (Groq) ---
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")          # génération de la réponse finale
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "openai/gpt-oss-20b")  # MultiQuery, HyDE, classification
JUDGE_LLM_MODEL = os.getenv("JUDGE_LLM_MODEL", "openai/gpt-oss-20b")  # juge RAGAS

# --- Reranking (cross-encoder) : LE MÊME pour tous les modes qui rerankent (advanced, graph) ---
RERANKER = os.getenv("RERANKER", "local")                     # local (reproductible, gratuit) | cohere
LOCAL_RERANK_MODEL = os.getenv("LOCAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")   # multilingue
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-v3.5")       # si RERANKER=cohere

# --- Cache des candidats du mode advanced (MultiQuery et HyDE sont générés par le LLM : sans cache,
#     deux exécutions n'ont pas les mêmes candidats ; le Graph RAG relit ce cache pour partir des mêmes graines) ---
SEED_CACHE = os.getenv("SEED_CACHE", "1") == "1"
SEED_CACHE_PATH = Path(os.getenv("SEED_CACHE_PATH", ROOT / "data" / "cache" / "advanced_candidates.json"))

# --- Knowledge Graph (Neo4j) ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# --- Divers ---
AGENT_DB_PATH = os.getenv("AGENT_DB_PATH", str(ROOT / "agent-memory-juridique.db"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT / "results"))
EURLEX_URL = "https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:{celex}"
