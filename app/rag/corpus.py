"""Accès au corpus EUR-Lex : documents (actes), articles, alias usuels.

Remplace la notion « company / year » du projet finance par :
    acte (CELEX) · type (REG / DIR / DEC) · année · article.
"""
import csv
import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from app import config

# Noms usuels des principaux actes du corpus (servent à l'affichage, au BM25 et à la
# résolution « article 17 du RGPD » -> 32016R0679_art_17).
ACT_ALIASES: dict[str, list[str]] = {
    "32016R0679": ["RGPD", "GDPR", "règlement général sur la protection des données"],
    "32016L0680": ["directive police-justice", "directive (UE) 2016/680"],
    "32018R1725": ["EUDPR", "règlement protection des données des institutions de l'Union"],
    "32002L0058": ["directive ePrivacy", "directive vie privée et communications électroniques"],
    "32022R2065": ["DSA", "Digital Services Act", "règlement sur les services numériques"],
    "32022R1925": ["DMA", "Digital Markets Act", "règlement sur les marchés numériques"],
    "32022R0868": ["DGA", "Data Governance Act", "règlement sur la gouvernance des données"],
    "32023R2854": ["Data Act", "règlement sur les données"],
    "32024R1689": ["AI Act", "RIA", "règlement sur l'intelligence artificielle", "IA Act"],
    "32022L2555": ["NIS2", "NIS 2", "directive NIS 2"],
    "32019R0881": ["Cybersecurity Act", "règlement sur la cybersécurité", "règlement ENISA"],
    "32025R0327": ["EHDS", "espace européen des données de santé"],
    "32019L0790": ["directive droit d'auteur", "directive sur le droit d'auteur dans le marché unique numérique"],
    "32022L2557": ["directive CER", "directive résilience des entités critiques"],
    "32024L2853": ["directive responsabilité du fait des produits défectueux"],
    "32024R1183": ["eIDAS 2", "eIDAS2", "identité numérique européenne"],
    "32022R2554": ["DORA", "résilience opérationnelle numérique"],
    "32024R2847": ["CRA", "Cyber Resilience Act", "règlement sur la cyberrésilience"],
    "32024R0903": ["Interoperable Europe Act", "règlement pour une Europe interopérable"],
    "32019R1150": ["règlement P2B", "platform to business"],
    "32018R1807": ["règlement libre flux des données à caractère non personnel"],
    "32024R0900": ["règlement publicité politique"],
    "32023R0988": ["GPSR", "règlement sécurité générale des produits"],
    "32019L0882": ["European Accessibility Act", "acte européen sur l'accessibilité"],
    "32018L1972": ["code des communications électroniques européen", "CCEE"],
    "32024R1083": ["EMFA", "European Media Freedom Act", "législation européenne sur la liberté des médias"],
    "32000L0031": ["directive commerce électronique", "directive e-commerce"],
    "32019L1024": ["directive open data", "directive données ouvertes"],
    "32019L1937": ["directive lanceurs d'alerte"],
    "32005L0029": ["directive pratiques commerciales déloyales"],
    "31993L0013": ["directive clauses abusives"],
    "32011L0083": ["directive droits des consommateurs"],
    "32019L0770": ["directive contenus numériques"],
    "32001L0029": ["directive InfoSoc", "directive société de l'information"],
    "32016L0943": ["directive secret des affaires"],
    "32004L0048": ["directive IPRED", "directive respect des droits de propriété intellectuelle"],
}

DOC_TYPE_LABELS = {"REG": "Règlement", "DIR": "Directive", "DEC": "Décision"}
CELEX_RE = re.compile(r"^\d{5}[A-Z]\d{4}$")


def normalize(text: str) -> str:
    """minuscules, sans accents, apostrophes et espaces normalisés."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("’", "'").replace("«", " ").replace("»", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def short_act_name(title: str) -> str:
    """'Règlement (UE) 2016/679 du Parlement européen…' -> 'Règlement (UE) 2016/679'."""
    head = re.split(r"\s+(?:du|de la|de l['’])\s", title, maxsplit=1)[0]
    head = re.sub(r"\bn\s*[o°]\s*(?=\d)", "n° ", head)   # 'n o  765/2008' -> 'n° 765/2008'
    head = re.sub(r"\s+", " ", head).strip().rstrip(",")
    return head


@dataclass
class LegalDocument:
    celex: str
    doc_type: str
    date: str
    title: str
    eurovoc: list[str]
    n_articles: int
    short_name: str = ""
    aliases: list[str] = field(default_factory=list)

    @property
    def year(self) -> int:
        return int(self.date[:4]) if self.date[:4].isdigit() else 0

    @property
    def label(self) -> str:
        """Nom affiché : 'Règlement (UE) 2016/679 (RGPD)'."""
        return f"{self.short_name} ({self.aliases[0]})" if self.aliases else self.short_name

    @property
    def url(self) -> str:
        return config.EURLEX_URL.format(celex=self.celex)


@dataclass
class Article:
    id: str
    celex: str
    num: str
    title: str
    text: str
    doc_type: str
    date: str
    eurovoc: list[str]


def normalize_article_num(num: str) -> str:
    num = normalize(str(num))
    num = re.sub(r"^(article|art\.?)\s*", "", num)
    return num.strip(" .")


class Corpus:
    def __init__(self, articles_path=config.ARTICLES_PATH, documents_path=config.DOCUMENTS_PATH):
        self.documents: dict[str, LegalDocument] = {}
        with open(documents_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                celex = row["celex"]
                self.documents[celex] = LegalDocument(
                    celex=celex,
                    doc_type=row["type"],
                    date=row["date"],
                    title=row["title"],
                    eurovoc=[e for e in row.get("eurovoc", "").split("|") if e],
                    n_articles=int(row.get("n_articles") or 0),
                    short_name=short_act_name(row["title"]),
                    aliases=ACT_ALIASES.get(celex, []),
                )

        self.articles: dict[str, Article] = {}
        self._by_ref: dict[tuple[str, str], Article] = {}
        with open(articles_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                art = Article(
                    id=r["id"],
                    celex=r["celex"],
                    num=str(r["article_num"]),
                    title=r.get("article_title") or "",
                    text=r.get("text") or "",
                    doc_type=r.get("doc_type", ""),
                    date=r.get("date", ""),
                    eurovoc=r.get("eurovoc") or [],
                )
                self.articles[art.id] = art
                self._by_ref[(art.celex, normalize_article_num(art.num))] = art

        # index alias normalisé -> celex
        self._alias_index: dict[str, str] = {}
        for celex, doc in self.documents.items():
            for alias in doc.aliases + [doc.short_name, celex]:
                self._alias_index[normalize(alias)] = celex

    # --- affichage ---
    def doc_label(self, celex: str) -> str:
        doc = self.documents.get(celex)
        return doc.label if doc else celex

    def article_header(self, art: Article) -> str:
        header = f"{self.doc_label(art.celex)} — Article {art.num}"
        return f"{header} — {art.title}" if art.title else header

    # --- résolution d'une référence d'acte ---
    def resolve_act(self, ref: str) -> str | None:
        """'RGPD', 'AI Act', '2016/679', 'directive 2002/58/CE', '32016R0679' -> CELEX."""
        if not ref:
            return None
        raw = ref.strip()
        if CELEX_RE.match(raw.upper()):
            return raw.upper() if raw.upper() in self.documents else None
        n = normalize(raw)
        if n in self._alias_index:
            return self._alias_index[n]
        # numéro d'acte (ex. 2016/679, 93/13, 182/2011)
        m = re.search(r"(\d{2,4})\s*/\s*(\d{1,4})", n)
        if m:
            number = f"{m.group(1)}/{m.group(2)}"
            for celex, doc in self.documents.items():
                if re.search(rf"(?<![\d/]){re.escape(number)}(?![\d])", doc.short_name):
                    return celex
        # alias contenu dans la référence (ex. « article 5 du RGPD »)
        best = None
        for alias_n, celex in self._alias_index.items():
            if len(alias_n) >= 3 and re.search(rf"\b{re.escape(alias_n)}\b", n):
                if best is None or len(alias_n) > len(best[0]):
                    best = (alias_n, celex)
        return best[1] if best else None

    def get_article(self, act_ref: str, num: str) -> Article | None:
        celex = self.resolve_act(act_ref)
        if not celex:
            return None
        return self._by_ref.get((celex, normalize_article_num(num)))

    def articles_of(self, celex: str) -> list[Article]:
        return [a for a in self.articles.values() if a.celex == celex]

    def search_documents(self, query: str, limit: int = 10) -> list[LegalDocument]:
        """Recherche simple par mots-clés dans les titres, alias et descripteurs EuroVoc."""
        celex = self.resolve_act(query)
        if celex:
            return [self.documents[celex]]
        words = [w for w in re.findall(r"\w+", normalize(query)) if len(w) > 2]
        scored = []
        for doc in self.documents.values():
            hay = normalize(" ".join([doc.title, *doc.aliases, *doc.eurovoc]))
            score = sum(1 for w in words if w in hay)
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda x: (-x[0], x[1].date))
        return [d for _, d in scored[:limit]]


@lru_cache(maxsize=1)
def get_corpus() -> Corpus:
    return Corpus()
