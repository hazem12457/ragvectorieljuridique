"""Collecte d'un corpus EUR-Lex (un seul volet juridique) depuis Cellar.

Etapes :
  1. SPARQL : actes voisins des actes de depart (cites / citant), classes par nombre de liens
  2. SPARQL : metadonnees (titre FR, date, type, EuroVoc)
  3. SPARQL : relations entre les actes retenus (cite, modifie, abroge, base legale)
  4. REST Cellar : telechargement XHTML (FR) + decoupage en articles
  5. Regex : references entre articles

Sorties (dans --out) :
  documents.csv, relations.csv, articles.jsonl, article_refs.csv, raw/<celex>.xhtml

Usage :
  python collect_eurlex.py --out data --max-docs 100
"""
import argparse
import csv
import json
import os
import re
import time
from collections import Counter

import requests
from bs4 import BeautifulSoup

# Actes de depart du volet "protection des donnees et numerique".
# Pour un autre volet, remplace simplement cette liste.
SEEDS = [
    "32016R0679",  # RGPD
    "32016L0680",  # Directive police-justice
    "32018R1725",  # Protection des donnees par les institutions UE
    "32002L0058",  # ePrivacy
    "32022R2065",  # DSA
    "32022R1925",  # DMA
    "32022R0868",  # Data Governance Act
    "32023R2854",  # Data Act
    "32024R1689",  # AI Act
    "32022L2555",  # NIS2
    "32019R0881",  # Cybersecurity Act
]

SPARQL = "https://publications.europa.eu/webapi/rdf/sparql"
CELLAR = "http://publications.europa.eu/resource/celex/"
CDM = "PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>\n"
XSD_STR = "^^<http://www.w3.org/2001/XMLSchema#string>"
# Legislation (secteur 3) : reglements (R), directives (L), decisions (D)
LEGIS_RE = "^3[0-9]{4}[RLD][0-9]{4}$"
REL_TYPES = {
    "work_cites_work": "CITES",
    "resource_legal_amends_resource_legal": "AMENDS",
    "resource_legal_repeals_resource_legal": "REPEALS",
    "resource_legal_based_on_resource_legal": "BASED_ON",
}
DOC_TYPES = {"R": "REG", "L": "DIR", "D": "DEC"}


def sparql(query, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(SPARQL, data={"query": CDM + query, "format": "application/sparql-results+json"},
                              timeout=180)
            r.raise_for_status()
            return [{k: v["value"] for k, v in b.items()} for b in r.json()["results"]["bindings"]]
        except requests.RequestException as e:
            print(f"  SPARQL erreur ({e}), nouvel essai...")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("SPARQL indisponible")


def values(celexes):
    return " ".join(f'"{c}"{XSD_STR}' for c in celexes)


def batches(items, n=40):
    for i in range(0, len(items), n):
        yield items[i:i + n]


# ---------------------------------------------------------------- 1. selection
def select_documents(max_docs):
    score = Counter()
    for batch in batches(SEEDS):
        v = values(batch)
        out = sparql(f"""SELECT ?n (COUNT(*) AS ?k) WHERE {{
            VALUES ?s {{ {v} }} ?w cdm:resource_legal_id_celex ?s .
            ?w cdm:work_cites_work ?x . ?x cdm:resource_legal_id_celex ?n .
            FILTER(REGEX(STR(?n), "{LEGIS_RE}")) }} GROUP BY ?n""")
        inc = sparql(f"""SELECT ?n (COUNT(*) AS ?k) WHERE {{
            VALUES ?s {{ {v} }} ?w cdm:resource_legal_id_celex ?s .
            ?x cdm:work_cites_work ?w . ?x cdm:resource_legal_id_celex ?n .
            FILTER(REGEX(STR(?n), "{LEGIS_RE}")) }} GROUP BY ?n""")
        for row in out + inc:
            score[row["n"]] += int(row["k"])
    selected = list(SEEDS)
    for celex, _ in score.most_common():
        if len(selected) >= max_docs:
            break
        if celex not in selected:
            selected.append(celex)
    return selected


# ---------------------------------------------------------------- 2. metadonnees
def fetch_metadata(celexes):
    docs = {}
    for batch in batches(celexes):
        rows = sparql(f"""PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT ?celex (SAMPLE(?d) AS ?date) (SAMPLE(?t) AS ?title)
                   (GROUP_CONCAT(DISTINCT ?ev; separator="|") AS ?eurovoc) WHERE {{
              VALUES ?celex {{ {values(batch)} }}
              ?w cdm:resource_legal_id_celex ?celex .
              OPTIONAL {{ ?w cdm:work_date_document ?d }}
              OPTIONAL {{ ?e cdm:expression_belongs_to_work ?w ;
                            cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/FRA> ;
                            cdm:expression_title ?t }}
              OPTIONAL {{ ?w cdm:work_is_about_concept_eurovoc ?c . ?c skos:prefLabel ?ev . FILTER(lang(?ev) = "fr") }}
            }} GROUP BY ?celex""")
        for r in rows:
            c = r["celex"]
            docs[c] = {"celex": c, "type": DOC_TYPES.get(c[5], "OTHER"), "date": r.get("date", ""),
                       "title": r.get("title", "").replace(" ", " "), "eurovoc": r.get("eurovoc", "")}
    return docs


# ---------------------------------------------------------------- 3. relations
def fetch_relations(celexes):
    keep = set(celexes)
    preds = " ".join(f"cdm:{p}" for p in REL_TYPES)
    rels = set()
    for batch in batches(celexes):
        rows = sparql(f"""SELECT DISTINCT ?s ?p ?t WHERE {{
            VALUES ?s {{ {values(batch)} }} VALUES ?p {{ {preds} }}
            ?w cdm:resource_legal_id_celex ?s . ?w ?p ?x . ?x cdm:resource_legal_id_celex ?t .
            FILTER(REGEX(STR(?t), "{LEGIS_RE}")) }}""")
        for r in rows:
            if r["t"] in keep and r["t"] != r["s"]:
                rels.add((r["s"], r["t"], REL_TYPES[r["p"].split("#")[1]]))
    return sorted(rels)


# ---------------------------------------------------------------- 4. textes
def download(celex, raw_dir):
    path = os.path.join(raw_dir, f"{celex}.xhtml")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    for accept in ("application/xhtml+xml", "text/html"):
        try:
            r = requests.get(CELLAR + celex, headers={"Accept": accept, "Accept-Language": "fra"}, timeout=180)
        except requests.RequestException:
            continue
        if r.status_code == 200 and len(r.content) > 2000:
            text = r.content.decode("utf-8", "replace")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return text
    return None


def clean(s):
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip()


def split_articles(celex, html):
    soup = BeautifulSoup(html, "html.parser")
    articles = []
    # Format recent : <div class="eli-subdivision" id="art_6">
    for div in soup.find_all("div", id=re.compile(r"^art_\d+[a-z]*$")):
        num = div["id"][4:]
        title_p = div.find("p", class_=re.compile(r"sti-art"))
        title = clean(title_p.get_text()) if title_p else ""
        for p in div.find_all("p", class_=re.compile(r"ti-art")):
            p.decompose()
        text = clean(div.get_text(" "))
        if text:
            articles.append({"num": num, "title": title, "text": text})
    if articles:
        return articles
    # Format ancien : decoupage sur les lignes "Article N"
    body = soup.get_text("\n")
    parts = re.split(r"\n\s*Article\s+(\d+[a-z]*)\s*\n", body)
    for i in range(1, len(parts) - 1, 2):
        text = clean(parts[i + 1])
        if text:
            articles.append({"num": parts[i], "title": "", "text": text})
    return articles


# ---------------------------------------------------------------- 5. references
ACT_RE = re.compile(
    r"(r[èe]glement|directive|d[ée]cision)\s+(?:d[’']ex[ée]cution\s+|d[ée]l[ée]gu[ée]e?\s+)?"
    r"(?:\((?:UE|CE|CEE|Euratom)\)\s+)?(?:n[°o]\s*)?(\d{2,4})/(\d{1,4})(?:/(?:CEE|CE|UE))?",
    re.IGNORECASE)
ART_RE = re.compile(r"\barticles?\s+(\d+[a-z]?)", re.IGNORECASE)


def year_ok(x):
    return len(x) == 4 and 1950 <= int(x) <= 2035


def act_to_celex(kind, a, b):
    kind = kind.lower()
    letter = "R" if kind.startswith("r") else ("L" if kind.startswith("di") else "D")
    if year_ok(a) and not year_ok(b):
        year, num = a, b
    elif year_ok(b) and not year_ok(a):
        year, num = b, a
    elif len(a) == 2 and letter in "LD":  # ex. directive 95/46/CE
        year, num = ("19" if int(a) > 50 else "20") + a, b
    else:
        year, num = a, b
    return f"3{year}{letter}{int(num):04d}"


def extract_refs(celex, article, known_articles):
    refs = set()
    text = article["text"]
    for m in ART_RE.finditer(text):
        target_num = m.group(1)
        after = text[m.end():m.end() + 140]
        # "article 9 [, paragraphe 1,] du reglement (UE) 2016/679" -> autre acte
        act = re.match(r"[^.;]{0,80}?\b(?:du|de la|de l[’'])\s*", after)
        act = ACT_RE.match(after[act.end():]) if act else None
        if act:
            target_doc = act_to_celex(*act.groups())
        elif re.match(r"[^.;]{0,80}?\b(?:du|de la|de l[’'])\s*(?:r[èe]glement|directive|d[ée]cision|trait[ée]|charte|"
                      r"convention|protocole)\b(?!\s+pr[ée]sent)", after, re.IGNORECASE):
            continue  # renvoi vers un acte non identifie
        else:
            target_doc = celex  # renvoi interne
        target = f"{target_doc}_art_{target_num}"
        if target != article["id"]:
            refs.add((target, target in known_articles))
    return refs


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-docs", type=int, default=100)
    args = ap.parse_args()
    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("1/5 Selection des documents...")
    celexes = select_documents(args.max_docs)
    print(f"    {len(celexes)} documents retenus")

    print("2/5 Metadonnees...")
    docs = fetch_metadata(celexes)

    print("3/5 Relations entre documents...")
    rels = fetch_relations(celexes)
    print(f"    {len(rels)} relations")

    print("4/5 Telechargement et decoupage en articles...")
    all_articles = []
    for i, c in enumerate(celexes, 1):
        html = download(c, raw_dir)
        arts = split_articles(c, html) if html else []
        docs.setdefault(c, {"celex": c, "type": DOC_TYPES.get(c[5], "OTHER"), "date": "", "title": "", "eurovoc": ""})
        docs[c]["n_articles"] = len(arts)
        for a in arts:
            all_articles.append({"id": f"{c}_art_{a['num']}", "celex": c, "doc_title": docs[c]["title"],
                                 "doc_type": docs[c]["type"], "date": docs[c]["date"],
                                 "article_num": a["num"], "article_title": a["title"], "text": a["text"],
                                 "eurovoc": [e for e in docs[c]["eurovoc"].split("|") if e]})
        print(f"    [{i}/{len(celexes)}] {c}: {len(arts)} articles" + ("" if html else " (texte indisponible)"))
        time.sleep(0.5)

    print("5/5 References entre articles...")
    known = {a["id"] for a in all_articles}
    art_refs = set()
    for a in all_articles:
        for target, in_corpus in extract_refs(a["celex"], a, known):
            art_refs.add((a["id"], target, in_corpus))

    with open(os.path.join(args.out, "documents.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["celex", "type", "date", "title", "eurovoc", "n_articles"])
        w.writeheader()
        for c in celexes:
            w.writerow(docs[c])
    with open(os.path.join(args.out, "relations.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "type"])
        w.writerows(rels)
    with open(os.path.join(args.out, "articles.jsonl"), "w", encoding="utf-8") as f:
        for a in all_articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "article_refs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_article", "target_article", "target_in_corpus"])
        w.writerows(sorted(art_refs))

    n_in = sum(1 for r in art_refs if r[2])
    print(f"\nTermine : {len(celexes)} documents, {len(all_articles)} articles, {len(rels)} relations documents, "
          f"{len(art_refs)} references d'articles ({n_in} vers un article du corpus).")


if __name__ == "__main__":
    main()
