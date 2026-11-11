"""Extraction structurée (Instructor + Groq) : transforme les articles récupérés en JSON.

Version finance : chiffre d'affaires, résultat net, R&D…
Version juridique : fiche de disposition — qui est visé, quelles obligations, quels
droits, quels délais, quelles sanctions, quels renvois. Tout champ absent du contexte
reste vide (le modèle ne peut pas répondre hors schéma : Mode.TOOLS).
"""
import os
from typing import Optional

from pydantic import BaseModel, Field

from app import config
from app.rag.prompts import EXTRACTION_SYSTEM_PROMPT

_client = None


def get_client():
    global _client
    if _client is None:
        import instructor
        from groq import Groq

        _client = instructor.from_groq(Groq(api_key=os.getenv("GROQ_API_KEY")), mode=instructor.Mode.TOOLS)
    return _client


class LegalProvision(BaseModel):
    acte: Optional[str] = Field(None, description="Acte concerné, ex. 'Règlement (UE) 2016/679 (RGPD)'.")
    article: Optional[str] = Field(None, description="Numéro de l'article principal, ex. '33'.")
    objet: Optional[str] = Field(None, description="Objet de la disposition en une phrase.")
    destinataires: list[str] = Field(default_factory=list,
        description="Acteurs soumis à la disposition, ex. 'responsable du traitement', 'fournisseur de système d'IA à haut risque', 'États membres'.")
    obligations: list[str] = Field(default_factory=list, description="Obligations imposées, formulées brièvement.")
    droits: list[str] = Field(default_factory=list, description="Droits conférés (ex. droit à l'effacement).")
    delais: list[str] = Field(default_factory=list, description="Délais explicites, ex. '72 heures après en avoir pris connaissance'.")
    sanctions: list[str] = Field(default_factory=list,
        description="Sanctions ou amendes explicites, ex. 'jusqu'à 20 000 000 EUR ou 4 % du chiffre d'affaires annuel mondial'.")
    renvois: list[str] = Field(default_factory=list, description="Autres articles ou actes cités, ex. 'article 58, paragraphe 2'.")
    source_quote: Optional[str] = Field(None, description="Courte citation exacte (moins de 25 mots) justifiant l'extraction.")
    not_found_in_context: bool = Field(False,
        description="true UNIQUEMENT si le contexte ne contient aucune disposition juridique exploitable.")


def extract_legal_provision(context: str) -> LegalProvision:
    return get_client().chat.completions.create(
        model=config.LLM_MODEL,
        response_model=LegalProvision,
        max_retries=2,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexte :\n{context}\n\nExtrais la fiche de la disposition juridique."},
        ],
    )
