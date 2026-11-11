"""Script manuel (nécessite GROQ_API_KEY) : extraction sur un contexte juridique réel.
    python test/test_extract_data.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag.extraction import extract_legal_provision

context = """[1] Règlement (UE) 2016/679 (RGPD) — Article 33 — Notification à l'autorité de contrôle d'une violation de données à caractère personnel
1. En cas de violation de données à caractère personnel, le responsable du traitement en notifie la violation en question à l'autorité de contrôle compétente conformément à l'article 55, dans les meilleurs délais et, si possible, 72 heures au plus tard après en avoir pris connaissance, à moins que la violation en question ne soit pas susceptible d'engendrer un risque pour les droits et libertés des personnes physiques. Lorsque la notification à l'autorité de contrôle n'a pas lieu dans les 72 heures, elle est accompagnée des motifs du retard.
2. Le sous-traitant notifie au responsable du traitement toute violation de données à caractère personnel dans les meilleurs délais après en avoir pris connaissance."""

if __name__ == "__main__":
    result = extract_legal_provision(context)
    print(result.model_dump_json(indent=2))
    assert not result.not_found_in_context
