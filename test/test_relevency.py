"""Script manuel de débogage RAGAS sur UN exemple codé en dur (nécessite GROQ_API_KEY).
Affiche les scores bruts au lieu d'un NaN silencieux.
    python test/test_relevency.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag.evaluate import evaluate_rag

if __name__ == "__main__":
    print(evaluate_rag(
        question="Dans quel délai notifier une violation de données à l'autorité de contrôle selon le RGPD ?",
        answer="Le responsable du traitement notifie la violation dans les meilleurs délais et, si possible, "
               "72 heures au plus tard après en avoir pris connaissance (article 33 du RGPD) [1].",
        contexts=["Règlement (UE) 2016/679 (RGPD) — Article 33 — Notification à l'autorité de contrôle\n"
                  "1. En cas de violation de données à caractère personnel, le responsable du traitement en notifie "
                  "la violation à l'autorité de contrôle compétente dans les meilleurs délais et, si possible, "
                  "72 heures au plus tard après en avoir pris connaissance."],
        reference="Dans les meilleurs délais et si possible dans les 72 heures (article 33 du RGPD).",
    ))
