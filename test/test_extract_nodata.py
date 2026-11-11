"""Script manuel (nécessite GROQ_API_KEY) : contexte sans disposition juridique -> champs vides.
    python test/test_extract_nodata.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag.extraction import extract_legal_provision

context = """Le Parlement européen siège à Strasbourg et à Bruxelles. Ses membres sont élus
tous les cinq ans au suffrage universel direct. La dernière élection a eu lieu en juin 2024."""

if __name__ == "__main__":
    result = extract_legal_provision(context)
    print(result.model_dump_json(indent=2))
