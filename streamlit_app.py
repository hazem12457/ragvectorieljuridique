import os
import time

import requests
import streamlit as st

API_URL = os.getenv("BACKEND_URL", "http://localhost:8001")

st.set_page_config(page_title="Assistant juridique — droit européen du numérique", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    .pipeline-badge { background:#eff6ff; color:#1d4ed8; padding:4px 10px; border-radius:12px;
                      font-size:0.75rem; font-weight:600; display:inline-block; margin:2px; }
    .source-card { background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:12px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

MODES = {
    "RAG vectoriel (baseline)": ("rag", "vector"),
    "RAG hybride (vectoriel + BM25)": ("rag", "hybrid"),
    "RAG avancé (MultiQuery + HyDE + BM25 + rerank cross-encoder)": ("rag", "advanced"),
    "Agent (outils juridiques)": ("agent", None),
}


@st.cache_data(ttl=300)
def load_documents():
    try:
        return requests.get(f"{API_URL}/documents", timeout=20).json()
    except Exception:
        return []


@st.cache_data(ttl=60)
def load_stats():
    try:
        return requests.get(f"{API_URL}/corpus/stats", timeout=20).json()
    except Exception:
        return None


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} sources"):
        for s in sources:
            link = f' · <a href="{s["url"]}" target="_blank">EUR-Lex</a>' if s.get("url") else ""
            titre = f" — {s['titre']}" if s.get("titre") else ""
            st.markdown(
                f"""<div class="source-card"><strong>[{s.get('ref')}] {s.get('acte', '?')} — Article {s.get('article', '?')}</strong>{titre}{link}<br>
                <small>{s.get('extrait', '')}…</small></div>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ barre latérale
with st.sidebar:
    st.markdown("## ⚖️ Assistant juridique")
    st.caption("Questions-réponses sur 100 actes européens : RGPD, AI Act, DSA, DMA, NIS2, Data Act, DORA…")

    stats = load_stats()
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Actes", stats["documents"])
        c2.metric("Articles", stats["articles"])
        c3.metric("Chunks", stats["chunks_indexes"] or 0)
        st.caption(f"Embeddings : `{stats['embedding_model']}` · LLM : `{stats['llm']}`")
    else:
        st.error("API injoignable")

    st.divider()
    mode_label = st.radio("Mode", list(MODES), index=0)
    kind, rag_mode = MODES[mode_label]
    k = st.slider("Nombre d'extraits dans le contexte (k)", 1, 15, 5, disabled=kind == "agent")

    docs = load_documents()
    options = ["Tous les actes"] + [d["label"] for d in docs]
    chosen = st.selectbox("Limiter à un acte", options, disabled=kind == "agent")
    celex = next((d["celex"] for d in docs if d["label"] == chosen), None)

    st.divider()
    st.markdown("**🔁 Session**")
    new_session = st.text_input("Identifiant de session", value=st.session_state.get("session_id", "default"))
    if st.button("Appliquer"):
        st.session_state.session_id = new_session
        st.session_state.messages = []
        st.rerun()
    if st.button("🗑 Effacer la conversation", use_container_width=True):
        try:
            requests.delete(f"{API_URL}/chat/{st.session_state.get('session_id', 'default')}", timeout=10)
        except Exception:
            pass
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------ zone principale
st.markdown("## Assistant juridique — droit européen des données et du numérique")
for badge in ["Articles EUR-Lex", "e5 multilingue", "BM25", "RRF", "Rerank cross-encoder", "Mémoire"]:
    st.markdown(f'<span class="pipeline-badge">{badge}</span>', unsafe_allow_html=True)
st.caption("Réponses générées à partir des textes officiels, avec citation des articles. Pas un conseil juridique.")
st.divider()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("session_id", "default")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        meta = message.get("meta")
        if meta:
            st.caption(f"⏱ {meta['latency']:.1f}s · {meta['mode']}")
            render_sources(meta.get("sources"))

if prompt := st.chat_input("Ex. : Dans quel délai notifier une violation de données selon le RGPD ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les textes…"):
            start = time.time()
            try:
                if kind == "rag":
                    params = {"question": prompt, "session_id": st.session_state.session_id,
                              "mode": rag_mode, "k": k}
                    if celex:
                        params["celex"] = celex
                    response = requests.post(f"{API_URL}/chat/memory", params=params, timeout=180)
                else:
                    response = requests.post(f"{API_URL}/agent/ask",
                                             params={"question": prompt, "session_id": st.session_state.session_id},
                                             timeout=180)
                latency = time.time() - start

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    st.write(answer)
                    st.caption(f"⏱ {latency:.1f}s · {mode_label}")
                    render_sources(sources)
                    st.session_state.messages.append({"role": "assistant", "content": answer,
                                                      "meta": {"latency": latency, "mode": mode_label, "sources": sources}})
                else:
                    st.error(f"Erreur {response.status_code} : {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Impossible de joindre l'API. Est-elle lancée (uvicorn app.main:app --port 8001) ?")
            except requests.exceptions.Timeout:
                st.error("Délai dépassé.")
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")
