import os
import requests
import hashlib
import streamlit as st
import pandas as pd

DEFAULT_FUNC_BASE = os.getenv(
    "FUNC_BASE_URL", "https://reco-func-43896.azurewebsites.net/api"
)

st.set_page_config(page_title="Reco POC", page_icon="✨", layout="centered")
st.title("🔎 Recommandations d’articles")


def recs_to_df(payload: dict) -> pd.DataFrame:
    recs = payload.get("recommendations", [])
    if isinstance(recs, dict):
        recs = list(recs.values())
    df = pd.DataFrame(recs)
    if df.empty:
        return df

    df.insert(0, "rank", range(1, len(df) + 1))
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce").round(4)

    if "created_at_ts" in df.columns:
        df["created_at"] = (
            pd.to_datetime(df["created_at_ts"], unit="ms", utc=True)
            .dt.tz_convert("Europe/Paris")
            .dt.strftime("%Y-%m-%d %H:%M")
        )

    pref = ["rank", "article_id", "score", "category_id", "words_count", "created_at"]
    cols = [c for c in pref if c in df.columns] + [
        c for c in df.columns if c not in pref
    ]
    return df[cols]


if "reco_cache" not in st.session_state:
    st.session_state.reco_cache = {}
if "last_uids" not in st.session_state:
    st.session_state.last_uids = []

with st.sidebar:
    st.header("⚙️ Paramètres")
    func_base = st.text_input("Function base URL", value=DEFAULT_FUNC_BASE)
    realtime = st.checkbox("Realtime (pas de cache Blob)", value=False)
    topn = st.slider(
        "Top N recommandations",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="Nombre d’articles renvoyés par user (max 5).",
    )

with st.sidebar:
    st.header("🗂️ Affichage")
    ALL_COLS = [
        "rank",
        "article_id",
        "score",
        "category_id",
        "words_count",
        "created_at",
        "created_at_ts",
        "publisher_id",
    ]
    DEFAULT_DISPLAY_COLS = ["rank", "article_id", "score"]

    if "selected_cols" not in st.session_state:
        st.session_state.selected_cols = DEFAULT_DISPLAY_COLS

    selected_cols = st.multiselect(
        "Colonnes à afficher",
        options=ALL_COLS,
        default=st.session_state.selected_cols,
        help="Les colonnes non présentes dans la réponse seront ignorées automatiquement.",
    )
    st.session_state.selected_cols = selected_cols

with st.sidebar:
    st.divider()
    with st.expander("❓ Centre d’aide", expanded=False):
        st.markdown("""
- **Function base URL** : URL de ton Azure Function (ex. `/api`).  
  Sert pour appeler `recommend_get`.

- **Realtime (pas de cache Blob)** : si activé, on ignore les JSON pré-calculés (`precomputed/`) et on recalcule en direct.

- **user_id(s)** : liste d’IDs séparés par des virgules (ex. `1, 2, 3`).  
  Le bouton **▶️ Lancer les recommandations** appelle l’API pour chaque user.

- **Colonnes à afficher** : choisis les colonnes visibles dans la table (par défaut **rank**, **article_id** et **score**).

- **⬇️ Export CSV** : télécharge exactement ce qui est affiché (colonnes et ordre).  

- **Modèle** : 
  - **Content-Based** (similarité cosinus sur embeddings) si l’utilisateur a de l’historique.
  - **Popularity** en **cold-start** (aucun historique), pour proposer des articles populaires non vus.
""")

st.write("Saisis 1 à N user_id (séparés par des virgules)")
uids_text = st.text_input("user_id(s)", value="1,2,3")
cols = st.columns(2)
with cols[0]:
    trigger = st.button("▶️ Lancer les recommandations")
# with cols[1]:
#     do_precompute = st.button("🧮 Pré-calcul (écritures Blob via Function)")


def call_function(path: str, params=None, method="GET"):
    url = f"{func_base.rstrip('/')}/{path.lstrip('/')}"
    try:
        if method == "POST":
            r = requests.post(url, params=params, timeout=300)
        else:
            r = requests.get(url, params=params, timeout=300)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e), "url": url}


# if do_precompute:
#     ok, data = call_function("precompute_http", method="POST")
#     if ok:
#         st.success(f"Pré-calcul OK: {data}")
#     else:
#         st.error(f"Pré-calcul KO: {data}")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_recos(func_base: str, uid: str, realtime: bool, topn: int):
    ok, data = call_function(
        "recommend_get",
        params={
            "user_id": uid,
            "realtime": str(realtime).lower(),
            "verbose": "1",
            "topn": topn,
        },
        method="GET",
    )
    return ok, data


model = None
source = None
if trigger:
    uids = [u.strip() for u in uids_text.split(",") if u.strip()]
    st.session_state.last_uids = uids[:]
    for uid in uids:
        ok, data = fetch_recos(func_base, uid, realtime, topn)
        if ok:
            st.session_state.reco_cache[uid] = recs_to_df(data)
            model = data.get("model", "—")
            source = data.get("source", "—")
        else:
            st.session_state.reco_cache[uid] = pd.DataFrame()  # vide = erreur
if model and source:
    st.subheader(f"model: {model} • source: {source}")
    st.subheader(f"TopN = {topn}")

for uid in st.session_state.last_uids:
    st.subheader(f"👤 user_id={uid}")
    df = st.session_state.reco_cache.get(uid, pd.DataFrame())
    if df.empty:
        st.info("Aucune recommandation renvoyée.")
        continue

    display_cols = [
        c
        for c in st.session_state.get("selected_cols", ["article_id", "score"])
        if c in df.columns
    ]
    if not display_cols:
        display_cols = list(df.columns)

    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    key = f"dl-{uid}-{hashlib.md5(','.join(display_cols).encode()).hexdigest()[:8]}"
    st.download_button(
        f"⬇️ Export CSV (user {uid})",
        data=df[display_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"reco_user_{uid}.csv",
        mime="text/csv",
        key=key,
    )
