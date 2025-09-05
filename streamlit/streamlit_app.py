import os
import requests
import streamlit as st

# Base URL de l’Azure Function (modif. possible dans la sidebar)
DEFAULT_FUNC_BASE = os.getenv("FUNC_BASE_URL", "https://reco-func-43896.azurewebsites.net/api")

st.set_page_config(page_title="Reco POC", page_icon="✨", layout="centered")
st.title("🔎 Recommandations d’articles (POC)")

with st.sidebar:
    st.header("⚙️ Paramètres")
    func_base = st.text_input("Function base URL", value=DEFAULT_FUNC_BASE)
    realtime = st.checkbox("Realtime (pas de cache Blob)", value=False)
    st.caption("Ex: https://<ton-app>.azurewebsites.net/api")

st.write("Saisis 1 à N user_id (séparés par des virgules)")
uids_text = st.text_input("user_id(s)", value="1,2,3")
cols = st.columns(2)
with cols[0]:
    trigger = st.button("▶️ Lancer les recommandations")
with cols[1]:
    do_precompute = st.button("🧮 Pré-calcul (écritures Blob via Function)")

def call_function(path: str, params=None, method="GET"):
    url = f"{func_base.rstrip('/')}/{path.lstrip('/')}"
    try:
        if method == "POST":
            r = requests.post(url, params=params, timeout=20)
        else:
            r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e), "url": url}

if do_precompute:
    ok, data = call_function("precompute_http", method="POST")
    if ok:
        st.success(f"Pré-calcul OK: {data}")
    else:
        st.error(f"Pré-calcul KO: {data}")

if trigger:
    uids = [u.strip() for u in uids_text.split(",") if u.strip()]
    if not uids:
        st.warning("Aucun user_id")
    else:
        for uid in uids:
            ok, data = call_function("recommend_get", params={"user_id": uid, "realtime": str(realtime).lower()})
            if ok:
                st.subheader(f"👤 user_id={uid}")
                st.json(data)
            else:
                st.error(f"user_id={uid} — échec: {data}")
