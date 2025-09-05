import os, io, json, pickle
import numpy as np
import pandas as pd
import azure.functions as func
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp()

# ------------- Config via app settings -------------
C_MODELS = os.environ.get("BLOB_CONTAINER_MODELS", "models")
C_DATA   = os.environ.get("BLOB_CONTAINER_DATA",   "data")
C_PRE    = os.environ.get("BLOB_CONTAINER_PRECOMPUTED", "precomputed")

# ------------- Blob helpers + caches -------------
_blob_client: BlobServiceClient | None = None
_embeddings: np.ndarray | None = None        # shape (n_items, dim)
_item_ids: np.ndarray | None = None          # shape (n_items,)
_id2idx: dict | None = None                  # article_id -> index in embeddings
_meta_df: pd.DataFrame | None = None         # articles metadata
_clicks_df: pd.DataFrame | None = None       # user clicks
_popular_items: np.ndarray | None = None     # fallback: top popular article_ids

def _client() -> BlobServiceClient:
    global _blob_client
    if _blob_client is None:
        conn = os.environ["AzureWebJobsStorage"]
        _blob_client = BlobServiceClient.from_connection_string(conn)
    return _blob_client

def _get_blob_bytes(container: str, name: str) -> bytes:
    bc = _client().get_container_client(container).get_blob_client(name)
    return bc.download_blob().readall()

def _get_blob_text(container: str, name: str) -> str:
    return _get_blob_bytes(container, name).decode("utf-8")

def _upload_json(container: str, name: str, data: dict):
    cc = _client().get_container_client(container)
    cc.upload_blob(name, json.dumps(data, ensure_ascii=False), overwrite=True)

# ------------- Loaders (lazy-cache) -------------
def _load_embeddings():
    """Load embeddings + ids from embeddings.pkl (and build id2idx)."""
    global _embeddings, _item_ids, _id2idx
    if _embeddings is not None:
        return
    raw = _get_blob_bytes(C_MODELS, "embeddings.pkl")
    obj = pickle.loads(raw)

    ids = None
    vecs = None

    # accepted shapes
    if isinstance(obj, dict) and "vectors" in obj:
        vecs = np.asarray(obj["vectors"], dtype=np.float32)
        ids = np.asarray(obj.get("ids", np.arange(vecs.shape[0])), dtype=np.int64)
    elif isinstance(obj, tuple) and len(obj) == 2:
        ids = np.asarray(obj[0])
        vecs = np.asarray(obj[1], dtype=np.float32)
    elif isinstance(obj, pd.DataFrame):
        # expects columns: article_id, vector (list/array)
        ids = obj["article_id"].to_numpy()
        vecs = np.vstack(obj["vector"].to_numpy()).astype(np.float32)
    else:
        raise ValueError("embeddings.pkl format non reconnu. Utilise dict{'ids','vectors'} ou (ids, vectors).")

    if ids.shape[0] != vecs.shape[0]:
        raise ValueError("Mismatch entre taille ids et vectors dans embeddings.pkl")

    _embeddings = vecs
    _item_ids = ids
    _id2idx = {int(aid): i for i, aid in enumerate(_item_ids)}

def _load_meta():
    global _meta_df
    if _meta_df is not None:
        return
    csv = _get_blob_text(C_DATA, "articles_metadata.csv")
    _meta_df = pd.read_csv(io.StringIO(csv))
    # assure la colonne article_id en int
    if _meta_df["article_id"].dtype != np.int64 and _meta_df["article_id"].dtype != np.int32:
        _meta_df["article_id"] = _meta_df["article_id"].astype(int)

def _load_clicks():
    global _clicks_df, _popular_items
    if _clicks_df is not None:
        return
    csv = _get_blob_text(C_DATA, "clicks_sample.csv")
    _clicks_df = pd.read_csv(io.StringIO(csv))
    if _clicks_df["user_id"].dtype != np.int64:
        _clicks_df["user_id"] = _clicks_df["user_id"].astype(int)
    if _clicks_df["click_article_id"].dtype != np.int64:
        _clicks_df["click_article_id"] = _clicks_df["click_article_id"].astype(int)
    # top popular (fallback si user sans historique)
    _popular_items = (
        _clicks_df["click_article_id"].value_counts().index.values.astype(int)
    )

def _ensure_all_loaded():
    _load_embeddings()
    _load_meta()
    _load_clicks()

# ------------- Cosine utils (numpy only) -------------
def _cosine_sim_topn(profile: np.ndarray, M: np.ndarray, topn: int) -> np.ndarray:
    """
    profile: (d,)
    M: (n_items, d)
    return: indices triés des topn plus similaires
    """
    # normalise
    p = profile / (np.linalg.norm(profile) + 1e-12)
    Ms = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    sims = Ms @ p  # (n_items,)
    # argsort descending
    if topn >= sims.shape[0]:
        return np.argsort(-sims)
    return np.argpartition(-sims, topn)[:topn][np.argsort(-sims[np.argpartition(-sims, topn)[:topn]])]

def _user_profile(user_id: int) -> tuple[np.ndarray, set]:
    """Moyenne des embeddings des articles cliqués par l'utilisateur."""
    _ensure_all_loaded()
    clicks = _clicks_df.loc[_clicks_df["user_id"] == int(user_id), "click_article_id"].astype(int).tolist()
    seen = set(int(a) for a in clicks)
    idxs = [ _id2idx[a] for a in clicks if a in _id2idx ]
    if not idxs:
        # aucun historique -> profil nul (on renverra populaires)
        return None, seen
    V = _embeddings[np.array(idxs)]
    prof = V.mean(axis=0)
    return prof, seen

def _recommend(user_id: int, topn=10, verbose=False) -> list[dict]:
    _ensure_all_loaded()
    prof, seen = _user_profile(user_id)

    if prof is None:
        # fallback: populaires non vus
        candidates = [aid for aid in _popular_items if aid not in seen][:topn]
        scores = [1.0 - 1e-6*i for i in range(len(candidates))]
        pairs = list(zip(candidates, scores))
    else:
        topk = max(topn * 5, 100)  # prends plus large avant filtrage
        idxs = _cosine_sim_topn(prof, _embeddings, topk)
        ordered = [int(_item_ids[i]) for i in idxs if int(_item_ids[i]) not in seen]
        # calcule scores cosinus pour top N final (recalcule proprement)
        sel = ordered[:topn]
        if sel:
            M = _embeddings[[ _id2idx[a] for a in sel ]]
            p = prof / (np.linalg.norm(prof) + 1e-12)
            Ms = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
            sims = (Ms @ p).tolist()
        else:
            sims = []
        pairs = list(zip(sel, sims))

    if not verbose:
        return [ {"article_id": aid, "score": float(s)} for aid, s in pairs ]

    # enrichi avec metadata
    df = pd.DataFrame(pairs, columns=["article_id", "score"])
    out = df.merge(
        _meta_df,
        how="left",
        on="article_id"
    )[["article_id","score","category_id","created_at_ts","publisher_id","words_count"]]

    return out.to_dict(orient="records")

# ------------- HTTP endpoints -------------
@app.route(route="recommend_get", auth_level=func.AuthLevel.ANONYMOUS)
def recommend_get(req: func.HttpRequest) -> func.HttpResponse:
    try:
        user_id = req.params.get("user_id")
        if not user_id:
            return func.HttpResponse(json.dumps({"error":"missing user_id"}), status_code=400, mimetype="application/json")

        topn = int(req.params.get("topn", "10"))
        verbose = req.params.get("verbose", "0").lower() in ("1","true","yes")
        realtime = req.params.get("realtime", "false").lower() == "true"

        if not realtime:
            # tente le pré-calcul (si dispo)
            bc = _client().get_container_client(C_PRE).get_blob_client(f"{user_id}.json")
            if bc.exists():
                data = json.loads(bc.download_blob().content_as_text())
                # tronque si besoin
                data["items"] = data.get("items", [])[:topn]
                if verbose and data["items"] and isinstance(data["items"][0], int):
                    # si précompute stocke juste des ids -> enrichir ici
                    enriched = _recommend(int(user_id), topn=topn, verbose=True)
                    data["items"] = enriched
                return func.HttpResponse(json.dumps(data, ensure_ascii=False), mimetype="application/json")

        # calcul en temps réel
        items = _recommend(int(user_id), topn=topn, verbose=verbose)
        body = {"user_id": int(user_id), "items": items, "model": "Content-Based", "source": "realtime"}
        return func.HttpResponse(json.dumps(body, ensure_ascii=False), mimetype="application/json")
    except Exception as e:
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=500, mimetype="application/json")

@app.route(route="precompute_http", methods=["GET","POST"], auth_level=func.AuthLevel.ANONYMOUS)
def precompute_http(req: func.HttpRequest) -> func.HttpResponse:
    try:
        _ensure_all_loaded()
        users = _clicks_df["user_id"].unique().tolist()
        wrote = 0
        for uid in users:
            items = _recommend(int(uid), topn=20, verbose=False)   # stocke léger (ids + score)
            payload = {"user_id": int(uid), "items": [it["article_id"] for it in items], "model":"Content-Based", "source":"precomputed"}
            _upload_json(C_PRE, f"{uid}.json", payload)
            wrote += 1
        return func.HttpResponse(json.dumps({"status":"ok","precomputed":wrote}), mimetype="application/json")
    except Exception as e:
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=500, mimetype="application/json")

@app.route(route="diag", auth_level=func.AuthLevel.ANONYMOUS)
def diag(req: func.HttpRequest) -> func.HttpResponse:
    try:
        _ensure_all_loaded()
        return func.HttpResponse(json.dumps({
            "embeddings": None if _embeddings is None else {"shape": list(_embeddings.shape)},
            "n_items": None if _item_ids is None else int(_item_ids.shape[0]),
            "meta_rows": None if _meta_df is None else int(len(_meta_df)),
            "clicks_rows": None if _clicks_df is None else int(len(_clicks_df)),
            "popular_fallback": None if _popular_items is None else int(len(_popular_items))
        }, ensure_ascii=False), mimetype="application/json")
    except Exception as e:
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=500, mimetype="application/json")

# @app.timer_trigger(schedule="0 0 */2 * * *", arg_name="mytimer")
# def precompute_timer(mytimer: func.TimerRequest) -> None:
#     _ = precompute_http  # tu peux factoriser si tu veux
