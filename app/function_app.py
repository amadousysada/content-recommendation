import os
import io
import json
import pickle
import logging
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Optional

import azure.functions as func
from azure.storage.blob import BlobServiceClient

from utils import _recommend_popular, _cache, ContentBasedRecommender

app = func.FunctionApp()

BLOB_CONTAINER_MODELS = os.environ.get("BLOB_CONTAINER_MODELS", "models")
BLOB_CONTAINER_DATA = os.environ.get("BLOB_CONTAINER_DATA", "data")
BLOB_CONTAINER_PRECOMPUTED = os.environ.get("BLOB_CONTAINER_PRECOMPUTED", "precomputed")

_blob_client: Optional[BlobServiceClient] = None


def _client() -> BlobServiceClient:
    global _blob_client
    if _blob_client is None:
        conn = os.environ["AzureWebJobsStorage"]
        _blob_client = BlobServiceClient.from_connection_string(conn)
    return _blob_client


def _get_blob_bytes(container: str, name: str) -> bytes:
    bc = _client().get_container_client(container).get_blob_client(name)
    return bc.download_blob().readall()


def _get_blob_text(container: str, name: str, encoding="utf-8") -> str:
    bc = _client().get_container_client(container).get_blob_client(name)
    return bc.download_blob().content_as_text(encoding=encoding)


def _load_articles_df() -> pd.DataFrame:
    if _cache["articles_df"] is not None:
        return _cache["articles_df"]  # type: ignore
    csv_txt = _get_blob_text(BLOB_CONTAINER_DATA, "articles_metadata.csv")
    df = pd.read_csv(io.StringIO(csv_txt))
    if "article_id" not in df.columns:
        raise ValueError("articles_metadata.csv doit contenir la colonne 'article_id'")
    df["article_id"] = df["article_id"].astype(int)
    _cache["articles_df"] = df
    return df


def _load_clicks_df() -> pd.DataFrame:
    if _cache["clicks_df"] is not None:
        return _cache["clicks_df"]  # type: ignore
    csv_txt = _get_blob_text(BLOB_CONTAINER_DATA, "clicks_sample.csv")
    df = pd.read_csv(io.StringIO(csv_txt))
    for col in ["user_id", "click_article_id"]:
        if col not in df.columns:
            raise ValueError(
                "clicks_sample.csv doit contenir 'user_id' et 'click_article_id'"
            )
    df["user_id"] = df["user_id"].astype(int)
    df["click_article_id"] = df["click_article_id"].astype(int)
    _cache["clicks_df"] = df
    return df


def _parse_embeddings_pickle(
    obj, articles_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supporte plusieurs formats de embeddings.pkl :
    - tuple/list: (item_ids, emb_matrix)
    - dict: {'item_ids': ..., 'embeddings': ...}
    - DataFrame avec colonne 'article_id' + colonnes vecteur
    - np.ndarray : suppose aligné à l’ordre d’articles_df
    Retourne (item_ids, embeddings) avec ordre **aligné** à item_ids.
    """
    if isinstance(obj, tuple) or isinstance(obj, list):
        item_ids, X = obj
        item_ids = np.asarray(item_ids).astype(int)
        X = np.asarray(X, dtype=float)
        assert X.shape[0] == item_ids.shape[0], (
            "Embeddings et item_ids de tailles différentes"
        )
        return item_ids, X

    if isinstance(obj, dict) and "item_ids" in obj and "embeddings" in obj:
        item_ids = np.asarray(obj["item_ids"]).astype(int)
        X = np.asarray(obj["embeddings"], dtype=float)
        assert X.shape[0] == item_ids.shape[0], (
            "Embeddings et item_ids de tailles différentes"
        )
        return item_ids, X

    if isinstance(obj, pd.DataFrame):
        if "article_id" not in obj.columns:
            raise ValueError("DataFrame embeddings doit contenir 'article_id'")
        df = obj.copy()
        df["article_id"] = df["article_id"].astype(int)
        vec_cols = [c for c in df.columns if c != "article_id"]
        X = df[vec_cols].to_numpy(dtype=float)
        item_ids = df["article_id"].to_numpy(dtype=int)
        return item_ids, X

    # np.ndarray brut -> on l’assume aligné à articles_df
    if isinstance(obj, np.ndarray):
        A = _ensure_numpy_2d(obj)
        item_ids = articles_df["article_id"].to_numpy(dtype=int)
        assert A.shape[0] == item_ids.shape[0], (
            "embeddings np.ndarray doit avoir même nombre de lignes que articles_metadata"
        )
        return item_ids, np.asarray(A, dtype=float)

    raise ValueError(
        "Format embeddings.pkl non supporté (tuple/dict/df/ndarray attendus)."
    )


def _ensure_numpy_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _load_embeddings() -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Retourne (item_ids, embeddings, id2idx)
    """
    raw = _get_blob_bytes(BLOB_CONTAINER_MODELS, "embeddings.pkl")
    obj = pickle.loads(raw)
    return obj


def _build_user_vector_from_clicks(
    user_id: int, embeddings: np.ndarray, clicks_df: pd.DataFrame
) -> Optional[np.ndarray]:
    clicked = (
        clicks_df.loc[clicks_df["user_id"] == int(user_id), "click_article_id"]
        .astype(int)
        .tolist()
    )
    if not clicked:
        return None
    X = embeddings[clicked]  # (k, d)
    v = X.mean(axis=0)  # (d,)
    return v


def _already_seen_articles(user_id: int, clicks_df: pd.DataFrame) -> List[int]:
    return (
        clicks_df.loc[clicks_df["user_id"] == int(user_id), "click_article_id"]
        .astype(int)
        .tolist()
    )


@app.route(route="recommend_get", auth_level=func.AuthLevel.ANONYMOUS)
def recommend_get(req: func.HttpRequest) -> func.HttpResponse:
    """
    GET /api/recommend_get?user_id=50&topn=10&verbose=1&realtime=true
    Si realtime=false, tente de lire precomputed/<user_id>.json, sinon calcule en temps réel.
    """
    try:
        user_id = req.params.get("user_id")
        if not user_id:
            return func.HttpResponse(
                json.dumps({"error": "missing user_id"}),
                status_code=400,
                mimetype="application/json",
            )

        topn = int(req.params.get("topn", "10"))
        verbose = req.params.get("verbose", "0").lower() in ("1", "true", "yes")
        realtime = req.params.get("realtime", "false").lower() in ("1", "true", "yes")

        if not realtime:
            name = f"{int(user_id)}.json"
            bc = (
                _client()
                .get_container_client(BLOB_CONTAINER_PRECOMPUTED)
                .get_blob_client(name)
            )
            if bc.exists():
                data = json.loads(bc.download_blob().content_as_text())
                return func.HttpResponse(
                    json.dumps(data, ensure_ascii=False),
                    mimetype="application/json",
                    status_code=200,
                )

        articles_df = _load_articles_df()
        clicks_df = _load_clicks_df()
        X = _load_embeddings()

        user_vec = _build_user_vector_from_clicks(int(user_id), X, clicks_df)
        already = _already_seen_articles(int(user_id), clicks_df)

        if user_vec is None:
            # ---- Cold-start user -> Popularity fallback ----
            recs = _recommend_popular(
                articles_df=articles_df,
                clicks_df=clicks_df,
                items_to_ignore=already,
                topn=topn,
                verbose=verbose,
            )
            payload = {
                "user_id": int(user_id),
                "source": "realtime-popularity",
                "model": "Popularity",
                "recommendations": json.loads(recs.to_json(orient="records")),
            }
            return func.HttpResponse(
                json.dumps(payload, ensure_ascii=False),
                mimetype="application/json",
                status_code=200,
            )

        model = ContentBasedRecommender(articles_df=articles_df, embeddings=X)
        recs = model.recommend_items(
            user_vec=user_vec, items_to_ignore=already, topn=topn, verbose=verbose
        )

        payload = {
            "user_id": int(user_id),
            "source": "realtime-sklearn",
            "model": model.get_model_name(),
            "recommendations": json.loads(recs.to_json(orient="records")),
        }
        return func.HttpResponse(
            json.dumps(payload, ensure_ascii=False),
            mimetype="application/json",
            status_code=200,
        )

    except Exception as e:
        logging.exception("recommend_get failed")
        return func.HttpResponse(
            json.dumps({"error": str(e)}), mimetype="application/json", status_code=500
        )


@app.route(
    route="precompute_http",
    methods=["GET", "POST"],
    auth_level=func.AuthLevel.ANONYMOUS,
)
def precompute_http(req: func.HttpRequest) -> func.HttpResponse:
    """
    Pré-calcule et push dans Blob `precomputed/{user_id}.json` pour tous les users de clicks_sample.csv
    """
    try:
        articles_df = _load_articles_df()
        clicks_df = _load_clicks_df()
        X = _load_embeddings()
        model = ContentBasedRecommender(articles_df=articles_df, embeddings=X)

        cc = _client().get_container_client(BLOB_CONTAINER_PRECOMPUTED)
        users = sorted(clicks_df["user_id"].unique().tolist())
        count = 0
        for uid in users:
            user_vec = _build_user_vector_from_clicks(int(uid), X, clicks_df)
            if user_vec is None:
                continue
            already = _already_seen_articles(int(uid), clicks_df)
            recs = model.recommend_items(
                user_vec=user_vec, items_to_ignore=already, topn=5, verbose=True
            )
            out = {
                "user_id": int(uid),
                "source": "precomputed-sklearn",
                "model": model.get_model_name(),
                "recommendations": json.loads(recs.to_json(orient="records")),
            }
            cc.upload_blob(
                f"{uid}.json", json.dumps(out, ensure_ascii=False), overwrite=True
            )
            count += 1

        return func.HttpResponse(
            json.dumps({"status": "ok", "precomputed": count}, ensure_ascii=False),
            mimetype="application/json",
            status_code=200,
        )
    except Exception as e:
        logging.exception("precompute_http failed")
        return func.HttpResponse(
            json.dumps({"error": str(e)}), mimetype="application/json", status_code=500
        )


@app.route(route="diag_model", auth_level=func.AuthLevel.ANONYMOUS)
def diag_model(req: func.HttpRequest) -> func.HttpResponse:
    try:
        info = {}
        for name in ("embeddings.pkl",):
            try:
                _ = _get_blob_bytes(BLOB_CONTAINER_MODELS, name)
                info[name] = True
            except Exception:
                info[name] = False
        for name in ("articles_metadata.csv", "clicks_sample.csv"):
            try:
                _ = _get_blob_text(BLOB_CONTAINER_DATA, name)[:64]
                info[name] = True
            except Exception:
                info[name] = False

        ok_load = {}
        try:
            _ = _load_articles_df()
            ok_load["articles_df"] = True
        except Exception as e:
            ok_load["articles_df"] = f"ERR: {e}"
        try:
            _ = _load_clicks_df()
            ok_load["clicks_df"] = True
        except Exception as e:
            ok_load["clicks_df"] = f"ERR: {e}"
        try:
            ids, X, id2idx = _load_embeddings()
            ok_load["embeddings"] = {"n_items": int(X.shape[0]), "dim": int(X.shape[1])}
        except Exception as e:
            ok_load["embeddings"] = f"ERR: {e}"

        return func.HttpResponse(
            json.dumps({"exists": info, "loads": ok_load}, ensure_ascii=False),
            mimetype="application/json",
            status_code=200,
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}), mimetype="application/json", status_code=500
        )
