import traceback

import pandas as pd
import numpy as np

import logging, os
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("azure-functions")

_cache: Dict[str, object] = {
    "articles_df": None,  # pd.DataFrame
    "clicks_df": None,  # pd.DataFrame
    "item_ids": None,  # np.ndarray[int]
    "embeddings": None,  # np.ndarray[float]
    "id2idx": None,  # Dict[int,int]
}

DTYPE = np.float32
TOPK_CHUNK = int(os.getenv("RECO_TOPK_CHUNK", "50000"))


# =========================
# Popularity (baseline)
# =========================
def _get_popularity_counts(clicks_df: pd.DataFrame) -> pd.Series:
    """
    Retourne une Series indexée par article_id avec le nb de clics (desc).
    Mise en cache process-lifetime.
    """
    if "_pop_counts" in _cache and _cache["_pop_counts"] is not None:
        return _cache["_pop_counts"]  # type: ignore

    item_popularity_df = (clicks_df
                          .groupby('click_article_id')
                          .agg(clicks=('user_id', 'size'),unique_users=('user_id', 'nunique'))
                          .reset_index()
                          .rename(columns={'click_article_id': 'article_id'})
                          )

    item_popularity_df['depth'] = item_popularity_df['clicks'] / item_popularity_df['unique_users'].clip(lower=1)  # >1 si des users cliquent plusieurs fois

    for col in ['clicks', 'unique_users', 'depth']:
        item_popularity_df[f'{col}_log'] = np.log1p(item_popularity_df[col])
        item_popularity_df[f'{col}_pct'] = item_popularity_df[f'{col}_log'].rank(pct=True)

    # pondérations
    w_clicks, w_users, w_depth = 0.3, 0.5, 0.2
    item_popularity_df['score'] = (w_users * item_popularity_df['unique_users_pct']
                                   + w_clicks * item_popularity_df['clicks_pct']
                                   + w_depth * item_popularity_df['depth_pct'])

    item_popularity_df_sorted = item_popularity_df.sort_values('score', ascending=False)

    _cache["_pop_counts"] = item_popularity_df_sorted
    return item_popularity_df_sorted


def _recommend_popular(
    articles_df: pd.DataFrame,
    clicks_df: pd.DataFrame,
    items_to_ignore: List[int] | None,
    topn: int,
    verbose: bool,
) -> pd.DataFrame:
    """
    Recommande les articles les plus populaires non vus.
    Score = nombre de clics (normalisé 0..1 pour lisibilité).
    """
    counts = _get_popularity_counts(clicks_df)  # Series: idx=article_id, val=count
    ignore = set(int(x) for x in (items_to_ignore or []))
    recs = counts[~counts.index.isin(ignore)].head(topn)

    if recs.empty:
        return pd.DataFrame(columns=["article_id", "score"])

    if verbose:
        recs = recs.merge(articles_df, on="article_id", how="left")
        cols_pref = [
            "article_id",
            "score",
            "category_id",
            "created_at_ts",
            "publisher_id",
            "words_count",
        ]
        cols = [c for c in cols_pref if c in recs.columns]
        recs = recs[cols]

    return recs


# =========================
# Modèle Content-Based (sklearn)
# =========================
class ContentBasedRecommender:
    MODEL_NAME = "Content-Based"

    def __init__(self, articles_df: pd.DataFrame, embeddings: np.ndarray):
        self.items_df = articles_df
        self._item_ids = np.asarray(articles_df.article_id.unique(), dtype=int)

        E = np.asarray(embeddings, dtype=DTYPE, order="C")
        if E.ndim != 2:
            E = E.reshape(E.shape[0], -1)

        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self._emb = E / norms

        self._id2row = {int(a): i for i, a in enumerate(self._item_ids)}


    def get_model_name(self) -> str:
        return self.MODEL_NAME

    def recommend_items(
        self, user_vec: np.ndarray, items_to_ignore: List[int], topn: int, verbose: bool
    ) -> pd.DataFrame:
        # user_vec -> float32 normalisé
        uv = np.asarray(user_vec, dtype=DTYPE).reshape(-1)
        uv_norm = np.linalg.norm(uv)
        if uv_norm == 0 or not np.isfinite(uv_norm):
            logger.warning("CBF:user_vec zero-norm → no recommendations")
            return pd.DataFrame(columns=["article_id", "score"])
        uv /= uv_norm

        n = self._emb.shape[0]
        ignore_mask = np.zeros(n, dtype=bool)
        for a in (items_to_ignore or []):
            idx = self._id2row.get(int(a))
            if idx is not None:
                ignore_mask[idx] = True

        best_idx = np.empty(0, dtype=np.int32)
        best_scores = np.empty(0, dtype=DTYPE)

        chunk = TOPK_CHUNK

        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            sim = self._emb[i:j].dot(uv)  # float32, (chunk,)
            if ignore_mask[i:j].any():
                sim[ignore_mask[i:j]] = -np.inf

            k_local = min(max(topn * 3, topn), j - i)
            part = np.argpartition(sim, -k_local)[-k_local:]
            cand_idx = (i + part).astype(np.int32, copy=False)
            cand_scores = sim[part]

            best_idx = np.concatenate([best_idx, cand_idx])
            best_scores = np.concatenate([best_scores, cand_scores])

            if best_idx.size > topn * 10:
                gk = np.argpartition(best_scores, -topn)[-topn:]
                best_idx, best_scores = best_idx[gk], best_scores[gk]

        if best_idx.size == 0:
            return pd.DataFrame(columns=["article_id", "score"])

        gk = np.argpartition(best_scores, -topn)[-topn:]
        order = gk[np.argsort(-best_scores[gk])]
        sel_idx = best_idx[order]
        sel_scores = best_scores[order].astype(float, copy=False)

        sel_ids = self._item_ids[sel_idx]
        recs = pd.DataFrame({"article_id": sel_ids, "score": sel_scores})

        if verbose:
            recs = recs.merge(
                self.items_df, how="left", on="article_id"
            )[
                [
                    "article_id",
                    "score",
                    "category_id",
                    "created_at_ts",
                    "publisher_id",
                    "words_count",
                ]
            ]
        return recs