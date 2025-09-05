import pandas as pd
import numpy as np

import logging
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

    counts = (
        clicks_df["click_article_id"]
        .astype(int)
        .value_counts()
        .sort_values(ascending=False)
    )
    _cache["_pop_counts"] = counts
    return counts


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
    keep = counts[~counts.index.isin(ignore)].head(topn)

    if keep.empty:
        return pd.DataFrame(columns=["article_id", "score"])

    scores = keep.astype(float)
    maxc = float(scores.iloc[0]) if len(scores) else 1.0
    scores = scores / (maxc if maxc > 0 else 1.0)

    recs = pd.DataFrame({"article_id": keep.index.astype(int), "score": scores.values})

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
        self._emb = embeddings  # (n_items, d)
        self._item_ids = np.asarray(
            articles_df.article_id.unique(), dtype=int
        )  # (n_items,)

    def get_model_name(self) -> str:
        return self.MODEL_NAME

    def recommend_items(
        self, user_vec: np.ndarray, items_to_ignore: List[int], topn: int, verbose: bool
    ) -> pd.DataFrame:
        user_vec = np.asarray(user_vec, dtype=float).reshape(1, -1)
        sims = cosine_similarity(user_vec, self._emb)[0]  # (n_items,)

        # Filtrage des déjà vus
        ignore = set(int(x) for x in (items_to_ignore or []))
        pairs = [
            (int(a), float(s))
            for a, s in zip(self._item_ids, sims)
            if int(a) not in ignore
        ]

        # tri décroissant
        pairs.sort(key=lambda t: -t[1])
        pairs = pairs[:topn]

        recs = pd.DataFrame(pairs, columns=["article_id", "score"])
        if verbose:
            recs = recs.merge(
                self.items_df, how="left", left_on="article_id", right_on="article_id"
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
        logger.info(recs)
        return recs
