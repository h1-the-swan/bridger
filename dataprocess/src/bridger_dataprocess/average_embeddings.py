# -*- coding: utf-8 -*-

# ORIGINALLY COPIED FROM THE `collabnetworks` REPO. THIS NEEDS UPDATING

DESCRIPTION = """average embeddings for authors"""

from pathlib import Path
from typing import Optional, List, Dict, Union

import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

import pandas as pd
import numpy as np

# from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix, save_npz

from .core import get_score_column, map_label_to_idx
# from .data_helper import DataHelper
from .matrix import SciSightMatrix


class AverageEmbeddings:
    """Store average embeddings (e.g., for authors)"""

    def __init__(
        self,
        avg_embeddings: pd.Series,
        ids: List[Union[int, str]],
        id_map: Optional[Dict[Union[int, str], int]],
    ) -> None:
        self.avg_embeddings = avg_embeddings
        self.ids = ids
        self.id_map = id_map

    @classmethod
    def load(
        cls, fname: Union[Path, str], fname_ids: Optional[Union[Path, str]] = None
    ) -> None:
        logger.debug(f"loading average embeddings from file: {fname}")
        avg_embeddings = pd.read_pickle(fname)  # pd.Series
        if fname_ids is None:
            # assume the index of avg_embeddings contains the ids
            ids = avg_embeddings.index.tolist()
        else:
            logger.debug(f"loading ids from file: {fname_ids}")
            ids = np.load(fname_ids)
        logger.debug("mapping labels to index")
        id_map = map_label_to_idx(ids)
        return cls(avg_embeddings, ids, id_map)


def get_paper_term_matrix(
    df: pd.DataFrame,
    label: str,
    terms: np.ndarray,
) -> SciSightMatrix:
    _df = df.loc[df["label"] == label, :]
    val_col = None
    description = "rows are papers, columns are terms, values are ones"

    return SciSightMatrix.from_df(
        _df,
        "s2_id",
        "embedding_term",
        val_col,
        col_labels=terms,
        description=description,
    )


def get_avg_embeddings(
    mat: csr_matrix, embeddings: np.ndarray, weighted: bool = True
) -> pd.Series:
    row_idx, col_idx = mat.nonzero()
    df = pd.DataFrame({"author_idx": row_idx, "term_idx": col_idx, "weight": mat.data})
    aembs = df["term_idx"].apply(lambda x: embeddings[x])
    if weighted is True:
        aembs = aembs * df["weight"]
    df["embs"] = aembs
    gb = df.groupby("author_idx")
    avg_embeddings = gb["embs"].apply(lambda x: np.mean(x, axis=0))
    return avg_embeddings


def get_author_term_matrix(
    terms: np.ndarray,
    df_paper_term_embeddings: pd.DataFrame,
    paa: pd.DataFrame,
    label: str,
    weighted: bool = True,
    dedup_titles: bool = True,
) -> SciSightMatrix:
    # papers = data_helper.df_ner.PaperId.unique()
    # ssmat_paper_term = get_paper_term_matrix(data_helper.df_ner, label, papers, terms)
    if dedup_titles is True:
        # TODO
        from collabnetworks.util import drop_duplicate_titles

        logger.debug("dropping duplicate paper titles")
        dedup = drop_duplicate_titles(data_helper.mag_data.papers)
        df_paper_term_embeddings = df_paper_term_embeddings[
            df_paper_term_embeddings["PaperId"].isin(dedup["PaperId"])
        ]

    # TODO
    ssmat_paper_term = get_paper_term_matrix(
        df_paper_term_embeddings, label, terms
    )

    papers = ssmat_paper_term.row_labels
    paa = paa.loc[paa["PaperId"].isin(papers)]
    paa.drop_duplicates(subset=["PaperId", "AuthorId"], inplace=True)
    if weighted is True:
        scores = get_score_column(paa)
        paa["score"] = scores
        val_col = "score"
    else:
        val_col = None

    ssmat_author_paper = SciSightMatrix.from_df(
        paa,
        "AuthorId",
        "PaperId",
        val_col,
        col_labels=papers,
        description="rows are authors, columns are papers, values are term relevance scores",
    )

    logger.debug(ssmat_author_paper.mat.shape)
    logger.debug(ssmat_paper_term.mat.shape)

    mat_author_term = ssmat_author_paper.mat @ ssmat_paper_term.mat

    ssmat_author_term = SciSightMatrix.from_matrix(
        mat_author_term,
        ssmat_author_paper.row_labels,
        ssmat_paper_term.col_labels,
        description=f"rows are authors, columns are terms ({label}), values are scores",
    )

    return ssmat_author_term
