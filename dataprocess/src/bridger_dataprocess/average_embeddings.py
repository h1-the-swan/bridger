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
from sklearn.metrics.pairwise import cosine_distances

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
    ssmat_paper_term = get_paper_term_matrix(df_paper_term_embeddings, label, terms)

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


def get_avg_specter(
    df_paper_authors, specter_embeddings, weighted: bool = True
) -> pd.Series:
    aembs = df_paper_authors["s2_idx"].apply(lambda x: specter_embeddings[x])
    if weighted is True:
        aembs = aembs * df_paper_authors["score"]
    df_paper_authors["embs"] = aembs
    gb = df_paper_authors.groupby("AuthorId")
    avg_specter = gb["embs"].apply(lambda x: np.mean(x, axis=0))
    return avg_specter


def get_df_dists(
    ssmat_author_term_task: SciSightMatrix,
    avg_embeddings_task: pd.Series,
    ssmat_author_term_method: SciSightMatrix,
    avg_embeddings_method: pd.Series,
    focal_embedding_task: np.ndarray,
    focal_embedding_method: np.ndarray,
    avg_embeddings_specter: Optional[pd.Series] = None,
    focal_embedding_specter: Optional[np.ndarray] = None,
    # get_paper_count=False,
) -> pd.DataFrame:
    # get a dataframe with Task and Method distances given single (focal) embeddings for task, method, and optionally specter
    arr = np.array(avg_embeddings_task.tolist())
    cdist_task = cosine_distances(focal_embedding_task.reshape(1, -1), arr)
    df_cdist_task = pd.DataFrame(
        {"cosine_distance_task": cdist_task[0], "author_idx": range(len(cdist_task[0]))}
    ).sort_values("cosine_distance_task")
    df_cdist_task["AuthorId"] = (
        df_cdist_task["author_idx"]
        .map(lambda x: ssmat_author_term_task.row_labels[x])
        .astype(int)
    )

    arr = np.array(avg_embeddings_method.tolist())
    cdist_method = cosine_distances(focal_embedding_method.reshape(1, -1), arr)
    df_cdist_method = pd.DataFrame(
        {
            "cosine_distance_method": cdist_method[0],
            "author_idx": range(len(cdist_method[0])),
        }
    ).sort_values("cosine_distance_method")
    df_cdist_method["AuthorId"] = (
        df_cdist_method["author_idx"]
        .map(lambda x: ssmat_author_term_method.row_labels[x])
        .astype(int)
    )

    # df_dists = pd.DataFrame(index=gb.index.tolist())
    df_dists = pd.DataFrame()
    df_dists["task_dist"] = df_cdist_task.set_index("AuthorId")["cosine_distance_task"]
    df_dists["method_dist"] = df_cdist_method.set_index("AuthorId")[
        "cosine_distance_method"
    ]
    if avg_embeddings_specter is not None and focal_embedding_specter is not None:
        arr = np.vstack(avg_embeddings_specter.values)
        cdist_specter = cosine_distances(focal_embedding_specter.reshape(1, -1), arr)
        df_cdist_specter = pd.DataFrame(
            cdist_specter[0],
            index=avg_embeddings_specter.index,
            columns=["cosine_distance_specter"],
        )

        df_dists["specter_dist"] = df_cdist_specter["cosine_distance_specter"]

    # if get_paper_count is True:
    #     df_dists["paper_count"] = (
    #         df_paper_authors[["AuthorId", "PaperId"]]
    #         .dropna()
    #         .drop_duplicates()
    #         .groupby("AuthorId")
    #         .size()
    #     )

    return df_dists


def get_df_dists_from_author_id(
    author_id: Union[str, int],
    ssmat_author_term_task: SciSightMatrix,
    avg_embeddings_task: pd.Series,
    ssmat_author_term_method: SciSightMatrix,
    avg_embeddings_method: pd.Series,
    avg_embeddings_specter: Optional[pd.Series] = None,
) -> pd.DataFrame:
    # get a dataframe with Task and Method distances for a single focal author
    author_idx = ssmat_author_term_task.row_label_to_row_idx[str(author_id)]
    focal_embedding_task = avg_embeddings_task.loc[author_idx]

    author_idx = ssmat_author_term_method.row_label_to_row_idx[str(author_id)]
    focal_embedding_method = avg_embeddings_method.loc[author_idx]

    if avg_embeddings_specter is not None:
        focal_embedding_specter = avg_embeddings_specter.loc[int(author_id)]
    else:
        focal_embedding_specter = None

    df_dists = get_df_dists(
        ssmat_author_term_task=ssmat_author_term_task,
        avg_embeddings_task=avg_embeddings_task,
        ssmat_author_term_method=ssmat_author_term_method,
        avg_embeddings_method=avg_embeddings_method,
        focal_embedding_task=focal_embedding_task,
        focal_embedding_method=focal_embedding_method,
        avg_embeddings_specter=avg_embeddings_specter,
        focal_embedding_specter=focal_embedding_specter,
    )
    return df_dists.drop(author_id)
