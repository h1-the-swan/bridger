# -*- coding: utf-8 -*-

DESCRIPTION = (
    """use agglomerative clustering on specter embeddings to get author personas"""
)

import sys, os, time
import pickle
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer

try:
    from humanfriendly import format_timespan
except ImportError:

    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)


import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def get_clusters(
    author_id: int, df_paper_authors: pd.DataFrame, specter_embeddings: pd.Series
):
    subset = df_paper_authors[df_paper_authors.AuthorId == author_id]
    paper_ids = subset.PaperId.values
    clusterer = AgglomerativeClustering(
        linkage="ward", affinity="euclidean", n_clusters=None, distance_threshold=88
    )
    author_papers_to_specter = specter_embeddings.loc[paper_ids]
    author_embeddings = np.vstack(author_papers_to_specter.values)
    clusterer.fit(author_embeddings)
    vc = pd.Series(clusterer.labels_).value_counts()
    vc = vc[vc >= 5]
    personas = [
        author_papers_to_specter.iloc[clusterer.labels_ == cl].index for cl in vc.index
    ]
    # `personas` is a list of list of paper IDs
    return personas


def load_embeddings(path_to_embeddings, path_to_ids) -> pd.Series:
    logger.debug(f"loading specter embeddings file: {path_to_embeddings}")
    embeddings_raw = np.load(path_to_embeddings)
    logger.debug(f"len(embeddings_raw)=={len(embeddings_raw)}")

    logger.debug(
        f"loading corpus ids file: {path_to_ids}"
    )
    corpus_ids = np.load(path_to_ids)
    logger.debug(f"len(corpus_ids)=={len(corpus_ids)}")

    embeddings = {
        corpus_id: embedding for corpus_id, embedding in zip(corpus_ids, embeddings_raw)
    }
    embeddings = pd.Series(embeddings, name="embedding")
    return embeddings


def run_get_personas(path_to_specter, path_to_specter_embeddings_corpus_ids, df_paper_authors, output, min_papers=5, min_year=2017):
    output_path = Path(output)

    embeddings = load_embeddings(
        path_to_specter, path_to_specter_embeddings_corpus_ids
    )

    df_paper_authors = df_paper_authors.dropna(subset=["AuthorId"])
    df_paper_authors["AuthorId"] = df_paper_authors.AuthorId.astype(int)
    logger.debug(f"df_paper_authors.shape=={df_paper_authors.shape}")

    logger.debug(f"discarding all papers published before {min_year}")
    df_paper_authors = df_paper_authors[df_paper_authors.pubYear >= min_year]
    logger.debug(
        f"after filtering by year: df_paper_authors.shape=={df_paper_authors.shape}"
    )

    logger.debug(
        f"keeping only authors with at least {min_papers} since year {min_year}"
    )
    gb = df_paper_authors.groupby("AuthorId").size()
    gb = gb[gb >= min_papers]
    df_paper_authors = df_paper_authors[df_paper_authors.AuthorId.isin(gb.index)]
    logger.debug(
        f"after filtering by number of papers: df_paper_authors.shape=={df_paper_authors.shape}"
    )

    df_paper_authors = df_paper_authors[df_paper_authors.PaperId.isin(embeddings.index)]

    author_ids = df_paper_authors['AuthorId'].unique()
    logger.debug(f"there are {len(author_ids)} author IDs")

    logger.debug("starting clustering...")
    personas_dict = {}
    for author_id in author_ids:
        try:
            this_personas = get_clusters(
                author_id, df_paper_authors, embeddings
            )
            personas_dict[author_id] = this_personas
            logger.debug(f"found {len(this_personas)} personas for author ID: {author_id}")
        except ValueError:
            personas_dict[author_id] = None
            logger.debug(f"ValueError encountered for author ID: {author_id}. No personas found")
    logger.debug(f"finished clustering {len(personas_dict)} authors. saving personas_dict as pickle file to: {output_path}")
    output_path.write_bytes(pickle.dumps(personas_dict))

def main(args):
    logger.debug(f"loading paper_authors file: {args.path_to_paper_authors}")
    df_paper_authors = pd.read_parquet(args.path_to_paper_authors)
    run_get_personas(args.path_to_specter, args.path_to_specter_embeddings_corpus_ids, df_paper_authors, args.output, args.min_papers, args.min_year)


if __name__ == "__main__":
    total_start = timer()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info(" ".join(sys.argv))
    logger.info("{:%Y-%m-%d %H:%M:%S}".format(datetime.now()))
    logger.info("pid: {}".format(os.getpid()))
    import argparse

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "path_to_specter", help="path to specter embeddings file (.npy)"
    )
    parser.add_argument(
        "path_to_specter_embeddings_corpus_ids",
        help="path to corpus ids file for specter embeddings (.npy)",
    )
    parser.add_argument(
        "path_to_paper_authors", help="path to paper_authors file (.parquet)"
    )
    parser.add_argument("output", help="path to output file (.pickle)")
    parser.add_argument(
        "--min-papers",
        type=int,
        default=5,
        help="authors with fewer than <min_papers> since <min_year> will be excluded (default: 5 papers since 2017)",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2017,
        help="exclude all papers before this year",
    )
    parser.add_argument("--debug", action="store_true", help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug("debug mode is on")
    main(args)
    total_end = timer()
    logger.info(
        "all finished. total time: {}".format(format_timespan(total_end - total_start))
    )
