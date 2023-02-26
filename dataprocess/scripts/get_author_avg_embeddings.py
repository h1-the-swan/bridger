# -*- coding: utf-8 -*-

DESCRIPTION = """Get author average embeddings"""

import pickle
import sys, os, time
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer

try:
    from humanfriendly import format_timespan
except ImportError:

    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from bridger_dataprocess.average_embeddings import (
    get_author_term_matrix,
    get_avg_embeddings,
)

import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)


def main(args):
    outdir = Path(args.outdir)
    if outdir.is_dir():
        raise FileExistsError()
    logger.debug(f"creating directory: {outdir}")
    outdir.mkdir()

    logger.debug(f"loading embeddings file: {args.path_to_embeddings}")
    embeddings = np.load(args.path_to_embeddings)
    logger.debug(f"len(embeddings)=={len(embeddings)}")

    logger.debug(f"loading embeddings_terms file: {args.path_to_terms}")
    embeddings_terms = pd.read_parquet(args.path_to_terms)
    logger.debug(f"len(embeddings_terms)=={len(embeddings_terms)}")

    logger.debug(
        f"loading paper_term_embeddings file: {args.path_to_paper_term_embeddings}"
    )
    df_paper_term_embeddings = pd.read_parquet(args.path_to_paper_term_embeddings)
    df_paper_term_embeddings["s2_id"] = df_paper_term_embeddings["s2_id"].astype(int)
    logger.debug(f"len(df_paper_term_embeddings)=={len(df_paper_term_embeddings)}")

    logger.debug(f"loading paper_authors file: {args.path_to_paper_authors}")
    df_paper_authors = pd.read_parquet(args.path_to_paper_authors)
    logger.debug(f"df_paper_authors.shape=={df_paper_authors.shape}")

    embeddings_terms = embeddings_terms.index.values
    logger.debug("dropping unused embeddings...")
    terms_set = set(df_paper_term_embeddings["embedding_term"].values)
    keep_idx = []
    for i, term in enumerate(embeddings_terms):
        if term in terms_set:
            keep_idx.append(i)
    embeddings_terms = embeddings_terms[keep_idx]
    embeddings = embeddings[keep_idx]
    logger.debug(
        f"len(embeddings): {len(embeddings)}; len(embeddings_terms): {len(embeddings_terms)}"
    )

    logger.debug("getting embeddings_terms_to_idx")
    embeddings_terms_to_idx = {
        val: idx[0] for idx, val in np.ndenumerate(embeddings_terms)
    }
    logger.debug(
        "mapping embeddings_terms_to_idx to paper-to-term-embeddings dataframe"
    )
    df_paper_term_embeddings["term_idx"] = df_paper_term_embeddings[
        "embedding_term"
    ].map(embeddings_terms_to_idx)

    logger.debug("filtering authors...")
    all_paper_ids = df_paper_term_embeddings[
        df_paper_term_embeddings.label.isin(["Task", "Method"])
    ].s2_id.drop_duplicates()
    all_author_ids = df_paper_authors[
        df_paper_authors.PaperId.isin(all_paper_ids)
    ].AuthorId.drop_duplicates()
    df_paper_authors = df_paper_authors[
        df_paper_authors.AuthorId.isin(all_author_ids.values)
    ]
    logger.debug(f"after filtering: df_paper_authors.shape=={df_paper_authors.shape}")

    logger.debug(f"discarding all papers published before {args.min_year}")
    df_paper_authors = df_paper_authors[df_paper_authors.pubYear >= args.min_year]
    logger.debug(
        f"after filtering by year: df_paper_authors.shape=={df_paper_authors.shape}"
    )

    logger.debug(
        f"keeping only authors with at least {args.min_papers} since year {args.min_year}"
    )
    gb = df_paper_authors.groupby("AuthorId").size()
    gb = gb[gb >= args.min_papers]
    df_paper_authors = df_paper_authors[df_paper_authors.AuthorId.isin(gb.index)]
    logger.debug(
        f"after filtering by number of papers: df_paper_authors.shape=={df_paper_authors.shape}"
    )

    logger.debug(f"getting average embeddings for Tasks")
    ssmat_author_term_task = get_author_term_matrix(
        embeddings_terms,
        df_paper_term_embeddings,
        df_paper_authors,
        label="Task",
        dedup_titles=False,
    )
    outfp = outdir.joinpath("ssmat_author_term_task.pickle")
    logger.debug(f"saving matrix to {outfp}")
    outfp.write_bytes(
        pickle.dumps(ssmat_author_term_task, protocol=pickle.HIGHEST_PROTOCOL)
    )
    avg_embeddings_task = get_avg_embeddings(ssmat_author_term_task.mat, embeddings)
    outfp = outdir.joinpath("avg_embeddings_task.pickle")
    logger.debug(f"saving average embeddings to {outfp}")
    avg_embeddings_task.to_pickle(outfp, protocol=pickle.HIGHEST_PROTOCOL)

    logger.debug(f"getting average embeddings for Methods")
    ssmat_author_term_method = get_author_term_matrix(
        embeddings_terms,
        df_paper_term_embeddings,
        df_paper_authors,
        label="Method",
        dedup_titles=False,
    )
    outfp = outdir.joinpath("ssmat_author_term_method.pickle")
    logger.debug(f"saving matrix to {outfp}")
    outfp.write_bytes(
        pickle.dumps(ssmat_author_term_method, protocol=pickle.HIGHEST_PROTOCOL)
    )
    avg_embeddings_method = get_avg_embeddings(ssmat_author_term_method.mat, embeddings)
    outfp = outdir.joinpath("avg_embeddings_method.pickle")
    logger.debug(f"saving average embeddings to {outfp}")
    avg_embeddings_method.to_pickle(outfp, protocol=pickle.HIGHEST_PROTOCOL)


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
    parser.add_argument("path_to_embeddings", help="path to embeddings file (.npy)")
    parser.add_argument("path_to_terms", help="path to terms file (.parquet)")
    parser.add_argument(
        "path_to_paper_term_embeddings",
        help="path to paper-term-embeddings file (.parquet)",
    )
    parser.add_argument(
        "path_to_paper_authors", help="path to paper_authors file (.parquet)"
    )
    parser.add_argument(
        "outdir", help="path to output directory (should not exist, will be created)"
    )
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
