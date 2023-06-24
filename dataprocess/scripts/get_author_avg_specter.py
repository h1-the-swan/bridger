# -*- coding: utf-8 -*-

DESCRIPTION = """Get author average specter embeddings"""

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

from bridger_dataprocess.average_embeddings import get_avg_specter, get_score_column

import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)


def run_get_author_avg_specter(embeddings, corpus_ids, df_paper_authors, outfp, min_papers, min_year):
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

    logger.debug("getting score column")
    df_paper_authors["score"] = get_score_column(df_paper_authors)

    logger.debug("mapping PaperId to s2_id")
    s2_to_idx = {val: idx[0] for idx, val in np.ndenumerate(corpus_ids)}
    df_paper_authors["s2_idx"] = df_paper_authors["PaperId"].map(s2_to_idx)
    df_paper_authors = df_paper_authors.dropna(subset=['s2_idx'])
    df_paper_authors['s2_idx'] = df_paper_authors['s2_idx'].astype(int)

    logger.debug(f"getting average specter embeddings")
    avg_specter = get_avg_specter(df_paper_authors, embeddings)
    logger.debug(f"avg_specter.shape: {avg_specter.shape}")
    logger.debug(f"saving average embeddings to {outfp}")
    avg_specter.to_pickle(outfp, protocol=pickle.HIGHEST_PROTOCOL)


def main(args):
    outdir = Path(args.outdir)
    if outdir.is_dir():
        raise FileExistsError()
    logger.debug(f"creating directory: {outdir}")
    outdir.mkdir()
    outfp = outdir.joinpath("average_author_specter_embeddings.pickle")

    logger.debug(f"loading specter embeddings file: {args.path_to_specter}")
    embeddings = np.load(args.path_to_specter)
    logger.debug(f"len(embeddings)=={len(embeddings)}")

    logger.debug(
        f"loading corpus ids file: {args.path_to_specter_embeddings_corpus_ids}"
    )
    corpus_ids = np.load(args.path_to_specter_embeddings_corpus_ids)
    logger.debug(f"len(corpus_ids)=={len(corpus_ids)}")

    logger.debug(f"loading paper_authors file: {args.path_to_paper_authors}")
    df_paper_authors = pd.read_parquet(args.path_to_paper_authors)
    df_paper_authors = df_paper_authors.dropna(subset=["AuthorId"])
    df_paper_authors["AuthorId"] = df_paper_authors.AuthorId.astype(int)
    logger.debug(f"df_paper_authors.shape=={df_paper_authors.shape}")

    run_get_author_avg_specter(embeddings, corpus_ids, df_paper_authors, outfp, args.min_papers, args.min_year)


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
