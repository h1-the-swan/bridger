# -*- coding: utf-8 -*-

DESCRIPTION = """taken from script01--- just the part where we combine old and new paper authors tables (this is just a one-off script)"""

import sys, os, time
import json
import gzip
import pickle
from typing import List, Dict, Sequence
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

from parse_ner_preds import run_parse_ner_preds
from normalize_ner_terms import run_normalize_ner_terms
from get_sentence_transformer_embeddings import run_get_sentence_transformer_embeddings
from final_processing_embeddings import run_final_processing_embeddings
from final_processing_terms import run_final_processing_terms
from get_author_avg_embeddings import run_get_author_avg_embeddings
from get_author_avg_specter import run_get_author_avg_specter
from get_author_distances import get_recs
from get_author_top_terms import get_top_terms
from get_personas import run_get_personas
from get_persona_top_terms import get_top_terms_one_persona
from api_download_author_info import run_api_download_author_info

from bridger_dataprocess.s2_api_download import get_batch_paper_data_from_api
from bridger_dataprocess.average_embeddings import get_df_dists_from_author_id, get_author_term_matrix, get_avg_embeddings
from bridger_dataprocess.matrix import SciSightMatrix

import pandas as pd
import numpy as np


def main(args):
    outdir = Path(args.outdir)
    path_to_paper_authors = Path(args.paper_authors)
    path_to_existing_paper_authors = Path(args.existing_paper_authors)
    logger.debug(f"loading paper_authors file: {path_to_paper_authors}")
    df_paper_authors = pd.read_parquet(path_to_paper_authors)
    logger.debug(f"df_paper_authors.shape=={df_paper_authors.shape}")
    logger.debug(
        f"loading existing paper_authors file: {path_to_existing_paper_authors}"
    )
    df_paper_authors_existing = pd.read_parquet(path_to_existing_paper_authors)
    logger.debug(f"df_paper_authors_existing.shape=={df_paper_authors_existing.shape}")
    logger.debug("merging existing and new paper-authors table")
    _to_merge = df_paper_authors_existing[
        ~(df_paper_authors_existing["PaperId"].isin(df_paper_authors["PaperId"]))
    ]
    df_paper_authors = pd.concat([df_paper_authors, _to_merge]).drop_duplicates()
    logger.debug(f"after merge: df_paper_authors.shape=={df_paper_authors.shape}")
    outfp_df_paper_authors = outdir.joinpath('df_paper_authors.parquet')
    logger.debug(f"saving df_paper_authors to file: {outfp_df_paper_authors}")
    df_paper_authors.to_parquet(outfp_df_paper_authors)

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
    parser.add_argument("outdir", help="output base directory")
    parser.add_argument("--paper-authors", help="path to paper-authors file (parquet)")
    parser.add_argument(
        "--existing-paper-authors",
        help="path to paper-authors file (parquet) for existing data from the last update",
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
