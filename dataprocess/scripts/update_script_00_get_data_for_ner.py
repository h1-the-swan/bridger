# -*- coding: utf-8 -*-

DESCRIPTION = """First step of update script to get data for PL-marker NER"""

import sys, os, time
import json
import gzip
import shutil
from typing import List, Union, Set
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

import pandas as pd
import numpy as np

from bulk_download import run_bulk_download
from extract_computer_science_papers import run_extract_computer_science_papers
from api_download_abstracts import run_api_download_abstracts
from format_data_for_ner import run_format_data_for_ner
from get_PaperAuthors_table import run_get_PaperAuthors_table


def find_papers_directory_in_bulk_download(dirpath: Path):
    dirpath = Path(dirpath)
    for hit in dirpath.rglob("papers"):
        if hit.is_dir():
            return hit
    raise RuntimeError(f"could not find papers directory in {dirpath}")


def identify_papers_to_update(papers_file: Path) -> List:
    update_papers = []
    with gzip.open(papers_file, mode="rt") as f:
        for i, line in enumerate(f):
            p = json.loads(line)
            update_papers.append(p["corpusid"])
    return update_papers


def get_ignore_ids(path_to_df: Union[Path, str]) -> Set[int]:
    logger.debug(f"getting paper IDs to ignore from file: {path_to_df}")
    df = pd.read_parquet(path_to_df)
    for column_name in ["s2_id", "PaperId"]:
        try:
            ignore_ids = df[column_name].dropna().drop_duplicates().tolist()
            ignore_ids = set(ignore_ids)  # better to convert this to set. much faster
            logger.debug(f"found {len(ignore_ids)} paper IDs to ignore")
            return ignore_ids
        except KeyError:
            logger.warning(f"could not find paper IDs using column name: {column_name}")
    logger.error("failed to find any paper IDs to ignore. returning empty set")
    return set()
        


def main(args):
    outdir = Path(args.outdir)
    bulk_download_dir = outdir.joinpath("bulk_papers_download")
    bulk_download_dir.mkdir(parents=True)
    run_bulk_download(bulk_download_dir, "papers")
    papers_dir = find_papers_directory_in_bulk_download(bulk_download_dir)
    papers_file = outdir.joinpath("computer_science_papers_update.gz")
    ignore_ids = get_ignore_ids(args.existing)
    run_extract_computer_science_papers(papers_dir, papers_file, ignore_ids=ignore_ids)
    update_papers = identify_papers_to_update(papers_file)
    abstracts_file = outdir.joinpath("update_papers_and_abstracts.parquet")
    logger.debug(f"len(update_papers): {len(update_papers)}")
    run_api_download_abstracts(update_papers, abstracts_file)
    formatted_titles_abstracts_file = outdir.joinpath(
        "titles_abstracts_update_plmarker_scierc.json"
    )
    run_format_data_for_ner(abstracts_file, formatted_titles_abstracts_file, chunksize=100000)
    paperauthors_file = outdir.joinpath("computer_science_paper_authors_update.parquet")
    run_get_PaperAuthors_table(papers_file, paperauthors_file)
    logger.debug(f"deleting directory: {bulk_download_dir}")
    shutil.rmtree(bulk_download_dir)


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
    parser.add_argument(
        "--existing", help='path to file (parquet): dataframe with column "s2_id" or "PaperId"'
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
