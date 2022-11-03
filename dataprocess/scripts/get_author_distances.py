# -*- coding: utf-8 -*-

DESCRIPTION = """save distance dataframes for specified authors"""

import sys, os, time, pickle, re
from ast import literal_eval
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from typing import Dict, Iterable, List, Set

try:
    from humanfriendly import format_timespan
except ImportError:

    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)


import pandas as pd
import numpy as np

from bridger_dataprocess.average_embeddings import get_df_dists


import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)


def get_existing_output_files() -> List[str]:
    dirpath = Path("/output")
    existing_files = list(dirpath.rglob("recs_*"))
    return [fp.name for fp in existing_files]


def get_author_id_from_fname(fname: str) -> int:
    author_id_str = re.search(r"(\d+)", fname).group(1)
    return int(author_id_str)


def remove_existing(author_ids: Iterable[int]) -> Set[int]:
    ignore_files = get_existing_output_files()
    ignore_author_ids = set(get_author_id_from_fname(fname) for fname in ignore_files)
    return set(author_ids).difference(ignore_author_ids)


def get_author_ids_from_env() -> Set[int]:
    author_ids = os.getenv("AUTHOR_IDS")
    if isinstance(author_ids, str):
        author_ids = literal_eval(author_ids)
    author_ids = [int(idx) for idx in author_ids]
    return remove_existing(author_ids)


class DistGetter:
    def __init__(self) -> None:
        fp_task = Path("/data/avg_embeddings/avg_embeddings_task.pickle")
        logger.debug(f"loading file: {fp_task}")
        self.avg_embeddings_task = pd.read_pickle(fp_task)
        fp_method = Path("/data/avg_embeddings/avg_embeddings_method.pickle")
        logger.debug(f"loading file: {fp_method}")
        self.avg_embeddings_method = pd.read_pickle(fp_method)
        fp_ssmat_task = Path("/data/avg_embeddings/ssmat_author_term_task.pickle")
        logger.debug(f"loading file: {fp_ssmat_task}")
        self.ssmat_author_term_task = pickle.loads(fp_ssmat_task.read_bytes())
        fp_ssmat_method = Path("/data/avg_embeddings/ssmat_author_term_method.pickle")
        logger.debug(f"loading file: {fp_ssmat_method}")
        self.ssmat_author_term_method = pickle.loads(fp_ssmat_method.read_bytes())
        fp_specter = Path("/specter/average_author_specter_embeddings.pickle")
        logger.debug(f"loading file: {fp_specter}")
        self.avg_embeddings_specter = pd.read_pickle(fp_specter)

    def get_df_dists_for_author(self, author_id) -> pd.DataFrame:
        return get_df_dists(
            author_id,
            ssmat_author_term_task=self.ssmat_author_term_task,
            ssmat_author_term_method=self.ssmat_author_term_method,
            avg_embeddings_task=self.avg_embeddings_task,
            avg_embeddings_method=self.avg_embeddings_method,
            avg_embeddings_specter=self.avg_embeddings_specter,
        )


def sort_distance_df(
    c: str,
    df_distances: pd.DataFrame,
) -> pd.DataFrame:
    """Given c: a condition in ['simTask', 'simMethod', 'simspecter', 'simTask_distMethod', 'simMethod_distTask']
    sort the df_distances dataframe appropriately for the condition, and return the sorted dataframe.

    Args:
        c (str): condition
        df_distances pd.DataFrame: distances dataframe

    Returns:
        pd.DataFrame: sorted distances dataframe
    """
    if c == "simTask_distMethod":
        _df = (
            df_distances.sort_values("task_dist")
            .head(1000)
            .sort_values("method_dist", ascending=False)
        )
    elif c == "simMethod_distTask":
        _df = (
            df_distances.sort_values("method_dist")
            .head(1000)
            .sort_values("task_dist", ascending=False)
        )
    else:
        k = c.replace("sim", "")  # will be "Task", "Method", or "specter"
        _df = df_distances.sort_values(f"{k.lower()}_dist")
    return _df


def get_recs(df_dists: pd.DataFrame) -> Dict:
    recs = {}
    for c in [
        "simTask",
        "simMethod",
        "simspecter",
        "simTask_distMethod",
        "simMethod_distTask",
    ]:
        sorted_df = sort_distance_df(c, df_dists)
        recs[c] = sorted_df.head(50).index.tolist()
    return recs


def main(args):
    author_ids = get_author_ids_from_env()
    dist_getter = DistGetter()
    outdir = Path("/output")
    for author_id in author_ids:
        logger.debug(f"getting df_dists for author_id: {author_id}")
        df_dists = dist_getter.get_df_dists_for_author(author_id)
        recs = get_recs(df_dists)
        outfp = outdir.joinpath(f"recs_{author_id}.pickle")
        logger.debug(f"saving to {outfp}")
        outfp.write_bytes(pickle.dumps(recs, protocol=pickle.HIGHEST_PROTOCOL))


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
