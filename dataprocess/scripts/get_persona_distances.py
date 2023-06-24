# -*- coding: utf-8 -*-

DESCRIPTION = """save recs for personas for specified authors"""

import sys, os, time, pickle, re
from ast import literal_eval
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from typing import Dict, Iterable, List, Set, Sequence, Optional

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
    existing_files = list(dirpath.rglob("persona_recs_*"))
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
        fp_task = Path("/data/bridger_embeddings/avg_embeddings/avg_embeddings_task.pickle")
        logger.debug(f"loading file: {fp_task}")
        self.avg_embeddings_task = pd.read_pickle(fp_task)
        fp_method = Path("/data/bridger_embeddings/avg_embeddings/avg_embeddings_method.pickle")
        logger.debug(f"loading file: {fp_method}")
        self.avg_embeddings_method = pd.read_pickle(fp_method)
        fp_ssmat_task = Path("/data/bridger_embeddings/avg_embeddings/ssmat_author_term_task.pickle")
        logger.debug(f"loading file: {fp_ssmat_task}")
        self.ssmat_author_term_task = pickle.loads(fp_ssmat_task.read_bytes())
        fp_ssmat_method = Path("/data/bridger_embeddings/avg_embeddings/ssmat_author_term_method.pickle")
        logger.debug(f"loading file: {fp_ssmat_method}")
        self.ssmat_author_term_method = pickle.loads(fp_ssmat_method.read_bytes())
        fp_specter = Path(
            "/data/specter_embeddings/avg_specter/average_author_specter_embeddings.pickle"
        )
        logger.debug(f"loading file: {fp_specter}")
        self.avg_embeddings_specter = pd.read_pickle(fp_specter)

    def get_df_dists(
        self,
        focal_embedding_task: np.ndarray,
        focal_embedding_method: np.ndarray,
        focal_embedding_specter: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        return get_df_dists(
            ssmat_author_term_task=self.ssmat_author_term_task,
            avg_embeddings_task=self.avg_embeddings_task,
            ssmat_author_term_method=self.ssmat_author_term_method,
            avg_embeddings_method=self.avg_embeddings_method,
            focal_embedding_task=focal_embedding_task,
            focal_embedding_method=focal_embedding_method,
            avg_embeddings_specter=self.avg_embeddings_specter,
            focal_embedding_specter=focal_embedding_specter,
        )


class PaperAvgEmbeddingGetter:
    def __init__(self) -> None:
        # outdir = Path(args.outdir)
        # if outdir.is_dir():
        #     raise FileExistsError()
        # logger.debug(f"creating directory: {outdir}")
        # outdir.mkdir()

        path_to_embeddings = Path("/data/final-processing-embeddings/embeddings.npy")
        logger.debug(f"loading embeddings file: {path_to_embeddings}")
        self.embeddings = np.load(path_to_embeddings)
        logger.debug(f"len(embeddings)=={len(self.embeddings)}")

        path_to_terms = "/data/final-processing-embeddings/embedding_term_to_id.parquet"
        logger.debug(f"loading embeddings_terms file: {path_to_terms}")
        self.embeddings_terms = pd.read_parquet(path_to_terms)
        logger.debug(f"len(embeddings_terms)=={len(self.embeddings_terms)}")

        path_to_paper_term_embeddings = "/data/final-processing-embeddings/dygie_embedding_term_ids_to_s2_id_scoreThreshold0.90.parquet"
        logger.debug(
            f"loading paper_term_embeddings file: {path_to_paper_term_embeddings}"
        )
        self.df_paper_term_embeddings = pd.read_parquet(path_to_paper_term_embeddings)
        self.df_paper_term_embeddings["s2_id"] = self.df_paper_term_embeddings[
            "s2_id"
        ].astype(int)
        logger.debug(
            f"len(df_paper_term_embeddings)=={len(self.df_paper_term_embeddings)}"
        )

        # logger.debug(f"loading paper_authors file: {args.path_to_paper_authors}")
        # df_paper_authors = pd.read_parquet(args.path_to_paper_authors)
        # logger.debug(f"df_paper_authors.shape=={df_paper_authors.shape}")

        self.embeddings_terms = self.embeddings_terms.index.values
        logger.debug("dropping unused embeddings...")
        terms_set = set(self.df_paper_term_embeddings["embedding_term"].values)
        keep_idx = []
        for i, term in enumerate(self.embeddings_terms):
            if term in terms_set:
                keep_idx.append(i)
        self.embeddings_terms = self.embeddings_terms[keep_idx]
        self.embeddings = self.embeddings[keep_idx]
        logger.debug(
            f"len(embeddings): {len(self.embeddings)}; len(embeddings_terms): {len(self.embeddings_terms)}"
        )

        logger.debug("getting embeddings_terms_to_idx")
        embeddings_terms_to_idx = {
            val: idx[0] for idx, val in np.ndenumerate(self.embeddings_terms)
        }
        logger.debug(
            "mapping embeddings_terms_to_idx to paper-to-term-embeddings dataframe"
        )
        self.df_paper_term_embeddings["term_idx"] = self.df_paper_term_embeddings[
            "embedding_term"
        ].map(embeddings_terms_to_idx)

    def get_focal_embeddings(self, paper_ids):
        this_df = self.df_paper_term_embeddings[
            self.df_paper_term_embeddings["s2_id"].isin(paper_ids)
        ]
        embs = (
            this_df[this_df["label"] == "Task"]["term_idx"]
            .apply(lambda x: self.embeddings[x])
            # .mean(axis=0)
        )
        focal_embedding_task = np.mean(embs.values)
        embs = (
            this_df[this_df["label"] == "Method"]["term_idx"]
            .apply(lambda x: self.embeddings[x])
            # .mean(axis=0)
        )
        focal_embedding_method = np.mean(embs.values)
        return {
            "focal_embedding_task": focal_embedding_task,
            "focal_embedding_method": focal_embedding_method,
        }


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
        # "simspecter",
        "simTask_distMethod",
        "simMethod_distTask",
    ]:
        sorted_df = sort_distance_df(c, df_dists)
        recs[c] = sorted_df.head(50).index.tolist()
    return recs


def run_get_persona_distances(personas):
    personas_file = Path(personas)
    logger.debug(f"loading personas file: {personas_file}")
    try:
        personas_dict: Dict[int, Sequence[int]] = pickle.loads(personas_file.read_bytes())
    except:
        logger.debug("error when loading personas file")
        logger.debug(f"testing for personas_file.exists(): {personas_file.exists()}")
        raise
    author_ids = get_author_ids_from_env()
    dist_getter = DistGetter()
    paper_avg_embedding_getter = PaperAvgEmbeddingGetter()
    outdir = Path("/output")
    max_personas = 5
    for author_id in author_ids:
        this_author = []
        this_author_num_personas = min(len(personas_dict[author_id]), max_personas)
        logger.debug(f"starting getting recs for {this_author_num_personas} personas for author_id: {author_id}")
        for i, persona in enumerate(personas_dict[author_id][:max_personas]):
            logger.debug(f"getting recs for {author_id}P{i}")
            try:
                focal_embeddings = paper_avg_embedding_getter.get_focal_embeddings(persona)
                df_dists = dist_getter.get_df_dists(
                    focal_embedding_task=focal_embeddings["focal_embedding_task"],
                    focal_embedding_method=focal_embeddings["focal_embedding_method"],
                )
                recs = get_recs(df_dists)
                this_author.append(recs)
            except KeyError:
                logger.info(f"SKIPPING author {author_id} persona {i}: not found (KeyError)")
                this_author.append(None)
            except ValueError:
                logger.info(f"SKIPPING author {author_id} persona {i}: (ValueError)")
                this_author.append(None)
        outfp = outdir.joinpath(f"persona_recs_{author_id}.pickle")
        logger.debug(f"saving to {outfp}")
        outfp.write_bytes(pickle.dumps(this_author, protocol=pickle.HIGHEST_PROTOCOL))

def main(args):
    run_get_persona_distances(args.personas)


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
    parser.add_argument("personas", help="path to personas file (.pickle)")
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
