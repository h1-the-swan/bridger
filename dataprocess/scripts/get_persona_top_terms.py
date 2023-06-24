# -*- coding: utf-8 -*-

DESCRIPTION = """get top tasks and methods for personas"""

import sys, os, time, pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Set, Sequence, Optional
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


def get_top_terms_one_persona(persona_papers: List, df: pd.DataFrame) -> List[str]:
    df_this_persona = df[df["s2_id"].isin(persona_papers)]
    this_persona_terms = df_this_persona["term_display"].value_counts()
    this_persona_top_terms = this_persona_terms.index[:5].tolist()
    return this_persona_top_terms


def get_top_terms_personas(
    df: pd.DataFrame, personas_dict, persona_recs
) -> Dict[int, List[List[str]]]:
    term_dict = (
        {}
    )  # author_id -> list of lists, inner lists are terms for a persona, one list per persona
    for author_id, this_author_recs in persona_recs.items():
        term_dict[author_id] = []
        for i, rec in enumerate(this_author_recs):
            if rec is None or len(rec) == 0:
                term_dict[author_id].append(None)
            else:
                persona_papers = personas_dict[author_id][i]
                this_persona_top_terms = get_top_terms_one_persona(persona_papers, df)
                term_dict[author_id].append(this_persona_top_terms)
    return term_dict


def main(args):
    personas_file = Path(args.personas)
    logger.debug(f"loading personas file: {personas_file}")
    personas_dict: Dict[int, Sequence[int]] = pickle.loads(personas_file.read_bytes())
    logger.debug(f"personas_dict has len: {len(personas_dict)}")
    persona_recs_file = Path(args.persona_recs)
    logger.debug(f"loading persona recommendations file: {persona_recs_file}")
    persona_recs: Dict[int, List] = pickle.loads(persona_recs_file.read_bytes())
    logger.debug(f"persona_recs has len: {len(persona_recs)}")
    terms_file = Path(args.terms)
    logger.debug(f"loading paper to terms dataframe: {terms_file}")
    df_terms = pd.read_parquet(terms_file)
    logger.debug(f"dataframe has shape: {df_terms.shape}")
    outdir = Path(args.outdir)
    for t in ["task", "method"]:
        subset = df_terms[df_terms["label"].str.lower() == t]
        logger.debug(f"getting top terms for all personas for: {t}")
        term_dict = get_top_terms_personas(subset, personas_dict, persona_recs)
        outfp = outdir.joinpath(f"persona_top_terms_{t}.pickle")
        logger.debug(f"saving to {outfp}")
        outfp.write_bytes(pickle.dumps(term_dict, protocol=pickle.HIGHEST_PROTOCOL))


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
    parser.add_argument(
        "persona_recs", help="path to persona recommendations file (.pickle)"
    )
    parser.add_argument("terms", help="path to file - papers to terms (.parquet)")
    parser.add_argument("outdir", help="path to output directory")
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
