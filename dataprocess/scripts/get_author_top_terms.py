# -*- coding: utf-8 -*-

DESCRIPTION = """get top tasks and methods for authors"""

import sys, os, time, pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from timeit import default_timer as timer
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

import logging
root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

from bridger_dataprocess.matrix import SciSightMatrix

def get_top_terms(ssmat: SciSightMatrix, cast_to_int=True) -> Dict[int, List[str]]:
    df = ssmat.to_df(colname_for_rowdata="author_id", colname_for_coldata="term", colname_for_vals="weight")
    if cast_to_int is True:
        df["author_id"] = df["author_id"].astype(int)
    term_dict = {}  # author_id -> list of top terms
    num_authors = df['author_id'].nunique()
    logger.debug(f"there are {num_authors} authors")
    for author_id, gbdf in df.groupby('author_id'):
        gbdf.sort_values('weight', ascending=False, inplace=True)
        term_dict[author_id] = gbdf['term'].iloc[:5].tolist()
    return term_dict

def main(args):
    outdir = Path(args.outdir)
    for t in ['task', 'method']:
        fp = Path(f"/data/bridger_embeddings/avg_embeddings/ssmat_author_term_{t}.pickle")
        logger.debug(f"loading file: {fp}")
        ssmat: SciSightMatrix = pickle.loads(fp.read_bytes())
        logger.debug(f"getting top terms for all authors for: {t}")
        term_dict = get_top_terms(ssmat)
        outfp = outdir.joinpath(f"author_top_terms_{t}.pickle")
        logger.debug(f"saving to {outfp}")
        outfp.write_bytes(pickle.dumps(term_dict, protocol=pickle.HIGHEST_PROTOCOL))

if __name__ == "__main__":
    total_start = timer()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    logger.info("pid: {}".format(os.getpid()))
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("outdir", help="path to output directory")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))