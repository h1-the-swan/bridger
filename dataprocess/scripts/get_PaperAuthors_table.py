# -*- coding: utf-8 -*-

DESCRIPTION = """Given a JSONL file of papers from S2AG, save a table mapping paper to author and author-sequence-number"""

import fileinput
import json
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


import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)


def yield_paper_author_row(file):
    fp = Path(file)
    with fileinput.hook_compressed(fp, mode="rt") as f:
        for line in f:
            paper = json.loads(line)
            for author_pos, author in enumerate(paper.get("authors", [])):
                yield {
                    "PaperId": paper["corpusid"],
                    "AuthorId": author["authorId"],
                    "AuthorSequenceNumber": author_pos + 1,
                    "pubYear": paper["year"],
                }


def main(args):
    fp = Path(args.input)
    d = []
    logger.debug(f"processing input file: {fp}")
    i = 0
    read_start = timer()
    for row in yield_paper_author_row(fp):
        d.append(row)
        i += 1
        if i in [10, 100, 1000, 5000, 10000, 50000, 100000, 500000] or i % 1000000 == 0:
            logger.debug(f"{i} rows processed")
    logger.debug(
        f"done processing {len(d)} rows. took {format_timespan(timer()-read_start)}"
    )
    df = pd.DataFrame(d)
    logger.debug(
        f"saving dataframe with shape {df.shape} to output file: {args.output}"
    )
    df.to_parquet(args.output)


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
        "input", help="path to input JSONL file (newline separated JSON paper objects)"
    )
    parser.add_argument("output", help="path to output file (parquet format)")
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
