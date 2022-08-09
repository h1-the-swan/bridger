# -*- coding: utf-8 -*-

DESCRIPTION = """Get the abstracts for the computer science papers found in a previously run script.
Save them as a gzipped JSONL file."""

import sys, os, time
import json
import gzip
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from typing import Dict, Set, Union

try:
    from humanfriendly import format_timespan
except ImportError:

    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)


import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)


def get_paper_ids(fp: Union[Path, str]) -> Set[int]:
    corpusids = []
    with gzip.open(fp, 'rt') as f:
        for line in f:
            if line:
                p = json.loads(line)
                corpusids.append(p['corpusid'])
    logger.debug(f"found {len(corpusids)} paper IDs")
    corpusids = set(corpusids)
    logger.debug(f"{len(corpusids)} paper IDs after deduplication")
    return corpusids


def main(args):
    input_dirpath = Path(args.input)
    papers_fp = Path(args.papers_file)
    outfp = Path(args.output)
    logger.debug(f"getting paper IDs for computer science papers from file {papers_fp}")
    paper_ids = get_paper_ids(papers_fp)
    logger.debug(f"opening file for write: {outfp}")
    outf = gzip.open(outfp, mode="wt")
    num_lines_written = 0
    try:
        for fp in input_dirpath.glob("*.gz"):
            logger.debug(f"processing input file: {fp}")
            with gzip.open(fp, mode="rt") as f:
                line_num = 0
                this_file_num_lines_written = 0
                for line in f:
                    if line:
                        record = json.loads(line)
                        if record['corpusid'] in paper_ids:
                            outf.write(line)
                            num_lines_written += 1
                            this_file_num_lines_written += 1
                    line_num += 1
            logger.debug(
                f"finished processing file: {fp}. {line_num} lines processed. {this_file_num_lines_written} lines written."
            )
            logger.debug(f"{num_lines_written} lines written total so far.")
    finally:
        logger.debug(f"closing output file. {num_lines_written} lines written.")
        outf.close()


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
        "input",
        help="input directory with abstracts in gzipped JSONL files (with extension .gz)",
    )
    parser.add_argument(
        "papers_file",
        help="path to papers file in gzipped JSONL format",
    )
    parser.add_argument("output", help="output filename (.gz)")
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
