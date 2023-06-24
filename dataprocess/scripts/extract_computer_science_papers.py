# -*- coding: utf-8 -*-

DESCRIPTION = """Get the papers that have the category "Computer Science" (as a gzipped JSONL file)"""

import sys, os, time
import json
import gzip
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from typing import Dict, Container, Optional, Union, Iterable

try:
    from humanfriendly import format_timespan
except ImportError:

    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)


import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)


def check_category(paper: Dict, name: str) -> bool:
    fields = paper.get("s2fieldsofstudy")
    if fields:
        return any([fos["category"].lower() == name.lower() for fos in fields])
    else:
        return False


def process_one_line(
    line: str,
    ignore_ids: Container[int] = set(),
    category_name: str = "computer science",
) -> Union[Dict, None]:
    record = json.loads(line)
    if (record["corpusid"] not in ignore_ids) and (
        check_category(record, category_name)
    ):
        return record
    else:
        return None


def run_extract_computer_science_papers(
    input_dirpath: Union[str, Path],
    outfp: Union[str, Path],
    ignore_ids: Container[int] = set(),
):
    input_dirpath = Path(input_dirpath)
    outfp = Path(outfp)
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
                        record = process_one_line(
                            line, ignore_ids, category_name="computer science"
                        )
                        if record is not None:
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


def main(args):
    ignore_ids = set()
    if args.existing:
        fp = Path(args.existing)
        logger.debug(f"loading paper ids to ignore from file: {fp}")
        with fp.open() as f:
            for line in f:
                ignore_ids.add(int(line.strip()))
        logger.debug(f"found {len(ignore_ids)} paper IDs to ignore")
    run_extract_computer_science_papers(args.input, args.output, existing=ignore_ids)


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
        help="input directory with papers in gzipped JSONL files (with extension .gz)",
    )
    parser.add_argument("output", help="output filename (.gz)")
    parser.add_argument(
        "--existing",
        help="path to file containing paper ids to ignore, because they have already been processed (newline separated text file with integer IDs)",
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
