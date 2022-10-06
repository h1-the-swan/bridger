# -*- coding: utf-8 -*-

DESCRIPTION = """Parse NER predictions from results file in one directory"""

import json
import sys, os, time
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from typing import Dict, List, Optional, Union

try:
    from humanfriendly import format_timespan
except ImportError:

    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)


import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

import pandas as pd

from bridger_dataprocess.util_ner import NERDoc, yield_ner_row


def load_jsonl_file(filepath: Union[Path, str]) -> Optional[List[Dict]]:
    fp = Path(filepath)
    contents = fp.read_text()
    if "predicted_ner" not in contents:
        return None
    data = [json.loads(line) for line in contents.split("\n") if line]
    return data


def main(args):
    rows = []
    dirpath = Path(args.input_dir)
    ext = args.ext.strip(".")
    if args.output.split(".")[-1] not in ["parquet", "csv"]:
        raise ValueError(f"invalid output filename: {args.output}")
    #  ner_docs = []
    for fp in dirpath.glob(f"*.{ext}"):
        ner_data = load_jsonl_file(fp)
        if not ner_data:
            logger.debug(f"skipping file: {fp}")
            continue
        this_file_docs = [NERDoc(p) for p in ner_data]
        for doc in this_file_docs:
            for row in yield_ner_row(doc):
                rows.append(row)
    df = pd.DataFrame(rows)
    logger.debug(f"writing to file: {args.output}")
    if args.output.endswith("parquet"):
        df.to_parquet(args.output)
    elif args.output.endswith("csv"):
        df.to_csv(args.output)
    else:
        # this shouldn't happen
        raise ValueError(f"invalid output filename: {args.output}")


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
    parser.add_argument("input_dir", help="directory with .json files")
    parser.add_argument(
        "output",
        help="path to output file (default: /output/terms.parquet). .parquet and .csv are supported",
    )
    parser.add_argument(
        "--ext",
        default=".json",
        help="use this extension to look for input files (default: '.json'",
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
