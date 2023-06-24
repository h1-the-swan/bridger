# -*- coding: utf-8 -*-

DESCRIPTION = """Use the S2 API to get batch abstracts"""

import sys, os, time
import requests
from typing import List, Union
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

from bridger_dataprocess.s2_api_download import get_batch_paper_data_from_api

S2_API_KEY = os.getenv("S2_API_KEY")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def make_api_request(url: str, paper_ids: List, api_key=None) -> requests.Response:
    fields = [
        "paperId",
        "corpusId",
        "url",
        "title",
        "year",
        "abstract",
    ]
    params = {"fields": ",".join(fields)}
    headers = {}
    if api_key is not None:
        headers["x-api-key"] = api_key
    body = {"ids": [f"CorpusId:{id}" for id in paper_ids]}
    r = requests.post(url, headers=headers, params=params, json=body)
    return r


def run_api_download_abstracts(paper_ids: List, outfp: Union[str, Path]):
    fields = [
        "paperId",
        "corpusId",
        "url",
        "title",
        "year",
        "abstract",
    ]
    outfp = Path(outfp)
    data = get_batch_paper_data_from_api(paper_ids, fields=fields, api_key=S2_API_KEY)
    logger.debug(f"saving to {outfp}")
    pd.DataFrame(data).to_parquet(outfp)


def main(args):
    fp = Path(args.input)
    outfp = Path(outfp)
    paper_ids = []
    logger.info(f"reading file: {fp}")
    with fp.open() as f:
        for line in f:
            if line:
                paper_ids.append(line.strip())
    logger.info(f"found {len(paper_ids)} paper_ids")
    run_api_download_abstracts(paper_ids, args.output)


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
        "input", help="path to newline separated file of paper ids (corpus ids)"
    )
    parser.add_argument("output", help="path to output file (.parquet)")
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
