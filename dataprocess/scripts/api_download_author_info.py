# -*- coding: utf-8 -*-

DESCRIPTION = (
    """Use the S2 API to get batch author info like name and number of papers"""
)

import sys, os, time
import pickle
import requests
from typing import List
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

S2_API_KEY = os.getenv("S2_API_KEY")
S2_API_ENDPOINT = "https://api.semanticscholar.org/graph/v1/author/batch"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def make_api_request(url: str, author_ids: List, api_key=None) -> requests.Response:
    fields = [
        "authorId",
        "externalIds",
        "name",
        "affiliations",
        "paperCount",
        "citationCount",
        "hIndex",
    ]
    params = {"fields": ",".join(fields)}
    headers = {}
    if api_key is not None:
        headers["x-api-key"] = api_key
    body = {"ids": author_ids}
    r = requests.post(url, headers=headers, params=params, json=body)
    return r


def run_api_download_author_info(author_ids, output_file):
    batch_size = 1000
    outfp = Path(output_file)
    data = []
    for i, chunk in enumerate(chunks(author_ids, batch_size)):
        logger.info(f"making API request. i={i}. number of author ids = {len(chunk)}")
        r = make_api_request(S2_API_ENDPOINT, chunk, api_key=S2_API_KEY)
        data.extend(r.json())
    logger.info(f"done collecting info for {len(data)} authors. saving to {outfp}")
    outfp.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))


def main(args):
    fp = Path(args.input)
    author_ids = []
    logger.info(f"reading file: {fp}")
    with fp.open() as f:
        for line in f:
            if line:
                author_ids.append(line)
    logger.info(f"found {len(author_ids)} author_ids")
    run_api_download_author_info(author_ids, args.output)


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
    parser.add_argument("input", help="path to newline separated file of author_ids")
    parser.add_argument("output", help="path to output file (.pickle)")
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
