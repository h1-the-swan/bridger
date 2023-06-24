# -*- coding: utf-8 -*-

DESCRIPTION = """Bulk download datasets from the S2 API"""

import sys, os, time
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from typing import List, Union
from urllib.parse import urlparse

from dotenv import load_dotenv, find_dotenv
import requests

try:
    from humanfriendly import format_timespan
except ImportError:

    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)


import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

load_dotenv(find_dotenv())

S2_API_KEY = os.getenv("S2_API_KEY")
S2_API_DATASETS_URL = "https://api.semanticscholar.org/datasets/v1/release/latest"


def make_api_request(url: str) -> requests.Response:
    headers = {"x-api-key": S2_API_KEY}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r


def get_dataset_download_urls(dataset_name: str) -> List[str]:
    url = f"{S2_API_DATASETS_URL}/dataset/{dataset_name}"
    r = make_api_request(url)
    return r.json()["files"]


def get_outfp(download_url: str, outdir_base: Path) -> Path:
    o = urlparse(download_url)
    outfp = Path(outdir_base).joinpath(o.path.strip("/"))
    return outfp


def download_file(download_url: str, outdir_base: Path) -> str:
    outfp = get_outfp(download_url, outdir_base)
    if not outfp.parent.exists():
        logger.info(f"creating directory: {outfp.parent}")
        outfp.parent.mkdir(parents=True)
    if outfp.exists():
        logger.info(f"file {outfp} already exists. skipping.")
        return "skipped"
    started = outfp.with_name(f"{outfp.name}.started")
    if started.exists():
        logger.info(
            f"file {started} was found, indicating that the file has already started downloading. skipping."
        )
        return "skipped"
    logger.debug(f"downloading file: {outfp.name}")
    started.touch()  # create .started file to keep track of what has started downloading
    # with requests.get(download_url, stream=True) as r:
    #     r.raise_for_status()
    #     with outfp.open("wb") as outf:
    #         for chunk in r.iter_content(chunk_size=int(1.049e7)):
    #             outf.write(chunk)
    with requests.get(download_url) as r:
        if r.status_code == 400:
            return "bad url"
        r.raise_for_status()
        with outfp.open("wb") as outf:
            outf.write(r.content)
    os.remove(started)
    logger.debug(f"finished downloading file: {outfp.name}")
    return "success"


def run_bulk_download(outdir: Union[str, Path], dataset_name: str):
    outdir_base = Path(outdir)
    files = get_dataset_download_urls(dataset_name)
    while True:
        if all([get_outfp(file, outdir_base).exists() for file in files]):
            break
        num_successes = 0
        for file in files:
            return_msg = download_file(file, outdir_base)
            if return_msg == "bad url":
                logger.debug("400 bad url error encountered. getting download urls again.")
                files = get_dataset_download_urls(dataset_name)
                break
            elif return_msg == "success":
                num_successes += 1
        if num_successes == 0:
            break


def main(args):
    run_bulk_download(args.outdir, args.dataset_name)


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
    parser.add_argument("dataset_name", help="dataset name")
    parser.add_argument("outdir", help="output base directory")
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
