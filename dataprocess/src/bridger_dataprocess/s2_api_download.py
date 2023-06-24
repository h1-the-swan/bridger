# -*- coding: utf-8 -*-

DESCRIPTION = """tools for downloading data from the semantic scholar API (requires an API Key, set as an environment varianble)"""

import sys, os, time
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

import requests
from requests import RequestException
import backoff

S2_API_KEY = os.getenv("S2_API_KEY")
S2_API_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/batch"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@backoff.on_exception(backoff.expo, RequestException, max_time=30)
@backoff.on_predicate(backoff.expo, lambda x: x.status_code >= 429, max_time=30)
def make_api_request(
    url: str, paper_ids: List, fields=None, api_key=None
) -> requests.Response:
    if fields is None:
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


def get_batch_paper_data_from_api(
    paper_ids: List, batch_size=500, fields=None, api_key=S2_API_KEY
):
    data = []
    logger.debug(f"starting data collection for {len(paper_ids)} paper_ids")
    for i, chunk in enumerate(chunks(paper_ids, batch_size)):
        logger.info(f"making API request. i={i}. number of paper ids = {len(chunk)}")
        r = make_api_request(S2_API_ENDPOINT, chunk, fields=fields, api_key=api_key)
        results = [item for item in r.json() if item is not None]
        data.extend(results)
    logger.info(f"done collecting info for {len(data)} papers")
    return data
