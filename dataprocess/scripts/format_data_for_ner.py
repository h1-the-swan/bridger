# -*- coding: utf-8 -*-

DESCRIPTION = """Format dataset for use with pytorch/transformers
"""

import sys, os, time, json
from typing import List, TypedDict, Union, Optional
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

import spacy
import pandas as pd
import numpy as np

from bridger_dataprocess import format_doc


# def get_outfile_name(p, i: int) -> Path:
#     """Append filename with a zero-padded number

#     :p: file path
#     :i: number to append
#     :returns: file path

#     """
#     p = Path(p)
#     n = f"{p.stem}_{i:04d}{p.suffix}"
#     return p.with_name(n)


def run_format_data_for_ner(input_parquet_file, output, start: Optional[int] = None, end: Optional[int] = None, expand_abbreviations=False):
    nlp = spacy.load("en_core_sci_sm")
    if expand_abbreviations is True:
        raise RuntimeError("expand_abbreviations has not been implemented yet")
        # from scispacy.abbreviation import AbbreviationDetector

        # # Add the abbreviation pipe to the spacy pipeline.
        # nlp.add_pipe("abbreviation_detector")

    outfpath = Path(output)
    logger.debug("loading input file: {}".format(input_parquet_file))
    columns = ["corpusId", "title", "abstract"]
    df = pd.read_parquet(input_parquet_file, columns=columns)
    logger.debug("input file has shape: {}".format(df.shape))

    idx_start = start
    if idx_start is None:
        idx_start = 0
    idx_end = end
    if idx_end is None:
        idx_end = len(df)
    outfpath = Path(output)
    df = df.reset_index()
    df = df.iloc[idx_start:idx_end]

    df = df.set_index("corpusId")
    data = df["title"] + ". " + df["abstract"]
    data.dropna(inplace=True)
    # data cleaning
    data = data.str.replace("\n", " ")
    # data = data.str.replace('\u2008', '')
    logger.debug("processing {} papers (titles and abstracts)".format(len(data)))

    logger.debug("writing to file: {}".format(outfpath))
    outf = outfpath.open("w")

    line_num = idx_start
    logger.debug("starting with paper {}".format(data.index[0]))
    try:
        for paper_id, text in data.iteritems():
            if expand_abbreviations is True:
                raise RuntimeError("expand_abbreviations has not been implemented yet")
                # outline = format_doc_expand_abbreviations(paper_id, text, nlp)
            else:
                outline = format_doc(paper_id, text, nlp)
            if outline["sentences"]:
                print(json.dumps(outline), file=outf)
                line_num += 1
    finally:
        outf.close()


def main(args):
    run_format_data_for_ner(args.input, args.output, args.start, args.end, args.expand_abbreviations)


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
    parser.add_argument("input", help="path to input file (parquet)")
    parser.add_argument("output", help="path to output JSON-lines file (.json)")
    parser.add_argument(
        "--start", type=int, default=None, help="if specified, start with this index"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="if specified, end with this index (not included)",
    )
    parser.add_argument(
        "--expand-abbreviations",
        action="store_true",
        help="use scispacy to expand all abbreviations to their long forms",
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
