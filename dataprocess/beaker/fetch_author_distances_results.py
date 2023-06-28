# -*- coding: utf-8 -*-

DESCRIPTION = (
    """Download (fetch) results for a group of Beaker experiments (by a string match)"""
)

import sys, os, time
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

from beaker import Beaker


def main(args):
    outdir = Path(args.outdir)
    if not outdir.is_dir():
        raise RuntimeError(
            f"output directory {outdir} does not exist, or is not a directory"
        )
    beaker = Beaker.from_env(default_workspace="ai2/bridger")
    logger.info(f"getting experiments that match: {args.match} (limit 10000)")
    experiments = beaker.workspace.experiments(match=args.match, limit=10000)
    logger.info(f"found {len(experiments)} experiments")
    for experiment in experiments:
        if 'TEST' in experiment.name:
            continue
        logger.info(
            f"fetching results for experiment: {experiment.id} and saving to {outdir}"
        )
        res = beaker.experiment.results(experiment)
        beaker.dataset.fetch(res, target=outdir, force=True, quiet=True)


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
        "match",
        help="get experiments that match this string (e.g., 'get_author_distances_min-year-2012_run02')",
    )
    parser.add_argument("outdir", help="output directory")
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
