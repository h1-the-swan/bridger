# -*- coding: utf-8 -*-

DESCRIPTION = (
    """submit author distance jobs to beaker (using the AUTHOR_IDS environment variable)"""
)

from distutils.errors import LibError
import sys, os, time
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from typing import List

try:
    from humanfriendly import format_timespan
except ImportError:

    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)


import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

from beaker import (
    Beaker,
    ExperimentSpec,
    TaskSpec,
    ImageSource,
    TaskContext,
    ResultSpec,
    DataMount,
    DataSource,
    EnvVar,
    TaskResources,
)

DEFAULT_BEAKER_IMAGE_ID = "jasonp/bridger_dataprocess"

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def create_experiment(beaker: Beaker, author_id_list: List, experiment_idx: int) -> str:
    author_id_list_as_str = "{}".format(author_id_list)
    experiment_name = f"get_author_distances_min-year-2017_run01_{experiment_idx:04d}"
    command = [
        "python",
        "-u",
        "scripts/get_author_distances.py",
        "--debug",
    ]
    logger.info(f"creating experiment with name: {experiment_name}")
    spec = ExperimentSpec(
        description=experiment_name,
        tasks=[
            TaskSpec(
                name="get_author_distances",
                image=ImageSource(beaker=DEFAULT_BEAKER_IMAGE_ID),
                command=command,
                datasets=[
                    DataMount(
                        source=DataSource(beaker="01GT7JRQBF270RVMAQ2FNW12KC"), # result of get avg embeddings min-year 2017
                        mount_path="/data",
                    ),
                    DataMount(
                        source=DataSource(beaker="01GWZ558F3XNN5Y427PP279SX3"), # result of get-avg-specter-min-year-2017
                        mount_path="/specter",
                    ),
                ],
                # context=TaskContext(cluster="ai2/s2-elanding", priority="normal"),
                context=TaskContext(priority="preemptible"),
                # constraints={"cluster": ["ai2/s2-elanding", "ai2/s2-cirrascale"]},
                result=ResultSpec(path="/output"),
                resources=TaskResources(memory="40GiB"),
                env_vars=[EnvVar(name="AUTHOR_IDS", value=author_id_list_as_str)],
            ),
        ],
    )
    experiment = beaker.experiment.create(
        name=experiment_name,
        spec=spec,
    )
    return experiment.id


def main(args):
    fp = Path(args.input)
    beaker = Beaker.from_env(default_workspace="ai2/bridger")
    author_ids = []
    logger.info(f"reading file: {fp}")
    with fp.open() as f:
        for line in f:
            if line:
                author_ids.append(int(line))
    logger.info(f"found {len(author_ids)} author_ids")
    for i, chunk in enumerate(chunks(author_ids, 1000)):
        experiment_id = create_experiment(beaker, chunk, i)
        logger.info(f"submitted experiment: {experiment_id}")


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

