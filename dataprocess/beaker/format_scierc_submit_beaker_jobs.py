# -*- coding: utf-8 -*-

DESCRIPTION = """Submit beaker jobs for formatting titles and abstracts for NER task"""

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

from beaker import (
    Beaker,
    ExperimentSpec,
    TaskSpec,
    ImageSource,
    TaskContext,
    TaskResources,
    ResultSpec,
    DataMount,
    DataSource,
    EnvVar,
)
import pandas as pd


DEFAULT_BEAKER_IMAGE_ID = "jasonp/bridger_dataprocess"


def run_one_experiment(
    beaker: Beaker, file_idx: int, start_idx: int, end_idx: int
) -> str:
    file_idx_str = f"{file_idx:06d}"
    experiment_name = f"format_scierc_update_{file_idx_str}"
    logger.info(
        f"creating experiment with name: {experiment_name}, file_idx: {file_idx_str}, start_idx: {start_idx}, end_idx: {end_idx}"
    )
    spec = ExperimentSpec(
        description="Format titles and abstracts for NER task",
        tasks=[
            TaskSpec(
                name="format-scierc",
                image=ImageSource(beaker=DEFAULT_BEAKER_IMAGE_ID),
                command=["bash", "format-scierc-entrypoint.sh"],
                datasets=[
                    DataMount(
                        source=DataSource(beaker="jasonp/en_core_sci_sm-0.5.0.tar.gz"),
                        mount_path="/model",
                    ),
                    DataMount(
                        source=DataSource(
                            beaker="jasonp/update_papers_and_abstracts.parquet"
                        ),
                        mount_path="/data",
                    ),
                ],
                env_vars=[
                    EnvVar(name="FILE_IDX", value=file_idx_str),
                    EnvVar(name="START_IDX", value=str(start_idx)),
                    EnvVar(name="END_IDX", value=str(end_idx)),
                ],
                context=TaskContext(priority="preemptible"),
                constraints={"cluster": ["ai2/s2-elanding", "ai2/s2-cirrascale"]},
                result=ResultSpec(path="/output"),
                resources=TaskResources(memory="50GiB"),
            ),
        ],
    )
    experiment = beaker.experiment.create(
        name=experiment_name,
        spec=spec,
    )
    return experiment.id


def main(args):
    beaker = Beaker.from_env(default_workspace="ai2/bridger")
    step = args.step
    file_idx = 0
    start_idx = 0
    experiments_data = []
    logger.info(
        f"experiments will be submitted until end_idx >= max_idx --- max_idx is {args.max_idx}"
    )
    while True:
        end_idx = start_idx + step
        if args.min_file_idx and file_idx < args.min_file_idx:
            # skip this
            pass
        else:
            experiment_id = run_one_experiment(
                beaker=beaker,
                file_idx=file_idx,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            logger.info(f"experiment_id: {experiment_id} (file_idx: {file_idx})")
            experiments_data.append(
                {
                    "file_idx": file_idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "experiment_id": experiment_id,
                }
            )
        if end_idx >= args.max_idx:
            break
        start_idx = end_idx
        file_idx += 1
    logger.info(f"writing to file: {args.output}")
    pd.DataFrame(experiments_data).to_csv(args.output)


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
        "output",
        help="path to output csv file that keeps track of experiment ids (csv)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1000,
        help="each task will run on this many documents",
    )
    parser.add_argument(
        "--max-idx",
        type=int,
        default=11000000,
        help="if we pass this document index, we will stop. this should typically equal the total number of documents",
    )
    parser.add_argument(
        "--min-file-idx",
        type=int,
        help="skip if file_idx is less than this number (i.e., if this has already been partially run and we're picking up where we left off",
    )
    parser.add_argument("--debug", action="store_true", help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.INFO)
        logger.debug("debug mode is on")
    main(args)
    total_end = timer()
    logger.info(
        "all finished. total time: {}".format(format_timespan(total_end - total_start))
    )
