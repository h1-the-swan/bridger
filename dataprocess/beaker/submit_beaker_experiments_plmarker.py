# -*- coding: utf-8 -*-

DESCRIPTION = (
    """submit NER jobs to beaker (using the FILE_INDICES environment variable)"""
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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def create_experiment(beaker: Beaker, file_idx_list: List, experiment_idx: int) -> str:
    file_idx_list_as_str = "{}".format(file_idx_list)
    experiment_name = f"PL-Marker_rerun_finalmissing_{experiment_idx:04d}"
    command = [
        "python",
        "-u",
        "entrypoint.py",
        "--use-env-file-indices",
        # --processes,
        # 1,
        "--gpu",
        "--batch",
        "60",
        "--debug",
    ]
    logger.debug(f"creating experiment with name: {experiment_name}")
    spec = ExperimentSpec(
        description=f"PL-Marker NER rerun chunks FINALMISSING{experiment_idx:04d} (gpu 1proc batch60)",
        tasks=[
            TaskSpec(
                name="ner",
                image=ImageSource(beaker="jasonp/pl-marker-ner"),
                command=command,
                datasets=[
                    # DataMount(
                    #     source=DataSource(beaker="jasonp/plmarker_ner_input"),
                    #     mount_path="/data",
                    # ),
                    DataMount(
                        source=DataSource(beaker="jasonp/missing_input_datasets_20221014"),
                        mount_path="/data",
                    ),
                    DataMount(
                        source=DataSource(beaker="jasonp/sciner-scibert-model"),
                        mount_path="/models/sciner-scibert/",
                    ),
                ],
                # context=TaskContext(cluster="ai2/s2-elanding", priority="normal"),
                context=TaskContext(priority="normal"),
                constraints={"cluster": ["ai2/s2-elanding", "ai2/s2-cirrascale"]},
                result=ResultSpec(path="/output"),
                resources=TaskResources(gpu_count=1),
                env_vars=[EnvVar(name="FILE_INDICES", value=file_idx_list_as_str)],
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
    file_indices = []
    logger.debug(f"reading file: {fp}")
    with fp.open() as f:
        for line in f:
            file_indices.append(int(line))
    logger.debug(f"found {len(file_indices)} file_indices")
    for i, chunk in enumerate(chunks(file_indices, 100)):
        experiment_id = create_experiment(beaker, chunk, i)
        logger.debug(f"submitted experiment: {experiment_id}")


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
    parser.add_argument("input", help="path to newline separated file of ner_chunk ids")
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
