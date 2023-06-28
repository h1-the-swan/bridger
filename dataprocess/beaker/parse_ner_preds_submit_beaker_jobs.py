# -*- coding: utf-8 -*-

DESCRIPTION = """Submit beaker job for parsing NER predictions"""

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
    ResultSpec,
    DataMount,
    DataSource,
    EnvVar,
)
import pandas as pd

# default if not specified as an argument
DEFAULT_BEAKER_IMAGE_ID = "jasonp/bridger_dataprocess"

RESULTS_DATASET_IDS = [
    "01GD1ENSVEZ040XMAS5HV36GNS",
    "01GD1EN67PCPF4AMZVE84DXNCD",
    "01GCD0MDDMPP65WPRPM2D59DRH",
    "01GCCN9J3WFQ0QSF3CXZ9HE3E0",
    "01GCCNAFC31W4W4JGKDM89QJSD",
    "01GCCNB2WSZ2JJ0M1PC53S0V5F",
    "01GCDE6NRNQ7BSTVEMC8CN3DX8",
    "01GCDE6NXCP9R81ETK47PTDRMT",
    "01GCDS82KDRH5XMNRX5SRBJY4H",
    "01GCDS82W4326XC87TM14GTWSB",
    "01GCEFYCJZNX03JH40WQX3D0P9",
    "01GCDR1XQGQ693WCET10VRY6NJ",
    "01GCDR2V12GJTR650J5SWCXBQP",
    "01GCDR3EQXG36YSEDV3YD8W0G7",
    "01GCJQG514CHJ0AXPGCV8PMCSF",
    "01GCJQEAE5G7BGXNKZSWDGFH8Z",
    "01GCG0BMFVC4ANX25ZT19S5VGE",
    "01GCJR0MCZVFDD4139WYDK7YV9",
    "01GCJQZDHDPA8BF913MEECH1E3",
    "01GCJQYSYSV02B7E0EPDVHZ7KF",
    "01GCJQEAKCMN21DF0E37HFEFZQ",
    "01GCJQCSVT26TE4H6TM23J366A",
    "01GCRHYAVPRF2XFD8SW33GDFE5",
    "01GCSMJDSKSC45PAY8P065Y9HW",
    "01GCQ8SGCA3MCT0B5DKYFRX4G3",
    "01GCQ8TQHJ1JNXV3PRC0BZVFTJ",
    "01GCVVEG03X1QK95DJ7F1ZEJBT",
    "01GCSWEG7X13332J49J9MGJRHP",
    "01GCSW5YNVRST030EYFH49Q6TF",
]


def run_one_experiment(beaker: Beaker, image_id: str, dataset_id: str) -> str:
    experiment_name = f"parse_ner_preds_{dataset_id}"
    logger.debug(
        f"creating experiment with name: {experiment_name}, dataset_id: {dataset_id}"
    )
    spec = ExperimentSpec(
        description="Parse NER predictions",
        tasks=[
            TaskSpec(
                name="parse-ner-preds",
                image=ImageSource(beaker=image_id),
                command=["python3", "scripts/parse_ner_preds.py", "/data", f"/output/terms_from_dataset_{dataset_id}.parquet", "--debug"],
                datasets=[
                    DataMount(
                        source=DataSource(beaker=dataset_id),
                        mount_path="/data",
                    ),
                ],
                context=TaskContext(cluster="ai2/s2-elanding", priority="normal"),
                result=ResultSpec(path="/output"),
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
    experiments_data = []
    logger.debug(
        f"submitting one experiment per results dataset: {len(RESULTS_DATASET_IDS)} in total"
    )
    image_id = args.image
    if image_id is None:
        image_id = DEFAULT_BEAKER_IMAGE_ID
    logger.debug(f"using image: {image_id}")
    for dataset_id in RESULTS_DATASET_IDS:
        experiment_id = run_one_experiment(
            beaker=beaker,
            image_id=image_id,
            dataset_id=dataset_id,
        )
        logger.debug(f"experiment_id: {experiment_id} (dataset_id: {dataset_id})")
        experiments_data.append(
            {
                "results_dataset_id": dataset_id,
                "experiment_id": experiment_id,
            }
        )
    logger.debug(f"writing to file: {args.output}")
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
    parser.add_argument("--image", help="Beaker image ID")
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
