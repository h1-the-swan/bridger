# -*- coding: utf-8 -*-

DESCRIPTION = """takes a directory of pickle files---each file contains bridger recommendations for the personas for one author---and combines them, saving to a single pickle file"""

import sys, os, time
import pickle
import re
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


def main(args):
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise RuntimeError(
            f"input_dir {input_dir} does not exist or is not a directory!!"
        )
    outfp = Path(args.output)
    recs = {}
    for fp in input_dir.rglob("*.pickle"):
        m = re.search(r"recs_(\d+)\.pickle", fp.name)
        author_id = int(m.group(1))
        rec = pickle.loads(fp.read_bytes())
        persona_recs_head = []
        for persona in rec:
            if persona is None:
                persona_recs_head.append(None)
            else:
                rec_head = {}
                for cond, rec_ids in persona.items():
                    rec_head[cond] = rec_ids[:10]
                persona_recs_head.append(rec_head)
        recs[author_id] = persona_recs_head
    logger.debug(f"collected recs for {len(recs)} authors.")
    logger.debug(f"saving to {outfp}")
    outfp.write_bytes(pickle.dumps(recs))


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
        "input",
        help="path to input directory. the files in this directory must have the extension .pickle",
    )
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

