# -*- coding: utf-8 -*-

DESCRIPTION = (
    """tools for tracking log files and output files from the PL-Marker NER task"""
)

import sys, os, time, re, json
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from typing import Dict, List, Optional

try:
    from humanfriendly import format_timespan
except ImportError:

    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)


import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

FILE_INDEX_PATTERN = re.compile(r"(\d\d\d\d\d\d)")


def get_group_index_from_fname(fname: str) -> int:
    idx_str = FILE_INDEX_PATTERN.search(fname).group(1)
    return int(idx_str)


def get_jsonl_data_from_dataset(beaker, dataset_id: str, filepath: str) -> List[Dict]:
    contents = b"".join(beaker.dataset.stream_file(dataset_id, filepath, quiet=True))
    data = [json.loads(line) for line in contents.decode("utf8").split("\n") if line]
    return data


def get_filepaths_from_dataset(beaker, dataset_id: str) -> List[str]:
    ret = []
    for item in beaker.dataset.ls(dataset_id):
        ret.append(item.path)
    return ret


def filter_dataset_files(fnames: List[str], filetype: str) -> List[str]:
    if filetype in ["log", ".log"]:
        return [fname.endswith(".log") for fname in fnames]
    elif filetype in ["results"]:
        return [
            fname
            for fname in fnames
            if (fname.endswith(".json") and fname != "results.json")
        ]
    else:
        raise ValueError()


def insert_into_results_dataset_from_experiment_id(
    experiment_id: str, beaker, con, get_num_files=True
):
    experiment = beaker.experiment.get(experiment_id)
    description = experiment.description
    task = "ner"
    job = beaker.experiment.latest_job(experiment_id)
    result_dataset_id = job.execution.result.beaker
    ts = job.status.finalized  # will be None for currently running jobs
    if ts is not None:
        ts = ts.timestamp()
    if get_num_files is True:
        files = get_filepaths_from_dataset(beaker, result_dataset_id)
        num_logfiles = len(filter_dataset_files(files, "log"))
        num_resultsfiles = len(filter_dataset_files(files, "results"))
        cur = con.execute(
            "INSERT INTO results_datasets(id, task, task_desc, num_files, num_logfiles, num_resultsfiles, completed_ts, experiment_id, last_job_id) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result_dataset_id,
                task,
                description,
                len(files),
                num_logfiles,
                num_resultsfiles,
                ts,
                experiment_id,
                job.id,
            ),
        )
    else:
        cur = con.execute(
            "INSERT INTO results_datasets(id, task, task_desc, completed_ts, experiment_id, last_job_id) VALUES(?, ?, ?, ?, ?, ?)",
            (result_dataset_id, task, description, ts, experiment_id, job.id),
        )


def update_ner_chunks_from_results(dataset_id: str, experiment_id: str, beaker, con):
    experiment = beaker.experiment.get(experiment_id)
    ts = experiment.created.timestamp()
    files = get_filepaths_from_dataset(beaker, dataset_id)
    resultsfiles = filter_dataset_files(files, "results")
    for fname in resultsfiles:
        chunk_id = get_group_index_from_fname(fname)
        cur = con.execute(
            "UPDATE ner_chunks SET pred_ents_dataset_id = ?, pred_ents_dataset_subpath = ?, submitted_experiment_id = ?, submitted_ts = ? WHERE id = ?",
            (dataset_id, fname, experiment_id, ts, chunk_id),
        )


class PLMarkerChunk:
    def __init__(
        self,
        id: int,
        format_experiment_id: Optional[str] = None,
        input_dataset_id: Optional[str] = None,
        submitted_experiment_id: Optional[str] = None,
        pred_ents_dataset_id: Optional[str] = None,
        pred_ents_dataset_subpath: Optional[str] = None,
    ) -> None:
        self.id = id
        self.format_experiment_id = format_experiment_id
        self.input_dataset_id = input_dataset_id
        self.submitted_experiment_id = submitted_experiment_id
        self.pred_ents_dataset_id = pred_ents_dataset_id
        self.pred_ents_dataset_subpath = pred_ents_dataset_subpath
