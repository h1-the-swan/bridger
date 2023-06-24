# -*- coding: utf-8 -*-

DESCRIPTION = """Second step of update script after PL-Marker NER extraction: parse results of NER extraction, get embeddings, get distances"""

import sys, os, time
import json
import gzip
import pickle
from typing import List, Dict, Sequence
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

from parse_ner_preds import run_parse_ner_preds
from normalize_ner_terms import run_normalize_ner_terms
from get_sentence_transformer_embeddings import run_get_sentence_transformer_embeddings
from final_processing_embeddings import run_final_processing_embeddings
from final_processing_terms import run_final_processing_terms
from get_author_avg_embeddings import run_get_author_avg_embeddings
from get_author_avg_specter import run_get_author_avg_specter
from get_author_distances import get_recs
from get_author_top_terms import get_top_terms
from get_personas import run_get_personas
from get_persona_top_terms import get_top_terms_one_persona
from api_download_author_info import run_api_download_author_info

from bridger_dataprocess.s2_api_download import get_batch_paper_data_from_api
from bridger_dataprocess.average_embeddings import get_df_dists_from_author_id, get_author_term_matrix, get_avg_embeddings
from bridger_dataprocess.matrix import SciSightMatrix

import pandas as pd
import numpy as np


def main(args):
    outdir = Path(args.outdir)
    input_dir = Path(args.ner_preds)
    parsed_ner_file = outdir.joinpath("terms_update.parquet")
    existing_embeddings_dir = Path(args.existing_embeddings)
    existing_terms_dir = Path(args.existing_terms)
    run_parse_ner_preds(input_dir, parsed_ner_file)
    normalized_file = outdir.joinpath("terms_lemmatized_cleaned.parquet")
    if args.debug:
        checkpoint = True
    else:
        checkpoint = False
    run_normalize_ner_terms(parsed_ner_file, normalized_file, checkpoint=checkpoint)
    sentence_transformer_embeddings_dir = outdir.joinpath(
        "sentence_transformer_embeddings"
    )
    existing_embeddings_file = existing_embeddings_dir.joinpath(
        "embedding_term_to_id.parquet"
    )
    existing_embeddings_terms = pd.read_parquet(existing_embeddings_file).index.values
    logger.debug(
        f"sentence_transformer_embeddings_dir: {sentence_transformer_embeddings_dir} | len(existing_embeddings_terms): {len(existing_embeddings_terms)}"
    )
    run_get_sentence_transformer_embeddings(
        normalized_file,
        sentence_transformer_embeddings_dir,
        existing=existing_embeddings_terms,
    )

    score_threshold = 0.9
    fpath_embeddings = list(
        sentence_transformer_embeddings_dir.glob("embeddings_*.npy")
    )[0]
    fpath_terms = list(sentence_transformer_embeddings_dir.glob("terms_*.npy"))[0]
    final_processing_embeddings_dir = outdir.joinpath("final-processing-embeddings")
    run_final_processing_embeddings(
        normalized_file,
        fpath_embeddings,
        fpath_terms,
        final_processing_embeddings_dir,
        score_threshold=score_threshold,
        existing=existing_embeddings_dir,
    )
    existing_terms_file = list(existing_terms_dir.glob("terms_to_s2_id*.parquet"))[0]
    final_processing_terms_dir = outdir.joinpath("final-processing-terms")
    run_final_processing_terms(
        normalized_file,
        final_processing_terms_dir,
        score_threshold=score_threshold,
        old_data=existing_terms_file,
    )

    path_to_embeddings = final_processing_embeddings_dir.joinpath("embeddings.npy")
    path_to_terms = final_processing_embeddings_dir.joinpath(
        "embedding_term_to_id.parquet"
    )
    path_to_paper_term_embeddings = final_processing_embeddings_dir.joinpath(
        "dygie_embedding_term_ids_to_s2_id_scoreThreshold0.90.parquet"
    )
    # path_to_paper_authors = Path("/data/paper_authors/computer_science_paper_authors_update.parquet")
    path_to_paper_authors = Path(args.paper_authors)
    path_to_existing_paper_authors = Path(args.existing_paper_authors)
    logger.debug(f"loading embeddings file: {path_to_embeddings}")
    embeddings = np.load(path_to_embeddings)
    logger.debug(f"len(embeddings)=={len(embeddings)}")

    logger.debug(f"loading embeddings_terms file: {path_to_terms}")
    embeddings_terms = pd.read_parquet(path_to_terms)
    logger.debug(f"len(embeddings_terms)=={len(embeddings_terms)}")

    logger.debug(f"loading paper_term_embeddings file: {path_to_paper_term_embeddings}")
    df_paper_term_embeddings = pd.read_parquet(path_to_paper_term_embeddings)
    df_paper_term_embeddings["s2_id"] = df_paper_term_embeddings["s2_id"].astype(int)
    logger.debug(f"len(df_paper_term_embeddings)=={len(df_paper_term_embeddings)}")

    logger.debug(f"loading paper_authors file: {path_to_paper_authors}")
    df_paper_authors = pd.read_parquet(path_to_paper_authors)
    logger.debug(f"df_paper_authors.shape=={df_paper_authors.shape}")
    logger.debug(
        f"loading existing paper_authors file: {path_to_existing_paper_authors}"
    )
    df_paper_authors_existing = pd.read_parquet(path_to_existing_paper_authors)
    logger.debug(f"df_paper_authors_existing.shape=={df_paper_authors_existing.shape}")
    logger.debug("merging existing and new paper-authors table")
    _to_merge = df_paper_authors_existing[
        ~(df_paper_authors_existing["PaperId"].isin(df_paper_authors["PaperId"]))
    ]
    df_paper_authors = pd.concat([df_paper_authors, _to_merge]).drop_duplicates()
    logger.debug(f"after merge: df_paper_authors.shape=={df_paper_authors.shape}")
    outfp_df_paper_authors = outdir.joinpath('df_paper_authors.parquet')
    logger.debug(f"saving df_paper_authors to file: {outfp_df_paper_authors}")
    df_paper_authors.to_parquet(outfp_df_paper_authors)

    # get new specter embeddings
    existing_specter_dir = Path(args.existing_specter)
    fname_specter_ids = "specter_embeddings_corpus_ids.npy"
    fname_specter_embeddings = "specter_embeddings.npy"
    fp = existing_specter_dir.joinpath(fname_specter_ids)
    logger.debug(f"loading file: {fp}")
    existing_specter_ids = np.load(fp)
    logger.debug(f"loaded {len(existing_specter_ids)} paper ids")
    paper_ids_to_get = np.setdiff1d(
        df_paper_term_embeddings["s2_id"].unique(), existing_specter_ids
    )
    logger.debug(
        f"there are {len(paper_ids_to_get)} papers missing from the existing specter embeddings. getting these from the api..."
    )
    fields = [
        "paperId",
        "corpusId",
        "embedding",
    ]
    specter_update = get_batch_paper_data_from_api(paper_ids_to_get, fields=fields)
    update_specter_embeddings = []
    update_specter_ids = []
    for i, item in enumerate(specter_update):
        try:
            if item.get("embedding"):
                update_specter_ids.append(item["corpusId"])
                update_specter_embeddings.append(item["embedding"]["vector"])
        except AttributeError as e:
            logger.error(f"error encountered when getting item {i}. Item: {item}  exception: {e}")
    update_specter_ids = np.asarray(update_specter_ids)
    update_specter_embeddings = np.asarray(update_specter_embeddings)
    # merge with existing
    fp = existing_specter_dir.joinpath(fname_specter_embeddings)
    logger.debug(f"loading existing specter_embeddings from file: {fp}")
    existing_specter_embeddings = np.load(fp)
    specter_dir = outdir.joinpath("specter_embeddings")
    specter_dir.mkdir()
    update_specter_ids = np.hstack((existing_specter_ids, update_specter_ids))
    update_specter_embeddings = np.vstack(
        (existing_specter_embeddings, update_specter_embeddings)
    )
    outfp_specter_ids = specter_dir.joinpath(fname_specter_ids)
    logger.debug(
        f"saving numpy array with shape {update_specter_ids.shape} to file: {outfp_specter_ids}"
    )
    np.save(outfp_specter_ids, update_specter_ids)
    outfp_specter_embeddings = specter_dir.joinpath(fname_specter_embeddings)
    logger.debug(
        f"saving numpy array with shape {update_specter_embeddings.shape} to file: {outfp_specter_embeddings}"
    )
    np.save(outfp_specter_embeddings, update_specter_embeddings)

    # free up memory
    del update_specter_embeddings

    this_year = datetime.now().year
    for year in [this_year - 11, this_year - 6]:
        this_year_dir = outdir.joinpath(f"min_year_{year}")
        this_year_dir.mkdir()
        author_avg_embeddings_dir = this_year_dir.joinpath("avg_embeddings")
        logger.debug(
            f"getting author average embeddings for min year {year}. using output directory: {author_avg_embeddings_dir}"
        )
        run_get_author_avg_embeddings(
            embeddings,
            embeddings_terms,
            df_paper_term_embeddings,
            df_paper_authors,
            outdir=author_avg_embeddings_dir,
            min_year=year,
        )
        avg_specter_file = this_year_dir.joinpath(
            "average_author_specter_embeddings.pickle"
        )
        # reload specter embeddings because we deleted them earlier to free up memory
        logger.debug(
            f"reloading specter embeddings in order to get average specter (from file: {outfp_specter_embeddings})"
        )
        update_specter_embeddings = np.load(outfp_specter_embeddings)
        run_get_author_avg_specter(
            update_specter_embeddings,
            update_specter_ids,
            df_paper_authors,
            avg_specter_file,
            min_papers=5,
            min_year=year,
        )
        del update_specter_embeddings

        # # load all the files for getting distances
        # fp_task = author_avg_embeddings_dir.joinpath("avg_embeddings_task.pickle")
        # logger.debug(f"loading file: {fp_task}")
        # avg_embeddings_task = pd.read_pickle(fp_task)
        # fp_method = author_avg_embeddings_dir.joinpath("avg_embeddings_method.pickle")
        # logger.debug(f"loading file: {fp_method}")
        # avg_embeddings_method = pd.read_pickle(fp_method)
        fp_ssmat_task = author_avg_embeddings_dir.joinpath(
            "ssmat_author_term_task.pickle"
        )
        logger.debug(f"loading file: {fp_ssmat_task}")
        ssmat_author_term_task = SciSightMatrix.read_pickle(fp_ssmat_task)
        fp_ssmat_method = author_avg_embeddings_dir.joinpath(
            "ssmat_author_term_method.pickle"
        )
        logger.debug(f"loading file: {fp_ssmat_method}")
        ssmat_author_term_method = SciSightMatrix.read_pickle(fp_ssmat_method)
        fp_specter = avg_specter_file
        logger.debug(f"loading file: {fp_specter}")
        avg_embeddings_specter = pd.read_pickle(fp_specter)
        avg_embeddings_specter.index = avg_embeddings_specter.index.astype(int)
        
        # get author info
        author_ids = np.intersect1d(
            ssmat_author_term_method.row_labels.values,
            ssmat_author_term_task.row_labels.values,
        ).astype(str)
        outfp_author_info = this_year_dir.joinpath("author_info.pickle")
        run_api_download_author_info(list(author_ids), outfp_author_info)

        # get top terms
        # task
        logger.debug("getting top terms for all authors for: task")
        term_dict = get_top_terms(ssmat_author_term_task)
        outfp = this_year_dir.joinpath(f"author_top_terms_task.pickle")
        logger.debug(f"saving to {outfp}")
        outfp.write_bytes(pickle.dumps(term_dict, protocol=pickle.HIGHEST_PROTOCOL))
        # method
        logger.debug("getting top terms for all authors for: method")
        term_dict = get_top_terms(ssmat_author_term_method)
        outfp = this_year_dir.joinpath(f"author_top_terms_method.pickle")
        logger.debug(f"saving to {outfp}")
        outfp.write_bytes(pickle.dumps(term_dict, protocol=pickle.HIGHEST_PROTOCOL))

        # get personas
        personas_dir = this_year_dir.joinpath("personas")
        personas_dir.mkdir()
        fp_personas = personas_dir.joinpath("personas.pickle")
        logger.debug(f"getting personas for min_year {year} and saving to {fp_personas}")
        run_get_personas(outfp_specter_embeddings, outfp_specter_ids, df_paper_authors, fp_personas, min_papers=5, min_year=year)

        # get average embeddings for personas
        personas_dict = pickle.loads(fp_personas.read_bytes())
        personas_dict = {k: v for k, v in personas_dict.items() if v and len(v) > 1}
        logger.debug(f"getting focal embeddings for personas for {len(personas_dict)} authors")
        # personas_focal_embeddings_task = {}
        # personas_focal_embeddings_method = {}
        data = []
        max_personas = 5
        for author_id, personas in personas_dict.items():
            for i, persona in enumerate(personas[:max_personas]):
                persona_id = f"{author_id}P{i:02d}"
                for paper_id in persona:
                    data.append({
                        "AuthorId": persona_id,
                        "PaperId": paper_id,
                    })
        df_persona_paa = pd.DataFrame(data)
        # task
        ssmat_persona_task = get_author_term_matrix(
            embeddings_terms.index.values,
            df_paper_term_embeddings,
            df_persona_paa,
            label="Task",
            weighted=False,
            dedup_titles=False,
        )
        outfp = personas_dir.joinpath("ssmat_personas_term_task.pickle")
        logger.debug(f"saving matrix to {outfp}")
        outfp.write_bytes(ssmat_persona_task.to_pickle())
        avg_embeddings_persona_task = get_avg_embeddings(ssmat_persona_task.mat, embeddings, weighted=False)
        outfp = personas_dir.joinpath("avg_embeddings_personas_task.pickle")
        logger.debug(f"saving average embeddings to {outfp}")
        avg_embeddings_persona_task.to_pickle(outfp, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("getting top terms for personas for: task")
        term_dict = get_top_terms(ssmat_persona_task, cast_to_int=False)
        outfp = personas_dir.joinpath(f"personas_top_terms_task.pickle")
        logger.debug(f"saving to {outfp}")
        outfp.write_bytes(pickle.dumps(term_dict, protocol=pickle.HIGHEST_PROTOCOL))
        # method
        ssmat_persona_method = get_author_term_matrix(
            embeddings_terms.index.values,
            df_paper_term_embeddings,
            df_persona_paa,
            label="Method",
            weighted=False,
            dedup_titles=False,
        )
        outfp = personas_dir.joinpath("ssmat_personas_term_method.pickle")
        logger.debug(f"saving matrix to {outfp}")
        outfp.write_bytes(ssmat_persona_method.to_pickle())
        avg_embeddings_persona_method = get_avg_embeddings(ssmat_persona_method.mat, embeddings, weighted=False)
        outfp = personas_dir.joinpath("avg_embeddings_personas_method.pickle")
        logger.debug(f"saving average embeddings to {outfp}")
        avg_embeddings_persona_method.to_pickle(outfp, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("getting top terms for personas for: method")
        term_dict = get_top_terms(ssmat_persona_method, cast_to_int=False)
        outfp = personas_dir.joinpath(f"personas_top_terms_method.pickle")
        logger.debug(f"saving to {outfp}")
        outfp.write_bytes(pickle.dumps(term_dict, protocol=pickle.HIGHEST_PROTOCOL))

        # # get top terms for personas
        # logger.debug(f"loading personas file: {fp_personas}")
        # personas_dict: Dict[int, Sequence[int]] = pickle.loads(fp_personas.read_bytes())
        # logger.debug(f"personas_dict has len: {len(personas_dict)}")
        # fp_terms = final_processing_terms_dir.joinpath("terms_to_s2_id_scoreThreshold0.90.parquet")
        # logger.debug(f"loading paper to terms dataframe: {fp_terms}")
        # df_terms = pd.read_parquet(fp_terms)
        # logger.debug(f"dataframe has shape: {df_terms.shape}")
        # for t in ["task", "method"]:
        #     subset = df_terms[df_terms["label"].str.lower() == t]
        #     logger.debug(f"getting top terms for all personas for: {t}")
        #     term_dict = (
        #         {}
        #     )  # author_id -> list of lists, inner lists are terms for a persona, one list per persona
        #     for author_id, personas_list in personas_dict.items():
        #         if personas_list:
        #             term_dict[author_id] = []
        #             for i, persona_papers in enumerate(personas_list):
        #                 this_persona_top_terms = get_top_terms_one_persona(persona_papers, subset)
        #                 term_dict[author_id].append(this_persona_top_terms)
        #     outfp = personas_dir.joinpath(f"persona_top_terms_{t}.pickle")
        #     logger.debug(f"saving to {outfp}")
        #     outfp.write_bytes(pickle.dumps(term_dict, protocol=pickle.HIGHEST_PROTOCOL))

        # logger.debug(f"GETTING DISTANCES for {len(author_ids)} author ids...")
        # outdir_recs = this_year_dir.joinpath("bridger_recs")
        # outdir_recs.mkdir()
        # for author_id in author_ids:
        #     logger.debug(f"getting df_dists for author_id: {author_id}")
        #     try:
        #         df_dists = get_df_dists_from_author_id(
        #             author_id,
        #             ssmat_author_term_task=ssmat_author_term_task,
        #             ssmat_author_term_method=ssmat_author_term_method,
        #             avg_embeddings_method=avg_embeddings_method,
        #             avg_embeddings_task=avg_embeddings_task,
        #             avg_embeddings_specter=avg_embeddings_specter,
        #         )
        #         recs = get_recs(df_dists)
        #         outfp = outdir_recs.joinpath(f"recs_{author_id}.pickle")
        #         logger.debug(f"saving to {outfp}")
        #         outfp.write_bytes(pickle.dumps(recs, protocol=pickle.HIGHEST_PROTOCOL))
        #     except KeyError:
        #         logger.info(f"SKIPPING author {author_id}: not found")


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
    parser.add_argument("outdir", help="output base directory")
    parser.add_argument(
        "--ner-preds", help="path to directory with json ner preds files"
    )
    parser.add_argument(
        "--existing-embeddings",
        help="path to directory: final-processing-embeddings where existing data is stored from the last update",
    )
    parser.add_argument(
        "--existing-terms",
        help="path to directory: final-processing-terms where existing data is stored from the last update",
    )
    parser.add_argument(
        "--existing-specter",
        help="path to directory: where existing data is stored from the last update for specter embeddings (two files: specter_embeddings.npy and specter_embeddings_corpus_ids.npy)",
    )
    parser.add_argument("--paper-authors", help="path to paper-authors file (parquet)")
    parser.add_argument(
        "--existing-paper-authors",
        help="path to paper-authors file (parquet) for existing data from the last update",
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
