from typing import Dict, List, Union, TypedDict
import logging

import pandas as pd
import numpy as np
import spacy

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)


def tokenize_expand_abbreviations(sent: spacy.tokens.Span, abrv_map: Dict) -> List[str]:
    toks = []
    for tok in sent:
        if tok.is_space:
            continue
        if tok.idx in abrv_map:
            toks.extend([x.text for x in abrv_map[tok.idx]._.long_form])
        else:
            toks.append(tok.text)
    return toks


def format_doc(
    doc_id: Union[int, str],
    text: str,
    nlp: spacy.Language,
    expand_abbreviations: bool = False,
) -> Dict:
    spacy_doc = nlp(text)
    sentences = []
    # need to have blank data for "ner" and "relations"
    ners = []
    relations = []
    for sent in spacy_doc.sents:
        if expand_abbreviations is True:
            abrv_map = {abrv.start_char: abrv for abrv in spacy_doc._.abbreviations}
            toks = tokenize_expand_abbreviations(sent, abrv_map)
        else:
            toks = [tok.text for tok in sent if not tok.is_space]
        if len(toks) != 0:
            sentences.append(toks)
            # need to have blank data for "ner" and "relations"
            ners.append([])
            relations.append([])
    return {
        "doc_key": doc_id,
        "sentences": sentences,
        "ner": ners,
        "relations": relations,
    }


def get_score_column(df: pd.DataFrame) -> np.ndarray:
    """Assign a score to each paper for each author based on paper metadata.
    Score is based on author position (first or last author, or middle author)
    and the rank of the paper (paper importance).

    NOTE: Rank is no longer available after the transition from MAG to S2AG.
    For now, we are ignoring it. We may need to find a replacement.

    Example usage:
    scores = get_score_columns(df_paper_authors)
    df_paper_authors["score"] = scores

    Args:
        df (pd.DataFrame): PaperAuthor dataframe, with one row per paper per author. Should have columns "PaperId", "AuthorId", "AuthorSequenceNumber"

    Returns:
        np.ndarray: array of scores corresponding to the rows in `df`
    """
    if "num_authors" not in df.columns:
        df["num_authors"] = df.groupby("PaperId")["AuthorSequenceNumber"].transform(
            "max"
        )
    if "is_last_author" not in df.columns:
        df["is_last_author"] = np.where(
            df["num_authors"] == df["AuthorSequenceNumber"], True, False
        )

    # df = df[["PaperId", "AuthorId", "AuthorSequenceNumber", "is_last_author"]]

    multiplier_first_or_last_author = 1.0
    cond1 = df["AuthorSequenceNumber"] == 1
    multiplier_middle_author = 0.75
    cond2 = df["is_last_author"] == True
    multiplier = np.where(
        cond1 | cond2, multiplier_first_or_last_author, multiplier_middle_author
    )

    ## Commented out below is the legacy code for the Rank multiplier (relied on an additional input to this function: df_papers)
    # from sklearn.preprocessing import MinMaxScaler
    # df["rank_scaled"] = df.PaperId.map(df_papers.set_index("PaperId")["rank_scaled"])
    # rank_scaled = (
    #     MinMaxScaler(feature_range=(0.5, 1))
    #     .fit_transform(df[["rank_scaled"]])
    #     .flatten()
    # )
    # multiplier = multiplier * rank_scaled

    return multiplier


def map_label_to_idx(labels: np.ndarray):
    return {val: idx[0] for idx, val in np.ndenumerate(labels)}
