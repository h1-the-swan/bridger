from typing import Dict, List, Union, TypedDict
import logging

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
