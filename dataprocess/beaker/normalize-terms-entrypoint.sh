#!/bin/sh

pip3 install /model/en_core_sci_sm-0.5.0.tar.gz && \
    python3 -u scripts/normalize_ner_terms.py /data/terms_from_papers.parquet /output/terms_lemmatized_cleaned.parquet --debug
