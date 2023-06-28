#!/bin/sh

# this script gets copied to the beaker image in Dockerfile

pip3 install /model/en_core_sci_sm-0.5.0.tar.gz && \
    python3 scripts/update_script_00_get_data_for_ner.py /output --existing /data/existing/df_paper_authors.parquet --debug