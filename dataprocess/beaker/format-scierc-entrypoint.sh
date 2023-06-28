#!/bin/sh

echo "FILE_IDX: $FILE_IDX"
echo "START_IDX: $START_IDX"
echo "END_IDX: $END_IDX"

pip3 install /model/en_core_sci_sm-0.5.0.tar.gz && \
    python3 scripts/format_data_for_ner.py /data/update_papers_and_abstracts.parquet /output/titles_abstracts_plmarker_scierc_${FILE_IDX}.json --start $START_IDX --end $END_IDX --debug