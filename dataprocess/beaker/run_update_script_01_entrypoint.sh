#!/bin/sh

pip3 install /model/en_core_sci_sm-0.5.0.tar.gz && \
    python3 scripts/update_script_01_parse_embed_and_get_averages.py /output --ner-preds /data/pl-marker-output --existing-embeddings /data/previous_update/final-processing-embeddings --existing-terms /data/previous_update/final-processing-terms --existing-specter /data/previous_update/specter_embeddings --paper-authors /data/previous_update/df_paper_authors.parquet --existing-paper-authors /data/script00_output/computer_science_paper_authors_update.parquet --debug