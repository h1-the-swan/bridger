# Notes for planning to create update script

## Bulk download papers from S2 datasets api, remove papers which we already have processed, and extract computer science papers.

`python -u scripts/bulk_download.py papers /home/jasonp/data/bridger/update_data --debug >& logs/bulk_download_papers_20230521T0837.log &`

`python -u scripts/extract_computer_science_papers.py /home/jasonp/data/bridger/update_data/staging/2023-05-16/papers/ /home/jasonp/data/bridger/update_data/computer_science_papers_update.gz --existing /home/jasonp/data/bridger/update_data/existing_papers.txt --debug`

`python -u scripts/api_download_abstracts.py /home/jasonp/data/bridger/update_data/update_papers.txt /home/jasonp/data/bridger/update_data/update_papers_and_abstracts.parquet --debug >& logs/api_download_abstracts_update_20230521T1336.log &`

`beaker session create --image beaker://jasonp/pl-marker-ner --gpus 1 --mount beaker://jasonp/update_papers_formatted_for_ner=/scierc --mount beaker://jasonp/sciner-scibert-model=/models/sciner-scibert/ --bare`
in pl_marker image:
```
>>> from entrypoint import run_command
>>> from pathlib import Path
>>> logfname = "/output/run_acener_modified.log"
>>> inputfname = "/scierc/titles_abstracts_plmarker_scierc_000000.json"
>>> logfp = Path(logfname)
>>> run_command(inputfname, logfp, gpu=True)
```
I ran the above on all of the update data in one beaker job. It took 7 hours.

`python scripts/parse_ner_preds.py /home/jasonp/data/bridger/update_data/plmarker_output /home/jasonp/data/bridger/update_data/plmarker_output/terms_update.parquet --debug >& logs/pares_ner_preds_update_20230527T0618.log &`

```
beaker session create --image beaker://jasonp/bridger_dataprocess --mount beaker://jasonp/en_core_sci_sm-0.5.0.tar.gz=/model

pip3 install /model/en_core_sci_sm-0.5.0.tar.gz

python3 -u scripts/normalize_ner_terms.py /home/jasonp/data/bridger/update_data/plmarker_output/terms_update.parquet /home/jasonp/data/bridger/update_data/terms_lemmatized_clean.parquet --debug >& logs/normalize_ner_terms_update_20230527T0644.log &
```

`python scripts/combine_embeddings.py /home/jasonp/data/bridger/update_data/sentence_transformer_embeddings/ /home/jasonp/data/bridger/update_data/combine-embeddings/ --debug`
I think we can skip the combine_embeddings step

`beaker experiment create beaker/final-processing-embeddings-update-beaker-conf.yaml`

`beaker experiment create beaker/final-processing-terms-update-beaker-conf.yaml`

## plan

- Bulk download all papers from datasets api
- see if papers are ordered by date. if not, loop through all of them, and extract computer science papers that we don't already have.
- download abstracts and specter embeddings for new papers
- extract terms from new papers (titles and abstracts)
- normalize terms for new papers 
- get term embeddings for new papers
- **combine all data, old papers and new papers**
- get average embeddings for all authors, min-years now()-5yr and now()-10yr
- get average specter for all authors, min-years
- get top terms for all authors, min-years
- get author distances, min-years
- get personas, min-year now()-10yr
- get persona distances
- get persona top terms


## identify files

I am at the beginning of the update process. I have data from previous updates. Where can I find...?

- All papers that have gone through processing (or attempted processing):
  - `/home/jasonp/data/bridger/computer_science_papers_and_abstracts.parquet`
- All papers for which we have embeddings:
  - `/home/jasonp/data/bridger/final-processing-embeddings/`
- All authors with their papers
  - ?
- All untouched authors (no new papers)
  - ?