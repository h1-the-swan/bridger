for task/method prediction:

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
model = AutoModelForTokenClassification.from_pretrained("./models/sciner-scibert")
tokenizer = AutoTokenizer.from_pretrained("./models/sciner-scibert")
classifier = pipeline("ner", model=model, tokenizer=tokenizer)
classifier(["a sentence"])
```

above doesn't really work.

```sh
python run_acener_modified.py  --model_type bertspanmarker --model_name_or_path models/sciner-scibert --do_lower_case --data_dir scierc --learning_rate 2e-5 --num_train_epochs 50 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 60 --gradient_accumulation_steps 1 --max_seq_length 256 --max_mention_ori_length 8 --do_eval --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.json --dev_file dev.json --test_file sample100.json --output_dir models/sciner-scibert --overwrite_output_dir --output_results
```



## Adapt average embeddings method to this repo

From collabnetworks `average_embeddings.py`
need to refactor:
- `get_author_term_matrix()`
- `get_avg_embeddings()`?

In `util.py`
need to refactor:
- `drop_duplicate_titles()`
- `get_score_column()`

In `matrix.py`
need to refactor:
- `SciSightMatrix.from_df()`?

Also will need to alter `pipeline110-get_average_dygie_embeddings.py` and `pipeline111-get_average_specter_embeddings.py`


## Data pipeline after NER extraction --- 2022-10-19

Parse NER predictions: one experiment per results dataset --- `submit_new_parse_beaker_experiments.ipynb`

Download all parse files and combine them into one parquet file --- `download_parse_files.ipynb`

Upload this parquet file --- `beaker dataset create --name terms_from_papers.parquet ./terms_from_papers.parquet`

Normalize --- `beaker experiment create beaker/normalize-ner-beaker-conf.yaml`

Get term embeddings --- `beaker experiment create beaker/get-sentence-transformer-embeddings-beaker-conf.yaml` (make sure to point toward the right dataset)

Combine embeddings --- `beaker experiment create beaker/combine-term-embeddings-beaker.conf.yaml` (make sure to point toward the right dataset)

Final processing for terms --- `beaker experiment create beaker/final-processing-terms-beaker-conf.yaml` (make sure to point toward the right dataset)

Final processing for embeddings --- `beaker experiment create beaker/final-processing-embeddings-beaker-conf.yaml` (make sure to point toward the right dataset)


## Datasets --- 2022-12-12

### `all_recs_20221107.pickle`

This file is used to get the author recommendations for the Bridger demo once a focal author has been selected.

top level is a dictionary mapping `author_id` to a dictionary of:
label -> list of `author_ids`
Where label is one of:

```ts
// from Home.tsx in bridger-demo
const labelInfo: { [key: string]: string } = {
    simMethod: 'Authors who use similar methods',
    simTask: 'Authors who work on similar tasks',
    simMethod_distTask: 'Authors who use similar methods, but work on less similar tasks',
    simTask_distMethod: 'Authors who work on similar tasks, but use less similar methods',
    simspecter: 'Authors with similar papers (non-Bridger recommendations)',
};
```

TODO: For the author-term matrix: see https://beaker.org/ds/01GGQP03EW50SQETRF3ZSHPWA0/details (result of `scripts/get_author_avg_embeddings.py`)
Will have to use the sparse matrix to get top terms per author.
Planned this out in `matrices_author_paper_term.ipynb`
used that notebook to create:
`author_avg_embeddings/avg_embeddings/author_top_terms_method.pickle`
and
`author_avg_embeddings/avg_embeddings/author_top_terms_task.pickle`
which have the top terms for all authors. But we may have to recreate the `TextRanker.get_dygie_terms_trunc` method from collabnetworks to get rid of very similar terms in the list

Paper to term data is in `final-processing-terms/terms_to_s2_id_scoreThreshold0.90.parquet`


TODOs 2023-02-20:
- I changed `get_author_avg_embeddings.py` to exclude papers before `min-year`, so need to rerun that for 2012 and 2017.
- I added `get_avg_specter` to `average_embeddings.py`. Need to write a script to run that for a given `min-year`, and run it for 2012 and 2017. See `Untitled10.ipynb`.
- Modify `dataprocess/beaker/submit_beaker_experiments_get_dists.py` and use it to get distances for all authors (for min-year 2012 and 2017).
- Combine distance data to get author recs. See `beaker_experiments_after_get_recs_run00.ipynb`.
- At some point in the above steps, or after them (try to figure out the best time to do it), get top terms (tasks and methods) for authors. see `matrices_author_paper_term.ipynb`.