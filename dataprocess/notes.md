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

