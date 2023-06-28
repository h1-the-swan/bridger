```sh
# run these in the dataprocess/ directory

# to build the docker image
docker build -t bridger_dataprocess -f ./beaker/Dockerfile .

# to upload the image to beaker
beaker image create --name bridger_dataprocess bridger_dataprocess

# to submit all of the beaker experiments
python3 beaker/format_scierc_submit_beaker_jobs.py beaker/format_scierc_beaker_experiments.csv --step 1000 --max-idx 10601326 --debug >& ./logs/format_scierc_submit_beaker_jobs_20220822T2221.log &
```