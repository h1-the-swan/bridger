# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# This is the directory that files will be copied into.
# It's also the directory that you'll start in if you connect to the image.
WORKDIR /stage/

RUN apt update && apt install -y gfortran

# RUN pip install --no-cache-dir \
#   humanfriendly \
#   python-dotenv \
#   python-dotenv \
#   pandas \
#   tqdm \
#   scispacy \
#   pyarrow 
RUN pip install "poetry==1.1.15"
COPY pyproject.toml .
# COPY poetry.lock .
RUN poetry config virtualenvs.create false \
  && poetry install --no-dev --no-interaction --no-ansi

COPY . .

# Running poetry install again will install the local package (bridger_dataprocess)
RUN poetry install --no-dev --no-interaction --no-ansi

RUN mkdir /output

RUN mv beaker/format-scierc-entrypoint.sh .
RUN mv beaker/normalize-terms-entrypoint.sh .
RUN mv beaker/run_update_script_00_entrypoint.sh .
RUN mv beaker/run_update_script_01_entrypoint.sh .