PY ?= .venv/bin/python

.PHONY: setup data all clean-data anomalies features forecast test app lint

setup:            ## create venv and install
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -e ".[dev]"

data:             ## unzip dataset into data/raw (expects data/raw/student_data.zip)
	cd data/raw && unzip -oj student_data.zip '*.csv'

all:              ## run the whole pipeline
	$(PY) -m celeris all

clean-data:
	$(PY) -m celeris clean

anomalies:
	$(PY) -m celeris anomalies

features:
	$(PY) -m celeris features

forecast:
	$(PY) -m celeris forecast

test:
	$(PY) -m pytest -q

app:
	$(PY) -m streamlit run app/app.py

lint:
	$(PY) -m ruff check src tests app
