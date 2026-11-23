PY ?= .venv/bin/python

.PHONY: setup data all clean-data anomalies features forecast test app lint

setup:            ## create venv and install
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -r requirements.lock -e ".[dev]"

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

app:              ## start the dashboard on http://127.0.0.1:8501
	$(PY) -m celeris.dashboard

lint:
	$(PY) -m ruff check src tests
