# Graph RAG Nobel Prize Demo

This repository contains a Graph RAG demo over a Nobel Prize knowledge graph stored in a Kuzu database, using DSPy and marimo.

## Setup

We recommend using `uv` to manage the Python environment:

```bash
uv sync
source .venv/bin/activate  # or your preferred activation method
```

Start Kuzu and ensure the `nobel.kuzu` database is available as configured in `graph_rag.py`.

## Basic workflow

1. Ingest the base Nobel graph (from `eda.py` / existing DB).
2. Enrich the graph with API data:

	```bash
	uv run create_nobel_api_graph.py
	```

3. Run the interactive Graph RAG demo app:

	```bash
	uv run marimo run graph_rag.py
	```

## Few-shot evaluation

We also ran the few-shot evaluation and analysis scripts:

```bash
python test_few_shot_examples.py
python analyze_test_results.py
```

These generate `test_results.json`, `analysis_summary.json`, and `summary_report.md`, which are already present in the repository.
