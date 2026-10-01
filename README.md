# F-UJI FAIR Assessment Pipeline

Pipeline for harvesting records from research-data repositories and evaluating them with [F-UJI](https://www.f-uji.net/).

Supported repository software architectures:

- Dataverse
- Zenodo
- Chemotion

Each repository is processed independently. Results can also be compared across repositories.

## Pipeline

```text
1. harvest
   data/<repo>/01_harvest/

2. evaluate with F-UJI
   data/<repo>/02_evaluations/

3. parse results
   data/<repo>/03_parsed/

4. analyze
   data/<repo>/04_analysis/

cross-repository comparison
   data/_comparison/
```

Stages are resumable and communicate through files on disk.

## Setup

### Docker

Docker is the recommended way to run the project.

```bash
cp .env.example .env
```

Add any required repository credentials to `.env`.

Start F-UJI:

```bash
docker compose up -d fuji
```

Run the full pipeline:

```bash
docker compose run --rm pipeline
```

Useful variants:

```bash
# One repository
docker compose run --rm pipeline --repo nfdi4cat_zenodo

# Selected stages
docker compose run --rm pipeline --stages parse analyze

# One stage for one repository
docker compose run --rm pipeline \
  --repo nfdi4cat_zenodo \
  --stages harvest

# Show configured repositories
docker compose run --rm pipeline --list-repos
```

Stop services:

```bash
docker compose down
```

### Run on the host

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d fuji

python -m pipeline --list-repos
python -m pipeline
```

Run commands from the repository root.

## Configuration

All pipeline configuration is in:

```text
config.yaml
```

Repositories are defined under `repositories:`.

Example:

```yaml
repositories:
  - name: my_repo
    type: dataverse
    base_url: https://example.org
```

The repository `name` becomes its output directory:

```text
data/<name>/
```

Repository-specific options depend on the backend. See the existing entries in `config.yaml` for examples.

Credentials should be stored in `.env`.

## Repository layout

```text
.
├── config.yaml                 # repositories and pipeline settings
├── .env.example                # credential template
├── docker-compose.yml          # F-UJI + pipeline services
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
│
├── pipeline/
│   ├── __main__.py             # pipeline orchestrator
│   ├── common.py               # config, paths, IO, retries
│   ├── content.py              # content classification
│   ├── stage1_harvest.py
│   ├── stage2_evaluate.py
│   ├── stage3_parse.py
│   ├── stage4_analyze.py
│   │
│   ├── harvesters/
│   │   ├── base.py
│   │   ├── dataverse.py
│   │   ├── zenodo.py
│   │   └── chemotion.py
│   │
│   └── analysis/
│       ├── analyze_with_plots.py
│       ├── decompose_effort.py
│       └── compare_repos.py
│
├── scripts/                    # maintenance/ad-hoc tools
├── tests/
│
├── data/                       # generated pipeline data
├── logs/                       # stage logs
└── runs/                       # run metadata
```

## Generated data

For each repository:

```text
data/<repo>/
├── 01_harvest/
│   ├── manifest.parquet
│   └── harvest_meta.json
├── 02_evaluations/
├── 03_parsed/
│   ├── summary.parquet
│   ├── metrics.parquet
│   └── metric_tests.parquet
└── 04_analysis/
```

Cross-repository results are written to:

```text
data/_comparison/
```

Raw API responses and F-UJI responses are retained alongside the processed outputs.

## Adding a repository

Add an entry to `config.yaml`.

To add a new repository software architectures:

1. Add a harvester under `pipeline/harvesters/`.
2. Register it in `pipeline/harvesters/__init__.py`.
3. Add the type to `_VALID_REPO_TYPES` in `pipeline/common.py`.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

## License

MIT. See `LICENSE`.

Citation metadata is available in `CITATION.cff`.