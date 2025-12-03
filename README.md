This project analyzes missing person cases in the Philippines to identify
patterns in location, demographics, and time. It also generates heatmaps and 
attempts future hotspot prediction (if possible).

## Goals
- Collect real-world missing person case data
- Clean and preprocess the dataset
- Identify patterns using clustering and spatial analysis
- Create geographic heatmaps
- Develop a simple web-based proof-of-concept for visualization

## Status
Dataset collection ongoing.

## Folder Structure
```
missing-person-heatmap/
├── config/
│   └── settings.yaml
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   └── assets/
├── notebooks/
│   ├── evaluation/
│   ├── exploratory/
│   ├── model_export/
│   ├── preprocessing/
│   └── training/
├── devops/
│   ├── docker/
│   │   ├── backend.Dockerfile
│   │   └── frontend.Dockerfile
│   └── docker-compose.yml
├── scripts/
├── src/
│   ├── backend/
│   ├── core/
│   │   ├── analysis/
│   │   ├── ingestion/
│   │   ├── preprocessing/
│   │   └── visualization/
│   └── frontend/
├── pyproject.toml
└── README.md
```

## Getting Started
1. Install Poetry if you do not have it: `curl -sSL https://install.python-poetry.org | python3 -`
2. Install project dependencies: `poetry install`
3. Activate the virtual environment when hacking locally: `poetry shell`
4. Copy `config/settings.yaml` and tailor values to match available datasets or environments.
5. Place raw case data inside `data/raw/` and implement ingestion logic under `src/core/ingestion/`.

### Experimentation Workflow
- Use the curated notebook folders for focused experimentation: preprocessing → exploratory → training → evaluation → model_export.
- Promote reusable code into `src/core/` modules once it stabilizes.

### Implementation Notes
- `src/backend/` is a placeholder for APIs or services; bring in a framework (FastAPI, Flask, etc.) when we decide on the stack.
- `src/core/` contains reusable domain logic shared across interfaces.
- `src/frontend/` can host the chosen UI framework (Streamlit, React+FastAPI integration, etc.). Keep framework-specific code contained there.
- `devops/` stores infrastructure assets; `docker/` currently holds placeholder Dockerfiles and `docker-compose.yml`.

### Tooling
- Format code: `poetry run black src`
- Sort imports: `poetry run isort src`

## Tools
- Python
- Pandas, Scikit-learn
- Folium / Leaflet maps
- Frontend framework (to be decided, e.g., Streamlit or a web stack)