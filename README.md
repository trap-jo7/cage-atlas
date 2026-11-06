# CAGE ATLAS

A fighter style-similarity scouting tool for UFC fighters. It measures how fighters fight
(strike volume, accuracy, defense, takedowns, control time), groups similar styles, finds the
closest matches, compares fighters, and replays past fights from their totals.
Similarity describes statistical resemblance only. It is not a prediction or betting tool.

## Data
UFC statistics originally from UFCStats.com, compiled on Kaggle by leandroiber (CC0-1.0):
https://www.kaggle.com/datasets/leandroiber/ufc-stats-complete-dataset
This is a snapshot, not a live feed. Not affiliated with or endorsed by the UFC or UFCStats.

## Run locally (Python 3.11+, Node 20+)
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.lock.txt
    cd data/raw && kaggle datasets download -d leandroiber/ufc-stats-complete-dataset
    unzip -o ufc-stats-complete-dataset.zip -d ufc-stats-complete-dataset && cd ../..
    python -m analysis.cageatlas.bouts
    python -m analysis.cageatlas.profiles
    python -m analysis.cageatlas.model train --k 4
    uvicorn backend.main:app --port 8000
    # in a second terminal:
    cd frontend && npm install --legacy-peer-deps && npm run dev

Open http://localhost:5174/#scout. Run the tests with `python -m pytest -q`.

## Notes
Style groups are exploratory (weak separation); see `models/ufc_career_model_meta.json` after training.
