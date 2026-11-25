# Delivery ETA

Predicts how long a food delivery will take, in minutes, with a likely range. It works from the
moment an order is placed, and copes when some details (rider, weather, traffic) aren't known yet.

It comes as a web API, a page to try it on, and a command line for training and batch predictions.

| On 9,916 orders from days the model never saw | |
|---|---|
| Average error | **3.1 minutes** (always guessing the average: 7.6) |
| Within 5 minutes | 81% of orders |
| Real time falls inside the range | 82% of orders (aim: 80%) |

**Before using it for real customers, read the [model card](docs/MODEL_CARD.md).** The training data
looks artificial, so these numbers won't carry over to real deliveries until it is retrained on
your own orders.

## Run it

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) (`pip install uv`).

```powershell
python -m uv sync
```

```powershell
python -m uv run python -m delivery_eta serve
```

Then open <http://localhost:8000>. The API documentation is at <http://localhost:8000/docs>.

Or with Docker:

```powershell
docker build -t delivery-eta .
```

```powershell
docker run -p 8000:8000 delivery-eta
```

## Use the API

Only the distance is required. Give it directly, or as two places:

```json
POST /api/v1/predict
{
  "restaurant": {"lat": 22.745, "lon": 75.892},
  "customer": {"lat": 22.765, "lon": 75.912},
  "ordered_at": "2026-11-25T20:30:00+05:30",
  "traffic": "jam",
  "weather": "fog"
}
```

The reply holds `minutes`, `low`, `high`, which inputs were `missing`, and `warnings` when the order
is unlike anything the model learned from.

| Address | What it does |
|---|---|
| `POST /api/v1/predict` | One order. Add `?explain=true` to see what moved the estimate. |
| `POST /api/v1/predict/batch` | Up to 500 orders. |
| `GET /api/v1/model` | Which model is running and how accurate it is. |
| `GET /health`, `GET /ready` | For the host: the process is up; the model is loaded. |
| `GET /metrics` | For Prometheus: traffic, speed, predictions, missing inputs. |

Settings are environment variables, listed in [.env.example](.env.example). To put it online, see
[docs/DEPLOY.md](docs/DEPLOY.md).

## Commands

Each is run as `python -m uv run python -m delivery_eta <command>`. All but `serve` need
`python -m uv sync --all-extras` first.

| Command | What it does |
|---|---|
| `train` | Trains, tests on the newest days, and saves the model to `model/` only if it passes the quality gate. Add `--report reports/metrics.json` to update the published numbers. |
| `predict IN.csv OUT.csv` | Predicts every order in a raw CSV. |
| `compare` | Compares 12 models on the same data. |
| `tune` | Searches for better settings. |
| `serve` | Runs the API and the page. |

## Layout

```
app.py                 Where hosts such as Vercel find the app
configs/train.json     Model settings and the quality gate
data/raw/              train.csv (with delivery times) and test.csv (without)
docs/                  Model card and deployment guide
model/                 The released model
notebooks/             Exploring the data
reports/               Published accuracy, and the model comparison
scripts/smoke_test.py  Checks that a running copy really works
src/delivery_eta/
  data.py              Reads and cleans raw CSVs
  features.py          Turns orders into model inputs (shared by training and the API)
  training.py          Training, evaluation, quality gate
  experiments.py       Model comparison and tuning
  forest.py            Runs the trained trees without LightGBM
  artifact.py          Saves and loads the model
  predictor.py         Estimates, ranges, warnings
  api/                 The web service and page
tests/
```

## Working on it

```powershell
python -m uv sync --all-extras
```

```powershell
python -m uv run python -m pytest
```

```powershell
python -m uv run python -m ruff check .
```

```powershell
python -m uv run python -m pyright
```

CI runs all of these, retrains the model, and tries the service both as Vercel runs it and in Docker.
