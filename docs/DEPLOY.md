# Putting it online

The service is the API, the page and the released model in `model/`. It needs no database, no disk and
no system libraries, so it runs on Vercel and on any host that runs a container. The model only gets
into `model/` by passing the quality gate, and CI retrains it on every change to check it still does.

| It needs | |
|---|---|
| Memory | 512 MB is a comfortable start |
| Port | The host's `PORT` variable, or 8000 |
| Health check | `/ready` |
| Settings | All optional, listed in [.env.example](../.env.example) |

To handle more traffic, run more containers. Keep one process in each, so metrics stay correct.

## Vercel

Import the repository at <https://vercel.com/new>. Nothing needs changing in the settings: the
framework is **FastAPI**, and `pyproject.toml` tells Vercel the app is in `app.py`. Every push to
`main` is then deployed.

- Under **Settings > Environment Variables**, set `ETA_METRICS` to `false`: Vercel starts and stops
  copies of the service, so counts kept in memory mean little. Add `ETA_API_KEYS` to protect it.
- Under **Settings > Functions**, choose the region nearest your users (Mumbai is `bom1`).

## Open or protected?

Without `ETA_API_KEYS`, anyone with the address can use it. With it, predictions and metrics need an
`X-API-Key` header, and the page asks for the key. There is no limit on requests per caller; use your
host's rate limiting if it is open to the public.

Make a key:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Store it in the host's secret settings, never in the repository.

## Google Cloud Run

Needs a Google Cloud project with billing, and the
[gcloud command](https://cloud.google.com/sdk/docs/install).

```powershell
gcloud run deploy delivery-eta --source . --region asia-south1 --memory 512Mi --cpu 1 --max-instances 3 --allow-unauthenticated
```

`--allow-unauthenticated` lets anyone open the address; use `ETA_API_KEYS` to protect predictions.

## Render, Railway, Fly.io and others

Create a web service from this repository, choose **Docker**, and set the health check to `/ready`.

## Your own server

```powershell
docker run -d --restart unless-stopped --read-only --cap-drop ALL -p 8000:8000 --env-file .env delivery-eta
```

Put it behind a web server that provides HTTPS (Caddy, nginx).

## After every release

```powershell
python scripts/smoke_test.py https://YOUR-ADDRESS --key YOUR-KEY
```

## Going back

Redeploy the previous image, or on Vercel choose **Instant Rollback**. The model is part of the
release, so code and model go back together.
`GET /api/v1/model` shows which model is running.

## Retraining on new orders

1. Replace `data/raw/train.csv` with your orders, in the same columns.
2. Run `python -m uv run python -m delivery_eta train --report reports/metrics.json`.
3. Read the new numbers. If the gate refuses the model, find out why before changing the gate.
4. Update the row counts in `tests/test_release.py` and the numbers in the model card.
5. Commit `model/` and `reports/metrics.json` with the data, and release as usual.
