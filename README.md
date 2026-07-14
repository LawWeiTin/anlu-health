# HealthBridge

HealthBridge is a safety-first, evidence-grounded health information assistant. It combines
deterministic emergency triage, hybrid retrieval from approved sources, citation-constrained
generation, privacy-aware authentication, and an offline evaluation gate. It supports biomedical
and traditional Chinese medicine questions while clearly separating traditional frameworks from
clinical evidence.

> **Scope:** education and care navigation only. It does not diagnose, prescribe, interpret images,
> or replace a licensed clinician. Any public launch needs clinical, legal, privacy, security, and
> accessibility review for the intended country and population.

## What is included

- FastAPI website with email/password authentication, revocable opaque sessions, CSRF protection,
  security headers, rate limiting, account deletion, and health/metrics endpoints.
- Medical safety layer that bypasses the model for emergencies and escalates concerning lump
  descriptions before retrieval or generation.
- PostgreSQL + pgvector knowledge store with source approval, evidence tiers, expiry dates,
  multilingual embeddings, hybrid reranking, and citation validation.
- A model-provider interface for a separately hosted open-weight model. The recommended starting
  point is `google/medgemma-1.5-4b-it`, subject to its terms and only after use-case validation.
- Render Blueprint and Docker deployment; no model weights or large datasets are stored locally.
- Colab QLoRA notebook, DVC pipeline, dataset/model cards, golden safety cases, CI, and scheduled
  evaluation workflows.

## Run locally

```powershell
Copy-Item .env.example .env
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
alembic upgrade head
python scripts/ingest.py data/seed_knowledge.jsonl
uvicorn app.main:app --reload
```

Open `http://localhost:8000`. Local mode uses deterministic mock inference and embeddings so the UI,
auth, safety paths, and tests work without downloading a model. Mock mode is deliberately rejected
when `APP_ENV=production`.

For a PostgreSQL development stack instead:

```powershell
docker compose up --build
docker compose exec app python scripts/ingest.py data/seed_knowledge.jsonl
```

## Production inference

Host the tuned model separately on a GPU service using vLLM/TGI or a managed open-model endpoint.
Configure an OpenAI-compatible chat and embedding endpoint using the variables in `.env.example`.
Render hosts the stateless web service and pgvector database; it does not download the 4B model.

Before enabling real users:

1. Review [docs/SAFETY.md](docs/SAFETY.md) and complete its clinical release checklist.
2. Populate the RAG index only from the approved [data/source_registry.yaml](data/source_registry.yaml).
3. Run `python scripts/evaluate.py --strict` and require all release gates to pass.
4. Have qualified Western and Chinese medicine clinicians review a representative, multilingual
   evaluation set. Benchmark scores are not a substitute for this review.
5. Complete the privacy, regulatory, threat-model, and incident-response work for your jurisdiction.

## Deployment on Render

1. Push this repository to GitHub and create a Render Blueprint from `render.yaml`.
2. Enter the model and embedding endpoint secrets when Render prompts for `sync: false` values.
3. After the database is ready, run `python scripts/ingest.py ...` from a one-off shell or controlled
   ingestion job.
4. Confirm `/health/ready`, then keep auto-deploy gated on passing GitHub checks.

Render PostgreSQL supports pgvector. Use paid plans with backups and high availability appropriate
to your risk assessment; the sample plans are starter defaults, not a clinical availability claim.

## Repository map

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/MLOPS.md](docs/MLOPS.md), and
[docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md).

