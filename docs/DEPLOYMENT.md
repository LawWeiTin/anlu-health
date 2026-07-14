# Production deployment

## Selected topology

- **Web:** Render Docker web service, Singapore region, Starter instance.
- **Database:** Render Postgres with pgvector, Singapore region, Basic-256mb instance.
- **Chat inference:** Hugging Face Inference Providers using the open-weight
  `Qwen/Qwen3.5-9B:cheapest` baseline.
- **Embeddings:** Hugging Face feature-extraction endpoint using
  `intfloat/multilingual-e5-small` (384 dimensions).
- **Training artifacts:** private Google Drive `Anlu Health` hierarchy, additive-only.

At the July 2026 listed prices, the Render web and database plans total USD 13/month before
bandwidth or other metered usage. Hugging Face inference is separately metered after account credits.
Do not activate paid resources without the account owner's approval.

## Secrets

The Blueprint asks for one `HF_TOKEN`. Create it as a fine-grained Hugging Face token with only the
permission to call Inference Providers. Do not grant repository write, organization administration,
or billing administration permissions. Render generates the data-encryption and metrics tokens.
Never commit any token to Git, Google Drive, screenshots, support tickets, or chat transcripts.

The inference provider receives user questions and retrieved source context. Before a public launch,
record its current retention and training-use terms, complete the intended-jurisdiction privacy
review, and update the user-facing notice if the provider or its terms change.

## Automated release path

1. GitHub Actions runs tests, linting, Bandit, the medical-safety gate, and a Docker build.
2. Render deploys only after required checks pass.
3. The pre-deploy command runs Alembic and idempotently ingests the approved seed corpus.
4. Render checks `/health/ready`, which requires both the database and approved sources.
5. Failed readiness prevents the new instance from receiving production traffic.

## Post-deploy verification

1. Confirm `/health/live` returns `{"status":"ok"}`.
2. Confirm `/health/ready` reports at least one approved source.
3. Register a disposable QA account and verify CSRF-protected authentication.
4. Exercise one emergency case, one lump case, one routine case, and one herb/medicine case.
5. Verify source links and confirm that application logs contain no question or answer text.
6. Delete the disposable account and confirm the session is revoked.

## Rollback

Use the latest known-good Render deploy if readiness, emergency recall, citations, authentication,
or upstream inference regresses. If the inference or retrieval dependency cannot be trusted, disable
generation rather than serving uncited fallback medical content.
