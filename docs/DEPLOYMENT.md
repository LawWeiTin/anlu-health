# Deployment

## Private V32 localhost validation

The current release-candidate path is deliberately narrower than a production deployment:

- **Web:** localhost-only FastAPI application.
- **Chat inference:** private Hugging Face Inference Endpoint running
  `google/medgemma-1.5-4b-it` with the private
  `lawwt/anlu-health-medgemma-v32-adapter` LoRA, served to the app as `anlu-v32`.
- **Compute:** one L4 GPU with automatic scale-to-zero.
- **Retrieval:** the existing approved-source hybrid retriever. A real embedding provider must be
  configured and the source index rebuilt before claiming neural semantic retrieval.
- **Artifacts:** private Kaggle output transferred directly to a private Hugging Face model
  repository; model weights and adapters are not retained on the laptop.

This endpoint is for bounded end-to-end safety and UX testing only. It must remain private, must be
scaled to zero after testing, and must not be connected to a public site while physician,
pharmacist, registered TCM practitioner, privacy/security, and retrieval-safety approvals are
pending.

## Future production topology

- **Web:** Render Docker web service, Singapore region.
- **Database:** Render Postgres with pgvector, Singapore region.
- **Chat inference:** a separately approved private open-model endpoint.
- **Embeddings:** a separately approved 384-dimensional multilingual embedding endpoint.
- **Training artifacts:** private Google Drive `Anlu Health` hierarchy, additive-only.

Do not activate additional paid resources without the account owner's explicit approval.

## Secrets

Use separate fine-grained tokens:

- A short-lived Kaggle transfer token may write only to the selected private V32 adapter repository.
- The localhost app token may call only the selected private Inference Endpoint.

Do not grant organization administration or billing administration permissions. Never commit any
token to Git, Google Drive, screenshots, support tickets, or chat transcripts. Revoke the transfer
token after the adapter repository and its checksums are verified.

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
