# Data governance

## Source policy

Only sources registered in `data/source_registry.yaml` may be ingested. Each source needs a stable
publisher, URL, reuse basis, evidence tier, language, review owner, and expiry window. A checksum
makes each indexed version auditable.

Preferred knowledge sources are public-health agencies, government health libraries, systematic
reviews, and professional guidelines. Individual blogs, forums, testimonials, retailer pages,
unlicensed scraped conversations, and model-generated medical text are prohibited.

## Dataset roles

- **RAG knowledge:** current, source-linked consumer information and guidelines. It is never mixed
  into model weights by default.
- **Supervised fine-tuning:** small, reviewed examples that teach response structure, abstention,
  evidence separation, and escalation. They do not teach new clinical facts.
- **Evaluation:** held-out safety cases plus open benchmarks such as PubMedQA. Benchmark items are not
  reused for training.

## Approved starting points

- MedlinePlus health-topic XML: official NLM data, downloadable and reusable with attribution.
- NCCIH public-domain fact sheets: evidence and safety information for complementary approaches.
- PubMed Central Open Access subset: only articles whose per-item license permits the intended use.
- PubMedQA: MIT-licensed research QA; use mainly for evaluation, not consumer-care policy.
- WHO traditional-medicine strategy/guidance: policy/evidence framing; verify WHO reuse terms for each
  document before ingesting full text.

## Excluded by default

- MIMIC and other credentialed clinical datasets: not needed for this consumer text use case.
- HealthCareMagic/ChatDoctor-style scraped patient conversations: privacy, consent, and quality risk.
- Unlicensed exam-question collections: uncertain copyright and weak fit for consumer triage.
- Raw social media or forum data.

## Privacy and retention

Do not use real user conversations for training. A separate, explicit research consent process,
de-identification assessment, governance approval, deletion path, and data-use agreement would be
required. Application logs must never contain prompt or answer text.

Generated corpora, embeddings, checkpoints, and experiment logs are ignored by Git. The configured
personal artifact store is the additive-only `Anlu Health` folder in Google Drive, with separate
`Datasets`, `Model Adapters`, `Evaluation Reports`, and `Release Artifacts` subfolders. Training jobs
must create timestamped run directories and must not delete or overwrite existing Drive content.
DVC tracks hashes and lineage without placing large files on the laptop or in GitHub. Access to the
Drive folder remains private to the account owner unless they explicitly change sharing settings.
