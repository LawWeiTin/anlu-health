# Model strategy

## Decision

Do not fine-tune only the original demonstration dataset. It is far too small to produce a safe
full-scale assistant, and the initial quality failures were safety-routing, retrieval, and validation
defects that fine-tuning could not repair. After those defects were corrected, a bounded open-data
pilot was authorized to measure whether adaptation helps without weakening the external safeguards.

## Path to a capable conversational assistant

1. Keep deterministic emergency triage outside the model and expand it through clinician-reviewed
   regression cases.
2. Evaluate capable instruction-tuned base models through a remote endpoint, with conversation
   context, a strong multilingual embedding model, a reranker, and approved-source RAG.
3. Use RAG for medical facts and freshness. The assistant must abstain when retrieval is weak and
   show the exact sources that support its answer.
4. Fine-tune only after collecting a much larger synthetic or properly de-identified, clinician- and
   TCM-practitioner-reviewed behavior dataset. Fine-tuning should teach response structure, tone,
   escalation, uncertainty, and refusal behavior, not serve as the medical knowledge base.
5. Run candidate training only in the prepared Google Colab workflow. Save timestamped adapters,
   manifests, and evaluations to the additive-only Google Drive project folder; never train or cache
   model weights on the laptop.
6. Promote a candidate only after held-out safety, citation, retrieval, multilingual, privacy, and
   human clinical-review gates pass against the unchanged base-model baseline.

## Experiment order

Keep the hosted base-model and RAG results as the unchanged comparison. The authorized QLoRA pilot
uses the pinned, filtered mixture in `training/open_datasets.yaml` and runs for one epoch in Colab.
Its adapter remains an unreviewed candidate until it beats the baseline without regressing emergency,
harmful-advice, citation, multilingual, privacy, or human-review gates.

MedGemma 1.5 4B remains a useful medical candidate, but its official model card says it has not been
evaluated or optimized for multi-turn applications. Benchmark it against a capable general
instruction model for conversational quality, and require the same external safety and grounding
controls for both.
