# Model strategy

## Decision

Do not fine-tune the current dataset yet. The local mock provider is deliberately deterministic and
contains no general medical language intelligence, while the demonstration SFT set is far too small
to produce a safe full-scale assistant. The immediate quality failures are safety-routing, retrieval,
and validation defects; fine-tuning would not repair them reliably.

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

First benchmark remote base models without fine-tuning. Then compare RAG plus reranking. Only if the
remaining measured errors are behavioral and repeatable should a Colab QLoRA run be authorized. The
existing `training/medgemma_qlora_colab.ipynb` is therefore a gated later-stage experiment, not the
next repair step.

MedGemma 1.5 4B remains a useful medical candidate, but its official model card says it has not been
evaluated or optimized for multi-turn applications. Benchmark it against a capable general
instruction model for conversational quality, and require the same external safety and grounding
controls for both.
