# Open-data QLoRA pilot

## Intended experiment

This bundle supports a small text-only QLoRA experiment on `google/medgemma-1.5-4b-it`. It tests
whether a license-audited mixture improves evidence-conditioned biomedical answers and general
consumer-health explanations without weakening the external safety layer. It is not a production
clinical model and is not intended to learn diagnosis or personalized treatment.

## Included training sources

- **MedQuAD** (`CC BY 4.0`): a maximum of 600 examples from answer-bearing NIH collections.
  Only lower-risk educational question types are accepted. Treatment, diagnosis, dosing, prognosis,
  procedure, and side-effect targets are not selected for this pilot.
- **PubMedQA PQA-L** (`MIT`): up to 150 expert-labeled, abstract-conditioned examples. Every PMID in
  the official test-ground-truth file is excluded from training and remains available for evaluation.
- **Anlu authored safety set**: synthetic project-authored behavior examples covering small talk,
  topic mismatch, citation integrity, red-flag navigation, lump assessment, medicine-herb safety,
  TCM evidence separation, evidence-state contrasts, and English/Chinese prompts. No real user
  messages. Clinical review is still pending, so these examples cannot independently authorize
  production use. During training, this behavior subset receives a sampling weight of twenty-four
  to keep generic biomedical QA from overwhelming the intended conversational behavior.

MedMCQA is recorded but evaluation-only because entrance-exam multiple-choice responses are not the
target conversational behavior. Scraped patient consultations, credentialed clinical records, and
datasets without explicit licensing are excluded.

## Processing and storage

The source repositories are cloned at exact commit hashes into ephemeral Kaggle storage. The pipeline
normalizes text, rejects obvious identifiers and medication-like dosing targets, deduplicates by
question, caps open-data targets at complete source-sentence boundaries, and splits by source group
so one source group does not occur in both train and validation.
The private Kaggle run writes only the adapter, progress log, attribution, source revisions, filter
counts, release evaluation, and SHA-256 checksums to a new UTC-timestamped output directory. It
never writes base-model weights or source-corpus clones to the laptop.

## Known limitations

MedQuAD is a legacy corpus and can contain information that has since changed. Fine-tuned weights
must therefore never replace current approved-source RAG. PubMedQA measures research reasoning, not
personal care navigation. Neither source validates multi-turn conversation, Chinese-language care,
TCM safety, emergency recall, calibration, or clinical usefulness. Those remain separate evaluation
and human-review gates.
