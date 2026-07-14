# Safety SFT data card

## Purpose

This small project-authored set demonstrates the required conversational schema and teaches response
behavior: emergency escalation, diagnostic abstention, useful follow-up questions, evidence
separation, and herb/medicine caution. It is not a clinical knowledge corpus and is far too small for
a production fine-tune.

## Provenance and privacy

Examples are synthetic and contain no real patient conversations. `prepare_dataset.py` rejects
obvious email addresses, phone numbers, and medical-record identifiers. Do not add real user chat.

## Production extension

A qualified clinician and registered TCM practitioner should author and independently review a
larger multilingual set. Split by scenario template before generation to avoid near-duplicate leakage.
Keep the held-out safety evaluation entirely separate. Use PubMedQA only as a research-reasoning
benchmark, not as a substitute for consumer safety evaluation.

## Limitations

Synthetic text cannot represent the full variety of symptom descriptions, literacy, language,
culture, disability, or adversarial inputs. Fine-tuning can also weaken base-model safety behavior;
the deterministic safety layer and post-training evaluation remain mandatory.

