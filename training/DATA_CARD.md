# Safety SFT data card

## Purpose

This 75-scenario project-authored set demonstrates the required conversational schema and teaches
response behavior: emergency escalation, diagnostic abstention, useful follow-up questions,
topic-matched retrieval, citation integrity, evidence separation, concise scope boundaries,
duration grounding, and herb/medicine caution. It is not a clinical knowledge corpus and remains
too small for a production fine-tune. It is
project-reviewed for schema and safety-policy consistency, but it has not yet received the required
physician, pharmacist, or registered TCM practitioner sign-off.

Version 15 adds ten independent synthetic scenarios for exact duration and medicine-name
preservation, explicit rejection of topic-mismatched sources, citation abstention,
TCM-versus-biomedical separation, pregnancy-product caution, and same-day hemoptysis navigation.
The held-out release prompts were not copied into training.

## Provenance and privacy

Examples are synthetic and contain no real patient conversations. Each record has a stable scenario
ID, behavior tags, and evidence-source keys. `audit_behavior_dataset.py` checks topic coverage,
source-key provenance, exact/near-duplicate held-out leakage, role/schema validity, and obvious
identifiers. It also rejects overlong authored answers so the target style remains concise.
The remote training job additionally renders every record through the MedGemma system/user/assistant
format, rejects reserved chat tokens or internal-instruction language in targets, and verifies that
the supervised completion includes exactly one end-of-turn boundary.
`prepare_dataset.py` rejects obvious email addresses, phone numbers, and medical-record identifiers.
Do not add real user chat.

## Production extension

A qualified clinician and registered TCM practitioner should author and independently review a
larger multilingual set. Split by scenario template before generation to avoid near-duplicate leakage.
Keep the held-out safety evaluation entirely separate. The open-data pilot may use the non-test
portion of PubMedQA for evidence-conditioned research reasoning, but it is not a substitute for
consumer safety evaluation. See `OPEN_DATA_CARD.md` and `open_datasets.yaml` for the pinned,
license-audited pilot mixture and exclusions.

## Limitations

Synthetic text cannot represent the full variety of symptom descriptions, literacy, language,
culture, disability, or adversarial inputs. Fine-tuning can also weaken base-model safety behavior;
the deterministic safety layer and post-training evaluation remain mandatory.
