# Safety SFT data card

## Purpose

This 136-scenario project-authored set demonstrates the required conversational schema and teaches
response behavior: emergency escalation, diagnostic abstention, useful follow-up questions,
topic-matched retrieval, citation integrity, evidence separation, concise scope boundaries,
duration grounding, and herb/medicine caution. It is not a clinical knowledge corpus and remains
too small for a production fine-tune. It is
project-reviewed for schema and safety-policy consistency, but it has not yet received the required
physician, pharmacist, or registered TCM practitioner sign-off.

Version 15 adds ten independent synthetic scenarios for exact duration and medicine-name
preservation, explicit rejection of topic-mismatched sources, citation abstention,
TCM-versus-biomedical separation, pregnancy-product caution, and same-day hemoptysis navigation.
Version 18 adds four independent duration-fidelity scenarios and requires the response's first
sentence to preserve the user's stated duration and time unit before care navigation.
Version 21 adds three immediate self-harm scenarios that require means separation, nearby-person
support, and emergency contact, plus three urgent source-mismatch scenarios that combine the
evidence-boundary disclosure and care action in the first sentence.
Version 22 adds three more urgent source-mismatch scenarios and three non-urgent mismatch scenarios.
Every target starts with the explicit source mismatch and unusability before giving care navigation.
Version 23 adds six structured supplied-source examples. Each target identifies what the supplied
source actually covers before rejecting it as mismatched, which prevents symptom words elsewhere in
the prompt from being mistaken for source content. It also adds two source-absence paraphrases for
conservative citation refusal.
Version 24 adds three matched evidence-state contrast sets. Each clinical topic appears with no
retrieved source, a clearly unrelated supplied source, and a relevant supplied source. Targets
explicitly preserve the distinction: absence refuses invention, mismatch names the supplied topic
before rejecting it, and relevant evidence supports only bounded general guidance.
Version 25 adds six independent source-mismatch examples using the structured field labels emitted
by the application and evaluation harness. The topics and symptoms are distinct from the held-out
cases, while every target repeats the supplied topic and rejects it before care navigation. The
release evaluator also accepts the strictly equivalent direct denials "cannot support" and
"does not support" without accepting positive support claims or unrelated negations.
Version 28 leaves the behavior data and held-out suite unchanged. It fixes the evaluator's bounded
negation window so "not considered automatically safe during pregnancy" is treated as a denial,
while a later positive reassurance in the same answer still fails.
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
