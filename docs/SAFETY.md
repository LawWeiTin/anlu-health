# Medical safety case

## Product claim

Anlu Health provides source-linked health education and care-navigation suggestions. It does not
provide a diagnosis, personalized treatment, a prescription, or a guarantee of accuracy. The UI,
API, prompt, and evaluation gates all enforce that scope.

## Layered controls

1. **Deterministic escalation:** high-recall English and Chinese patterns for breathing difficulty,
   stroke signs, severe allergic reaction, uncontrolled bleeding, coughing up blood, self-harm, and
   concerning lumps. Any coughing up blood is at least urgent; larger amounts or associated breathing,
   chest, fainting, or fast-heartbeat symptoms bypass generation as an emergency.
   Emergency responses bypass retrieval and generation.
2. **Approved-source RAG:** chunks carry publisher, URL, license, evidence tier, language, review date,
   expiry date, checksum, and approval state. Retrieval fuses pgvector semantic similarity with
   indexed full-text/BM25 keyword evidence, filters stale or unapproved content, and abstains when
   candidates lack sufficient lexical or strong semantic support. Regex topic routing does not
   authorize evidence.
3. **Bounded generation:** training and inference use the same system/user chat template. Generation
   stops on Gemma's `<end_of_turn>` token as well as generic EOS; the prompt forbids diagnosis,
   dosing, fabricated citations, urgency downgrades, and presentation of traditional concepts as
   established biomedical mechanisms.
4. **Output validation:** only citations present in the retrieved set survive. Generated answers
   without a valid approved citation, with personalized numeric dosing, diagnostic certainty, or
   leaked prompt/meta instructions, overlong answers, or incomplete final sentences fail closed and
   are replaced by a deterministic abstention.
5. **Privacy controls:** opaque revocable sessions; no raw health text in logs; history is off by
   default and app-level encrypted when explicitly enabled.
6. **Release gates:** emergency recall, abstention, citation validity, traditional-medicine
   separation, response termination, scope adherence, security tests, and clinician review.

## Lump questions

The assistant asks about location, duration, growth, pain, redness/warmth, mobility, texture, fever,
night sweats, unexplained weight loss, and swallowing/breathing effects. It recommends in-person
assessment for persistent, growing, hard/fixed, breast/testicular, or otherwise concerning lumps.
It never labels a photo or text description as benign or malignant.

## Traditional Chinese medicine

- Traditional pattern language is labelled as a historical/practice framework, not a confirmed
  biomedical explanation.
- Effectiveness claims require a named evidence source and evidence tier.
- Herbal suggestions never include personalized formulas or doses.
- The assistant checks for pregnancy, allergies, liver/kidney disease, surgery, and concurrent
  medicines and warns that herb-drug evidence is incomplete.
- It advises consultation with a licensed practitioner and pharmacist/doctor, especially for
  anticoagulants, transplant medicines, chemotherapy, digoxin, and other narrow-therapeutic-index
  medicines.

## Clinical release checklist

- [ ] Intended users, countries, languages, exclusions, and emergency numbers are documented.
- [ ] A physician and registered TCM practitioner approve prompts, red flags, and test cases.
- [ ] Multilingual evaluations include age, sex, pregnancy, disability, and skin-tone considerations.
- [ ] Sensitivity/recall is measured for emergency scenarios; false reassurance is manually reviewed.
- [ ] RAG source licenses, freshness, and evidence tiers are verified by a named owner.
- [ ] Model endpoint, model/adaptor hashes, prompt version, data manifest, and evaluation report match.
- [ ] Security, privacy, legal/regulatory, accessibility, and incident-response reviews are complete.
- [ ] A rollback target exists and the deploy has a kill switch.
- [ ] Users can report harmful answers without submitting additional personal health data.

## Known limitations

- Rule-based triage cannot cover every wording or language.
- RAG can retrieve incomplete or outdated evidence despite freshness controls.
- Open-weight models can hallucinate, overstate confidence, or behave differently after fine-tuning.
- Medical benchmark accuracy does not measure safe consumer symptom triage.
- Traditional-medicine evidence is uneven; absence of evidence is not proof of benefit or safety.
