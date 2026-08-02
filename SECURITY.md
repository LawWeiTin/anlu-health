# Security policy

## Supported version

Security fixes target the current `main` branch. Historical research candidates are retained only
for reproducibility and are not supported deployments.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository. If that option is unavailable,
contact the repository owner privately through their GitHub profile. Do not open a public issue
containing exploit details, credentials, endpoint locations, or real health information.

Include the affected commit, a minimal reproduction using synthetic data, the expected impact, and
any suggested mitigation. Never test against infrastructure or accounts you do not own or have
explicit permission to assess.

## Credential handling

- Never commit `.env` files, Hugging Face tokens, API keys, database passwords, private keys, or
  private endpoint URLs.
- Use fine-grained, least-privilege credentials in the deployment platform's secret manager.
- Run `python scripts/check_public_release.py` before publishing or sharing a branch.
- If a real credential is ever committed, revoke and rotate it immediately. Removing it from the
  latest commit is not sufficient because Git history and cached pull-request refs may retain it.

## Medical safety boundary

A software-security report is not a channel for medical advice or emergencies. This project is an
educational prototype and must not be relied on for diagnosis, treatment, or emergency response.
