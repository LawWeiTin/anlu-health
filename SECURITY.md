# Security policy

This project handles potentially sensitive health text. Do not open a public issue containing health
information, credentials, logs, or screenshots with personal data.

## Production requirements

- Keep chat retention disabled unless app-level encryption is configured and users explicitly opt in.
- Use HTTPS-only cookies, a managed secret store, database backups, least-privilege service accounts,
  dependency scanning, and an incident-response process.
- Never log prompts, answers, session tokens, passwords, or model/API credentials.
- Rotate model, database, session, and encryption credentials after any suspected exposure.
- Run a professional penetration test and privacy/regulatory review before serving real users.

Report vulnerabilities privately to the repository owner. Include reproduction steps without real
patient data.

