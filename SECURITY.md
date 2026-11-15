# Security Policy

## Supported version

Security updates are applied to the latest version on the `main` branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not
open a public issue for a suspected vulnerability and do not include face
images, enrolment crops, biometric models, credentials, or personal data in a
report.

Include:

- affected command or module;
- reproduction steps using synthetic or public benchmark data;
- expected and observed behavior;
- security impact; and
- a suggested mitigation, if available.

## Security boundaries

- Interactive `login` enforces active liveness and persistent rate limiting.
- Low-level `verify` is an evaluation primitive, not an authentication boundary.
- DPAPI ciphertext is bound to the current Windows account. It does not protect
  data from code already executing as that user.
- The liveness challenge is not certified presentation-attack detection.
- A cohort FAR measured on benchmark data is not a deployment guarantee.
