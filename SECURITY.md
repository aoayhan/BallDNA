# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's **Security → Report a vulnerability** feature for this repository. Do not open a public issue containing a secret, exploit, personal data, or instructions that could harm the public app.

Include the affected page, reproduction steps, impact, and a safe way to contact you. Please do not access other users' data, degrade availability, or run destructive tests.

## Deployment notes

- No credentials belong in Git. Local `.env` and `.streamlit/secrets.toml` files are ignored; deployment secrets must use Streamlit's secrets settings.
- The public application does not need an API key and does not accept file uploads.
- Inference data and model artifacts committed to the public repository are public by design. They must not contain confidential information.
- Streamlit Community Cloud terminates HTTPS and provides platform security controls. Application security remains a shared responsibility.
