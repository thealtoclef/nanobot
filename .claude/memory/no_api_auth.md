---
name: No API Authentication
description: Nanobot is IaC-deployed agentic workflow tool — no web UI, no API auth needed
type: feedback
---

No bearer token auth, no gateway hardening, no public-facing HTTP server concerns.

**Why:** Nanobot is an agentic workflow tool, not a service with an API. All config from config.json + env vars. IaC approach, stateless deployment.

**How to apply:** Don't implement API authentication, gateway rate limiting, or web UI features. Focus on the agent runtime itself.
