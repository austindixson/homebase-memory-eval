# Public AML deployment

The participant-operated Textual API is deployed on the always-on Ghost128 host.
The initial setup on the operator's personal Mac was stopped and its public route removed.

| Setting | Deployed value |
| --- | --- |
| Add | `https://ghost128s-macbook-pro.tailf163d8.ts.net/aml-memory/add` |
| Search | `https://ghost128s-macbook-pro.tailf163d8.ts.net/aml-memory/search` |
| Health | `https://ghost128s-macbook-pro.tailf163d8.ts.net/aml-memory/health` |
| Authentication | `Authorization: Token <Memory System Key>` |
| Public source commit | `4aec8864ec39166681e1ff876b9aead390bbcd91` |
| Candidate tag | `v0.1.1-candidate-only` |
| Memory code hash reported by health | `2d57f688d71b` |
| Profile | `model-free`, full recall enabled |
| LLM / embedding components | None in Add or Search |
| Recall budget | 100,000 estimated tokens; 6,000-character overflow blocks |
| Runtime | Python 3.11.16, dependencies pinned in `requirements.txt` |
| Execution | One CPU-only worker; requests serialize inside the memory backend |
| Storage | Persistent SQLite; separate database for each exact `user_id` |

The service uses an isolated release directory and Python environment, binds to
loopback port 8755, and is published over HTTPS through Tailscale Funnel. A
dedicated macOS LaunchAgent restarts the process after a crash and starts it in
the logged-in user's session. Host availability, network connectivity and that
session remain operational dependencies. No GPU server or personal Home Base
database is connected to this service. This deployment incurred no Railway or
model API charges.

The Memory System Key is stored outside Git in a mode-0600 service configuration
and supplied separately to AML. It is different from the Leaderboard API Key
that AML issues after reviewing the access request. No key appears in these
instructions or the verification receipts. Request access logging is disabled.

## Verification

Receipts are in [`../../results/deployment_20261003/`](../../results/deployment_20261003/).
All final checks passed:

- Fixed commit, model-free profile and full-recall configuration reported by health.
- Missing and invalid credentials rejected.
- Synchronous Add echoes the request identifiers after persistence.
- Identical retries are idempotent; conflicting reuse of a request ID is rejected.
- All 100 synthetic source facts returned; `top_k` respected.
- A different `user_id` sees none of those facts.
- Previously stored facts survive a service restart.
- Authenticated Add/Search verified through a public DNS IP with TLS certificate
  validation, bypassing tailnet DNS routing.
- A 512-turn history exceeding the budget exercises overflow selection: the
  requested source anchor was retained and returned evidence estimated at
  99,465 tokens. Local Search took 0.181 seconds; median Add took 0.009 seconds.
  These synthetic measurements are not a throughput SLA or an AML score.

One initial public request received HTTP 502 because the operator overlapped it
with a deliberate process restart. That failed receipt is retained. The complete
post-restart check passed. An external web-reading tool could not access the
health URL; public-IP-forced curl checks subsequently verified actual internet
ingress instead.

## Start the frozen service elsewhere

Check out the fixed candidate commit, install `deploy/aml/requirements.txt` in an
isolated Python 3.11 or 3.12 environment, and supply these environment variables:

```text
AML_MEMORY_PROFILE=model-free
AML_FULL_RECALL=1
AML_RECALL_BUDGET_TOKENS=100000
AML_SOURCE_COMMIT=4aec8864ec39166681e1ff876b9aead390bbcd91
AML_MEMORY_DATA_DIR=<absolute persistent data directory>
AML_MEMORY_SECRET=<private random service key>
```

Start one worker from that checkout:

```sh
python -m uvicorn homebase.brain.aml_service:build_app --factory \
  --host 127.0.0.1 --port 8755 --workers 1 --no-access-log
```

Publish only that service through the chosen HTTPS reverse proxy. Keep its data
directory separate from personal memory or any other evaluation instance. Plan
for one request at a time; this deployment has not established a full-suite
concurrency or maximum-dataset capacity guarantee. AML evaluation data must be
handled and removed according to AML's retention rules after a run.

The verification helper lives on the documentation branch/main after deployment;
it does not change the frozen service code. With `AML_MEMORY_SECRET` supplied
privately to its environment:

```sh
python scripts/check_deployment.py \
  --base https://ghost128s-macbook-pro.tailf163d8.ts.net/aml-memory \
  --out .runs/deployment-check.json
```

This creates a fresh synthetic user. `--resume <previous-receipt.json>` additionally
checks that the prior user's source facts survived a restart. The helper makes
no reader or judge calls. It is not AML's official Smoke evaluation. The access
application and AML-issued key are still required before that official test.
