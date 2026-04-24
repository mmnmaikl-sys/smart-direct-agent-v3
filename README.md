> **WARNING: DO NOT RUN `railway up` LOCALLY**
>
> Deploy is performed **only** via GitHub Actions (`.github/workflows/deploy.yml`, configured in Task 3). Running `railway up` from this folder bypasses all quality gates and caused the 24.04 incident that wiped 16 scheduled jobs in production. See `decisions.md` Decision 10.

# Smart Direct Agent v3

Autonomous Yandex.Direct management agent for 24bankrotsttvo.ru (БФЛ). Hypothesis-driven experimentation with budget caps per hypothesis type, 4-level decision engine (AUTO/NOTIFY/ASK/FORBIDDEN), trust-state machine (shadow → assisted → autonomous), 7 kill-switches.

Built on top of `agents-core` (ReAct loop, ToolRegistry, ReflectionStore, Langfuse observability).

## Setup

```bash
git clone git@github.com:mmnmaikl-sys/smart-direct-agent-v3.git
cd smart-direct-agent-v3
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Documentation

- [user-spec.md](./user-spec.md) — product requirements, acceptance criteria
- [tech-spec.md](./tech-spec.md) — architecture, decisions, implementation plan
- [decisions.md](./decisions.md) — post-task execution reports
- [code-research.md](./code-research.md) — implementation references (agents-core API, v2.1/v3.0 legacy)

## Deployment

See `.github/workflows/deploy.yml` (created in Task 3). Deploy is triggered by push to `main` + manual approval in environment=`production`. Local `railway up` is forbidden — see WARNING above.

## Development

- Python ≥ 3.12 (Railway default)
- Dependencies pinned by commit SHA (agents-core) + requirements.lock with hashes (Task 3)
- Pre-commit hooks (Task 2) block imports from `/tmp/sda-v2*` (legacy paths)
- CI runs: pytest, ruff, pyright, pip-audit, shellcheck, legal canary, grill audit
