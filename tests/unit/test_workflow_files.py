"""Task 3 TDD: GitHub Actions workflows are valid YAML with required structure."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DEPLOY_YML = REPO_ROOT / ".github" / "workflows" / "deploy.yml"
DEPENDABOT_YML = REPO_ROOT / ".github" / "dependabot.yml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_ci_workflow_has_all_steps() -> None:
    """CI workflow defines 7 jobs covering the full verification pipeline."""
    ci = _load(CI_YML)
    job_names = set(ci.get("jobs", {}).keys())
    expected = {
        "lint",
        "typecheck",
        "test",
        "supply-chain-audit",
        "shellcheck",
        "legal-canary",
        "grill-audit",
    }
    missing = expected - job_names
    assert not missing, f"CI workflow missing jobs: {missing}"


def test_deploy_workflow_uses_token_secret() -> None:
    """RAILWAY_TOKEN must come from ${{ secrets.RAILWAY_TOKEN }}, never hardcoded."""
    raw = DEPLOY_YML.read_text()
    assert "${{ secrets.RAILWAY_TOKEN }}" in raw, "deploy.yml must reference secrets.RAILWAY_TOKEN"
    # Defense-in-depth: search for common hardcoding anti-patterns
    assert 'RAILWAY_TOKEN="' not in raw, "RAILWAY_TOKEN must not be hardcoded (double-quoted)"
    assert "RAILWAY_TOKEN='" not in raw, "RAILWAY_TOKEN must not be hardcoded (single-quoted)"


def test_deploy_workflow_has_production_environment() -> None:
    """Deploy job runs in `production` environment (gating via required reviewers)."""
    deploy = _load(DEPLOY_YML)
    deploy_job = deploy["jobs"]["deploy"]
    env = deploy_job.get("environment")
    assert env is not None, "deploy job must specify environment"
    env_name = env if isinstance(env, str) else env.get("name")
    assert env_name == "production", f"environment must be 'production', got {env_name!r}"


def test_dependabot_has_three_ecosystems() -> None:
    """Dependabot covers pip, github-actions, docker."""
    dependabot = _load(DEPENDABOT_YML)
    ecosystems = {u["package-ecosystem"] for u in dependabot.get("updates", [])}
    expected = {"pip", "github-actions", "docker"}
    assert expected.issubset(ecosystems), f"Dependabot missing ecosystems: {expected - ecosystems}"
