"""FastAPI app tests: /health endpoint + lifespan behaviour.

Lifespan is verified with a mocked DB pool and a mocked ``PGReflectionStore``
so the tests do not require a real PG. The real DB path is covered by
``test_db.py`` integration tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent_runtime.config import Settings
from agent_runtime.main import create_app


def _make_settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
    )


def test_health_returns_200_with_db_ok() -> None:
    with (
        patch("agent_runtime.main.create_pool") as mock_create_pool,
        patch("agent_runtime.main.PGReflectionStore") as mock_reflection_cls,
        patch("agent_runtime.main.run_migrations", new=AsyncMock(return_value=[])),
        patch("agent_runtime.main.db_ping", new=AsyncMock(return_value=True)),
    ):
        mock_pool = MagicMock()
        mock_pool.open = AsyncMock()
        mock_pool.close = AsyncMock()
        mock_create_pool.return_value = mock_pool

        mock_reflection = MagicMock()
        mock_reflection.ensure_schema = AsyncMock()
        mock_reflection_cls.return_value = mock_reflection

        app = create_app(_make_settings())
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"
            assert body["db"] == "ok"
            assert body["agents_count"] == 0
            assert body["jobs_count"] == 0
            assert "version" in body


def test_health_returns_degraded_when_db_down() -> None:
    with (
        patch("agent_runtime.main.create_pool") as mock_create_pool,
        patch("agent_runtime.main.PGReflectionStore") as mock_reflection_cls,
        patch("agent_runtime.main.run_migrations", new=AsyncMock(return_value=[])),
        patch("agent_runtime.main.db_ping", new=AsyncMock(return_value=False)),
    ):
        mock_pool = MagicMock()
        mock_pool.open = AsyncMock()
        mock_pool.close = AsyncMock()
        mock_create_pool.return_value = mock_pool
        mock_reflection = MagicMock()
        mock_reflection.ensure_schema = AsyncMock()
        mock_reflection_cls.return_value = mock_reflection

        app = create_app(_make_settings())
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "degraded"
            assert body["db"] == "error"


def test_lifespan_runs_migrations_and_ensures_reflection_schema() -> None:
    run_migrations_mock = AsyncMock(return_value=["001_initial.sql"])
    ensure_schema_mock = AsyncMock()

    with (
        patch("agent_runtime.main.create_pool") as mock_create_pool,
        patch("agent_runtime.main.PGReflectionStore") as mock_reflection_cls,
        patch("agent_runtime.main.run_migrations", new=run_migrations_mock),
        patch("agent_runtime.main.db_ping", new=AsyncMock(return_value=True)),
    ):
        mock_pool = MagicMock()
        mock_pool.open = AsyncMock()
        mock_pool.close = AsyncMock()
        mock_create_pool.return_value = mock_pool
        mock_reflection_cls.return_value = MagicMock(ensure_schema=ensure_schema_mock)

        app = create_app(_make_settings())
        with TestClient(app) as client:
            client.get("/health")

        mock_pool.open.assert_awaited_once()
        mock_pool.close.assert_awaited_once()
        ensure_schema_mock.assert_awaited_once()
        run_migrations_mock.assert_awaited_once()


def test_create_app_uses_provided_settings_version() -> None:
    settings = _make_settings()
    settings = settings.model_copy(update={"APP_VERSION": "99.99.99"})

    with (
        patch("agent_runtime.main.create_pool") as mock_create_pool,
        patch("agent_runtime.main.PGReflectionStore") as mock_reflection_cls,
        patch("agent_runtime.main.run_migrations", new=AsyncMock(return_value=[])),
        patch("agent_runtime.main.db_ping", new=AsyncMock(return_value=True)),
    ):
        mock_pool = MagicMock()
        mock_pool.open = AsyncMock()
        mock_pool.close = AsyncMock()
        mock_create_pool.return_value = mock_pool
        mock_reflection_cls.return_value = MagicMock(ensure_schema=AsyncMock())

        app = create_app(settings)
        assert app.version == "99.99.99"
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.json()["version"] == "99.99.99"


def test_create_app_has_health_route() -> None:
    with patch("agent_runtime.main.create_pool"), patch("agent_runtime.main.PGReflectionStore"):
        app = create_app(_make_settings())
    routes = {r.path for r in app.routes}  # type: ignore[attr-defined]
    assert "/health" in routes


@pytest.mark.parametrize("protocol", ["postgres://", "postgresql://"])
def test_settings_normalize_postgres_protocol(protocol: str) -> None:
    settings = Settings(  # type: ignore[call-arg]
        DATABASE_URL=f"{protocol}u:p@h:5432/db",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
    )
    assert settings.DATABASE_URL.startswith("postgresql://")
