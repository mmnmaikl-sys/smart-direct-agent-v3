"""Unit tests for agent_runtime.signal_detector (Task 11).

SignalDetector is dependency-injected, so tests pass in tiny stubs for the
Direct / Metrika / Bitrix callouts — no network, deterministic. Parallel
``asyncio.gather`` isolation is verified too: one detector raising must not
silence the other five.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agent_runtime.config import Settings
from agent_runtime.models import Signal, SignalType
from agent_runtime.signal_detector import SignalDetector

_PROTECTED = [708978456, 708978457, 708978458]


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        PROTECTED_CAMPAIGN_IDS=_PROTECTED,
        TARGET_CPA=5000,
        DAILY_BUDGET_LIMIT=3000,
        PROTECTED_LANDING_URLS=[
            "https://example.test/ad/one",
            "https://example.test/ad/two",
        ],
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _mock_pool() -> MagicMock:
    # signal_detector does not currently read/write sda_state (cache TBD) —
    # a plain MagicMock is enough to satisfy the constructor.
    return MagicMock()


def _build_detector(
    *,
    direct: Any = None,
    metrika: Any = None,
    bitrix: Any = None,
    http_handler=None,
    settings: Settings | None = None,
) -> tuple[SignalDetector, httpx.AsyncClient]:
    direct = direct or SimpleNamespace(
        get_campaign_stats=AsyncMock(return_value={"cost": 0, "clicks": 0}),
        get_campaigns=AsyncMock(return_value=[]),
    )
    metrika = metrika or SimpleNamespace(
        get_stats=AsyncMock(return_value={"data": []}),
    )
    bitrix = bitrix or SimpleNamespace(
        get_lead_list=AsyncMock(return_value=[]),
    )

    def default_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>" + b"x" * 2000 + b"</html>")

    http = httpx.AsyncClient(transport=httpx.MockTransport(http_handler or default_handler))
    detector = SignalDetector(
        pool=_mock_pool(),
        direct=direct,
        metrika=metrika,
        bitrix=bitrix,
        http=http,
        settings=settings or _settings(),
    )
    return detector, http


# ---- budget threshold ------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_threshold_warning() -> None:
    # First campaign trips the 70% warning; others are at zero.
    direct = SimpleNamespace(
        get_campaign_stats=AsyncMock(
            side_effect=[
                {"cost": 2200, "clicks": 15},  # 73% of 3000 -> warning
                {"cost": 0, "clicks": 0},
                {"cost": 0, "clicks": 0},
            ]
        ),
        get_campaigns=AsyncMock(return_value=[]),
    )
    detector, http = _build_detector(direct=direct)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    budget = [s for s in signals if s.type == SignalType.BUDGET_THRESHOLD]
    assert len(budget) == 1
    assert budget[0].severity == "warning"
    assert budget[0].data["pct"] == 73


@pytest.mark.asyncio
async def test_budget_threshold_critical() -> None:
    direct = SimpleNamespace(
        get_campaign_stats=AsyncMock(
            return_value={"cost": 2800, "clicks": 50}  # >90% all campaigns
        ),
        get_campaigns=AsyncMock(return_value=[]),
    )
    detector, http = _build_detector(direct=direct)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    critical = [
        s for s in signals if s.type == SignalType.BUDGET_THRESHOLD and s.severity == "critical"
    ]
    assert len(critical) == 3  # one per protected campaign


@pytest.mark.asyncio
async def test_spend_no_clicks() -> None:
    direct = SimpleNamespace(
        get_campaign_stats=AsyncMock(
            side_effect=[
                {"cost": 600, "clicks": 0},
                {"cost": 0, "clicks": 0},
                {"cost": 0, "clicks": 0},
            ]
        ),
        get_campaigns=AsyncMock(return_value=[]),
    )
    detector, http = _build_detector(direct=direct)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    spend = [s for s in signals if s.type == SignalType.SPEND_NO_CLICKS]
    assert len(spend) == 1
    assert spend[0].data["cost_today"] == 600


# ---- bounce ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_high_bounce() -> None:
    metrika = SimpleNamespace(
        get_stats=AsyncMock(
            return_value={
                "data": [
                    {
                        "dimensions": [
                            {"name": "https://24bankrotsttvo.ru/pages/ad/bankrotstvo-v4.html"}
                        ],
                        "metrics": [120, 72],
                    },
                    {
                        "dimensions": [
                            {"name": "https://24bankrotsttvo.ru/pages/ad/price-v4.html"}
                        ],
                        "metrics": [150, 82],
                    },
                ]
            }
        ),
    )
    detector, http = _build_detector(metrika=metrika)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    by_sev = {s.severity: s for s in signals if s.type == SignalType.HIGH_BOUNCE}
    assert by_sev["warning"].data["bounce_pct"] == 72
    assert by_sev["critical"].data["bounce_pct"] == 82


@pytest.mark.asyncio
async def test_high_bounce_skips_low_visits() -> None:
    metrika = SimpleNamespace(
        get_stats=AsyncMock(
            return_value={
                "data": [
                    {
                        "dimensions": [{"name": "https://example.com/p"}],
                        "metrics": [10, 90],  # bounce=90 but visits<50
                    }
                ]
            }
        ),
    )
    detector, http = _build_detector(metrika=metrika)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    assert not [s for s in signals if s.type == SignalType.HIGH_BOUNCE]


# ---- zero leads / high CPA -------------------------------------------------


@pytest.mark.asyncio
async def test_zero_leads() -> None:
    direct = SimpleNamespace(
        get_campaign_stats=AsyncMock(return_value={"cost": 600, "clicks": 50}),
        get_campaigns=AsyncMock(return_value=[]),
    )
    bitrix = SimpleNamespace(get_lead_list=AsyncMock(return_value=[]))
    detector, http = _build_detector(direct=direct, bitrix=bitrix)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    zero = [s for s in signals if s.type == SignalType.ZERO_LEADS]
    assert len(zero) == 1
    assert zero[0].severity == "critical"


@pytest.mark.asyncio
async def test_high_cpa() -> None:
    # 3 campaigns × 20000 cost = 60000 total. 5 our leads → CPA 12000 > 2×target(5000)
    direct = SimpleNamespace(
        get_campaign_stats=AsyncMock(return_value={"cost": 20000, "clicks": 30}),
        get_campaigns=AsyncMock(return_value=[]),
    )
    leads = [{"ID": str(i), "SOURCE_DESCRIPTION": "24bankrotsttvo/pages/ad"} for i in range(5)]
    bitrix = SimpleNamespace(get_lead_list=AsyncMock(return_value=leads))
    detector, http = _build_detector(direct=direct, bitrix=bitrix)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    cpa = [s for s in signals if s.type == SignalType.HIGH_CPA]
    assert len(cpa) == 1
    assert cpa[0].data["cpa"] == 12000


# ---- landing health --------------------------------------------------------


@pytest.mark.asyncio
async def test_landing_broken_status() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        # First URL 500, second 200 with body.
        if req.url.path.endswith("/one"):
            return httpx.Response(500, content=b"")
        return httpx.Response(200, content=b"x" * 2000)

    detector, http = _build_detector(http_handler=handler)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    broken = [s for s in signals if s.type == SignalType.LANDING_BROKEN]
    assert len(broken) == 1
    assert broken[0].data["status"] == 500


@pytest.mark.asyncio
async def test_landing_small_body_treated_as_broken() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"tiny")  # <1000 bytes

    detector, http = _build_detector(http_handler=handler)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    broken = [s for s in signals if s.type == SignalType.LANDING_BROKEN]
    assert len(broken) == 2  # both URLs


@pytest.mark.asyncio
async def test_landing_transport_error_becomes_broken() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    detector, http = _build_detector(http_handler=handler)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    broken = [s for s in signals if s.type == SignalType.LANDING_BROKEN]
    assert len(broken) == 2
    assert all("error" in s.data for s in broken)


# ---- garbage queries -------------------------------------------------------


@pytest.mark.asyncio
async def test_garbage_queries() -> None:
    tsv = (
        "Query\tCost\tConversions\tClicks\n"
        "бесплатная стрижка\t8000\t0\t12\n"
        "банкротство физлиц\t3000\t5\t40\n"  # conversions>0 → not waste
    )
    # Override get_campaign_stats to return TSV for the garbage-queries path,
    # while budget path still reads cost/clicks from the same method. In the
    # test we only care about this run's first call output.
    direct = SimpleNamespace(
        get_campaign_stats=AsyncMock(
            side_effect=lambda campaign_id, **_: (
                {"tsv": tsv} if campaign_id == _PROTECTED[0] else {"cost": 0, "clicks": 0}
            )
        ),
        get_campaigns=AsyncMock(return_value=[]),
    )
    detector, http = _build_detector(direct=direct)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    garbage = [s for s in signals if s.type == SignalType.GARBAGE_QUERIES]
    assert len(garbage) == 1
    assert garbage[0].data["count"] == 1
    assert garbage[0].data["top"][0]["query"] == "бесплатная стрижка"


# ---- campaign state --------------------------------------------------------


@pytest.mark.asyncio
async def test_campaign_stopped_in_work_hours(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime as real_datetime

    from agent_runtime import signal_detector as module

    class _FrozenNow:
        @staticmethod
        def now_msk() -> real_datetime:
            return real_datetime(2026, 4, 24, 14, 0, 0, tzinfo=module.MSK)

    monkeypatch.setattr(module, "_now_msk", _FrozenNow.now_msk)
    direct = SimpleNamespace(
        get_campaign_stats=AsyncMock(return_value={"cost": 0, "clicks": 0}),
        get_campaigns=AsyncMock(return_value=[{"Id": 708978456, "State": "SUSPENDED"}]),
    )
    detector, http = _build_detector(direct=direct)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    stopped = [s for s in signals if s.type == SignalType.CAMPAIGN_STOPPED]
    assert len(stopped) == 1
    assert stopped[0].data["campaign_id"] == 708978456


@pytest.mark.asyncio
async def test_campaign_stopped_night_no_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime as real_datetime

    from agent_runtime import signal_detector as module

    monkeypatch.setattr(
        module,
        "_now_msk",
        lambda: real_datetime(2026, 4, 24, 2, 0, 0, tzinfo=module.MSK),
    )
    direct = SimpleNamespace(
        get_campaign_stats=AsyncMock(return_value={"cost": 0, "clicks": 0}),
        get_campaigns=AsyncMock(return_value=[{"Id": 708978456, "State": "SUSPENDED"}]),
    )
    detector, http = _build_detector(direct=direct)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    assert not [s for s in signals if s.type == SignalType.CAMPAIGN_STOPPED]


# ---- exception isolation --------------------------------------------------


@pytest.mark.asyncio
async def test_api_error_on_direct_exception() -> None:
    direct = SimpleNamespace(
        get_campaign_stats=AsyncMock(side_effect=httpx.HTTPError("503")),
        get_campaigns=AsyncMock(return_value=[]),
    )
    detector, http = _build_detector(direct=direct)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    api_errors = [s for s in signals if s.type == SignalType.API_ERROR]
    # _check_budget and _check_zero_leads both call get_campaign_stats → 2 errors
    sources = {s.data["source"] for s in api_errors}
    assert {"budget", "zero_leads"}.issubset(sources)


@pytest.mark.asyncio
async def test_parallel_detection_isolates_failures() -> None:
    metrika = SimpleNamespace(get_stats=AsyncMock(side_effect=RuntimeError("metrika down")))
    direct = SimpleNamespace(
        get_campaign_stats=AsyncMock(return_value={"cost": 0, "clicks": 0}),
        get_campaigns=AsyncMock(return_value=[]),
    )
    detector, http = _build_detector(direct=direct, metrika=metrika)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    sources = {s.data.get("source") for s in signals if s.type == SignalType.API_ERROR}
    assert "bounce" in sources
    # Other detectors still ran — no API_ERROR for 'budget' / 'landing_health'
    assert "budget" not in sources
    assert "landing_health" not in sources


# ---- config-source-of-truth -----------------------------------------------


def test_metrika_counter_not_hardcoded_in_module() -> None:
    source_path = (
        Path(__file__).resolve().parent.parent.parent / "agent_runtime" / "signal_detector.py"
    )
    text = source_path.read_text(encoding="utf-8")
    assert "107734488" not in text, "METRIKA_COUNTER_ID must come from Settings only"


def test_no_legacy_urllib_or_app_imports() -> None:
    source_path = (
        Path(__file__).resolve().parent.parent.parent / "agent_runtime" / "signal_detector.py"
    )
    text = source_path.read_text(encoding="utf-8")
    # Look for actual import lines, not docstring mentions.
    code_lines = [
        line for line in text.splitlines() if line.lstrip().startswith(("import ", "from "))
    ]
    for line in code_lines:
        assert "urllib.request" not in line, f"urllib.request import found: {line}"
        assert not line.startswith("from app."), f"legacy app.* import found: {line}"


# ---- signal ts is MSK-aware -----------------------------------------------


@pytest.mark.asyncio
async def test_signal_ts_is_tzaware_msk() -> None:
    direct = SimpleNamespace(
        get_campaign_stats=AsyncMock(return_value={"cost": 2800, "clicks": 50}),
        get_campaigns=AsyncMock(return_value=[]),
    )
    detector, http = _build_detector(direct=direct)
    try:
        signals = await detector.detect_all()
    finally:
        await http.aclose()
    assert signals  # at least one budget signal
    from agent_runtime.signal_detector import MSK

    for sig in signals:
        assert isinstance(sig, Signal)
        assert sig.ts.tzinfo is not None
        assert sig.ts.utcoffset() == MSK.utcoffset(sig.ts)
