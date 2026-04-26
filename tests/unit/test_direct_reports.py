"""Tests for DirectReportAdapter TSV → list[dict] parser.

The Direct /reports endpoint returns TSV with a header row + data rows
(no trailing total row when IncludeVAT=NO + IncludeDiscount=NO).
"""

from __future__ import annotations

from agent_runtime.tools.direct_reports import _parse_tsv_rows


def test_parse_minimal_tsv() -> None:
    tsv = (
        "CampaignId\tAdGroupId\tQuery\tImpressions\tClicks\tCost\n"
        "709353005\t1\tбанкротство\t100\t5\t1500000\n"
    )
    rows = _parse_tsv_rows(tsv)
    assert len(rows) == 1
    assert rows[0]["query"] == "банкротство"
    assert rows[0]["campaignid"] == "709353005"
    assert rows[0]["clicks"] == "5"


def test_parse_skips_metadata_lines() -> None:
    """Direct prepends metadata («Report name», «Date range», ...) before header."""
    tsv = (
        '"Report name: sqr_2026-04-19_2026-04-26"\n'
        '"Date range: 2026-04-19 - 2026-04-26"\n'
        "CampaignId\tQuery\tClicks\n"
        "709353005\tмой арбитр\t3\n"
        "709353005\tкак подать на банкротство\t1\n"
    )
    rows = _parse_tsv_rows(tsv)
    assert len(rows) == 2
    assert rows[0]["query"] == "мой арбитр"
    assert rows[1]["query"] == "как подать на банкротство"


def test_parse_empty_tsv() -> None:
    assert _parse_tsv_rows("") == []
    assert _parse_tsv_rows("\n\n") == []


def test_parse_no_header_returns_empty() -> None:
    """If no row containing both 'Query' and a tab — refuse to guess."""
    rows = _parse_tsv_rows("just garbage\nno tabs at all")
    assert rows == []


def test_parse_strips_total_row() -> None:
    """Direct may include a 'Total rows' summary at the end — skip it."""
    tsv = "CampaignId\tQuery\tClicks\n709353005\tбанкротство\t5\nTotal rows: 1\n"
    rows = _parse_tsv_rows(tsv)
    assert len(rows) == 1
    assert rows[0]["query"] == "банкротство"


def test_parse_lowercase_field_names() -> None:
    """Field names are lowercased for compat with query_analyzer (row.get('query'))."""
    tsv = "CampaignId\tQuery\tImpressions\n123\tтест\t10\n"
    rows = _parse_tsv_rows(tsv)
    assert "query" in rows[0]
    assert "campaignid" in rows[0]
    assert "Query" not in rows[0]
