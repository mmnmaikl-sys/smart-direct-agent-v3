-- smart-direct-agent-v3 — initial schema (Wave 1, Task 5).
-- Idempotent: every DDL uses IF NOT EXISTS (per Decision 3 — no schema_migrations table).
-- Applied by agent_runtime.db.run_migrations(); agent_reflections lives separately,
-- created by agents_core.memory.reflection.PGReflectionStore.ensure_schema().

BEGIN;

CREATE TABLE IF NOT EXISTS sda_state (
    key         TEXT PRIMARY KEY,
    value       JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT trust_level_enum CHECK (
        key <> 'trust_level'
        OR value #>> '{}' IN ('shadow', 'assisted', 'autonomous', 'FORBIDDEN_LOCK')
    )
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id                         TEXT PRIMARY KEY,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent                      TEXT NOT NULL,
    hypothesis_type            TEXT NOT NULL CHECK (
        hypothesis_type IN (
            'ad', 'neg_kw', 'image', 'landing',
            'new_camp', 'format_change', 'strategy_switch', 'account_level'
        )
    ),
    signals                    JSONB NOT NULL,
    hypothesis                 TEXT NOT NULL,
    reasoning                  TEXT NOT NULL,
    actions                    JSONB NOT NULL,
    expected_outcome           TEXT NOT NULL,
    budget_cap_rub             INTEGER NOT NULL CHECK (budget_cap_rub > 0),
    ad_group_id                BIGINT,
    campaign_id                BIGINT,
    autonomy_level             TEXT NOT NULL CHECK (
        autonomy_level IN ('AUTO', 'NOTIFY', 'ASK', 'FORBIDDEN')
    ),
    risk_score                 DOUBLE PRECISION NOT NULL,
    state                      TEXT NOT NULL DEFAULT 'running' CHECK (
        state IN (
            'running', 'confirmed', 'rejected',
            'inconclusive', 'rolled_back', 'waiting_budget'
        )
    ),
    metrics_before             JSONB NOT NULL,
    metrics_before_captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metrics_after              JSONB,
    metrics_after_captured_at  TIMESTAMPTZ,
    baseline_at_promote        JSONB,
    promoted_at                TIMESTAMPTZ,
    outcome                    TEXT CHECK (outcome IN ('positive', 'negative', 'neutral')),
    lesson                     TEXT,
    clicks_at_record           INTEGER DEFAULT 0,
    check_after_clicks         INTEGER DEFAULT 0,
    maximum_running_days       INTEGER DEFAULT 14,
    CONSTRAINT attribution_single CHECK (
        ad_group_id IS NOT NULL
        OR campaign_id IS NOT NULL
        OR hypothesis_type = 'account_level'
    )
);

CREATE INDEX IF NOT EXISTS ix_hypotheses_state_created
    ON hypotheses (state, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_hypotheses_campaign
    ON hypotheses (campaign_id)
    WHERE state = 'confirmed';
CREATE INDEX IF NOT EXISTS ix_hypotheses_agent_outcome
    ON hypotheses (agent, outcome);
CREATE INDEX IF NOT EXISTS ix_hypotheses_regression_watch
    ON hypotheses (promoted_at)
    WHERE state = 'confirmed' AND promoted_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS audit_log (
    id                   BIGSERIAL PRIMARY KEY,
    ts                   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hypothesis_id        TEXT REFERENCES hypotheses(id),
    trust_level          TEXT NOT NULL,
    tool_name            TEXT NOT NULL,
    tool_input           JSONB NOT NULL,
    tool_output          JSONB,
    is_mutation          BOOLEAN NOT NULL,
    is_error             BOOLEAN NOT NULL DEFAULT FALSE,
    error_detail         TEXT,
    user_confirmed       BOOLEAN NOT NULL DEFAULT FALSE,
    kill_switch_triggered TEXT
);

CREATE INDEX IF NOT EXISTS ix_audit_log_ts ON audit_log (ts DESC);
CREATE INDEX IF NOT EXISTS ix_audit_log_mutation_shadow
    ON audit_log (ts DESC)
    WHERE is_mutation = TRUE AND trust_level = 'shadow';

CREATE TABLE IF NOT EXISTS creative_patterns (
    id                     BIGSERIAL PRIMARY KEY,
    discovered_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source                 TEXT NOT NULL,
    competitor             TEXT,
    pattern_type           TEXT NOT NULL,
    pattern_text           TEXT NOT NULL,
    metadata               JSONB NOT NULL DEFAULT '{}'::jsonb,
    used_in_hypothesis_ids TEXT[]
);

CREATE INDEX IF NOT EXISTS ix_creative_patterns_type
    ON creative_patterns (pattern_type, discovered_at DESC);

CREATE TABLE IF NOT EXISTS ask_queue (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hypothesis_id       TEXT NOT NULL REFERENCES hypotheses(id),
    question            TEXT NOT NULL,
    options             JSONB NOT NULL DEFAULT '["approve","reject","defer_24h"]'::jsonb,
    resolved_at         TIMESTAMPTZ,
    answer              TEXT,
    telegram_message_id BIGINT
);

CREATE INDEX IF NOT EXISTS ix_ask_queue_unresolved
    ON ask_queue (created_at)
    WHERE resolved_at IS NULL;

CREATE TABLE IF NOT EXISTS watchdog_heartbeat (
    service       TEXT PRIMARY KEY,
    last_beat_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMIT;
