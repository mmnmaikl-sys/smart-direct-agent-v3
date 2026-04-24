-- Rollback for 001_initial.sql. CASCADE because ask_queue / audit_log reference
-- hypotheses via FK. Dropping in reverse dependency order is explicit here even
-- though CASCADE would handle it — keeps intent clear and the file safe to run
-- against a partially-applied schema.

BEGIN;

DROP TABLE IF EXISTS ask_queue CASCADE;
DROP TABLE IF EXISTS audit_log CASCADE;
DROP TABLE IF EXISTS watchdog_heartbeat CASCADE;
DROP TABLE IF EXISTS creative_patterns CASCADE;
DROP TABLE IF EXISTS hypotheses CASCADE;
DROP TABLE IF EXISTS sda_state CASCADE;

COMMIT;
