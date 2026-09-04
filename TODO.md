# TODO

- **Early-bail check in `NnPlaysSnake::run_one_game` never fires.**  The loop tests the per-move `penalty` (0–3) against 50 instead of the running total, so the “too many unsafe choices” bail is dead code.  Compare the running total instead (and fix its spelling: `pentalies` → `penalties`).  Noted 2026-09-04 while reviewing v0.2.0.
