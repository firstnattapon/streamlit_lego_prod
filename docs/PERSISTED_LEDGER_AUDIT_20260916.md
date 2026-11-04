# Persisted ledger integrity and funding baseline

The live integrity panel now checks the raw database rows before any display
recomputation. Historical frozen-v2 and new `execution_terminal_funding_v3`
rows remain displayed as stored; the reader does not repair database history.

Previously frozen-v2 recurrence checks compared the stored ledger to its own
unchanged copy and excluded v2 from execution-provenance validation. The new
independent validation checks positive fill facts, prior action price and A,
delta arithmetic, R_basis, funding offset and frozen A/E. Missing execution
history in a clipped window is reported as incomplete rather than certified.
Chain genesis is determined by version 1 as well as the legacy DNA step 0;
a time-aligned chain may begin at DNA step 133.

The companion backend treats a new flat account's first confirmed BUY as funding:
delta/A/E start at zero and the fill VWAP becomes the next action price. It
stores explicit funding provenance; later fills use their normal recurrence.
An old first BUY that booked quote-to-fill slippage is flagged but never silently
rewritten. Existing corrected data needs the reviewed offline reconciliation
plan from the backend repository. Install this reader before the v3 writer.

The result concerns arithmetic and provenance in loaded rows only. It does not
certify broker execution completeness, profitability or production readiness.
R_market is the market benchmark, R_basis is the frozen excess benchmark;
model A/E are not fee-inclusive realized broker P&L.

## Addendum 2026-11-03: the table starts at n = 0

The 17-column table shows the first row of a chain (n = 0) as R₀ = ΔA₀ = A₀ = E₀ = 0. A first fill's
quote-to-fill slippage is not booked as a return (P_fill := P₀, so FIX_C × (P₀ / P₀ − 1) = 0) and the fill
price seeds the next P_acted, as `initial_funding_zero_v1` does for a flat account.

Why it was needed: a chain that starts with holdings ≠ 0 (a LEAPS chain, e.g. row 1 of
`fixtures/paper_session_leaps.json`: ΔA₀ = 0.9162 = 3000 × (65.51 / 65.49 − 1), no `initial_funding`) does not
get the funding baseline, and the database keeps ΔA₀ = FIX_C × (fill / quote − 1) inside A and E of every later
row. (This is inferred from the stored rows and from the `holdings == 0` test in `_frozen_v2_errors`; the backend
code is not in this repository.) Such a chain is internally consistent under the normal-fill contract, so the
integrity check passes; only the starting point of the displayed ledger was wrong.

What changed and what did not:

- Display only and opt-in: `genesis_baseline=True` on `recompute_gated_ledger`, `ledger_explanations` and
  `table_tooltips` (the live page passes it). `integrity_report`, `count_ledger_corrections` and the default
  recompute still read the persisted rows unchanged, so the earlier statement that stored history is never
  silently rewritten still holds for everything that audits it.
- For frozen v2/v3 rows `normalize_genesis_baseline` subtracts the persisted ΔA₀ from A, E,
  `E_mark_at_observation` (rows observed after the first fill was finalized) and `previous_actual_cumulative`
  (later fills). Later fills keep their own ΔA. It rebases only when ΔA₀ equals
  FIX_C × (execution_price / previous_action_price − 1) with previous_action_price = P₀ and the first fill's
  `cashflow_finalized_at` is known; anything else is left for the integrity check.
- Not silent: the page shows a banner with the persisted ΔA₀, and each affected tooltip shows the persisted
  value and the amount removed. Nothing is written to the database.
- Genesis is decided by `version == 1` or `DNA step == 0` everywhere (it was `DNA step == 0` only in the
  recompute and the reference line), so a time-aligned chain that begins at step 133 or 1040 is no longer
  shown as a clipped window.

The root cause is upstream: the backend should treat the first fill of such chains as funding and the stored
rows need the reviewed offline reconciliation. Once ΔA₀ is 0 in the database the rebase is a no-op.
