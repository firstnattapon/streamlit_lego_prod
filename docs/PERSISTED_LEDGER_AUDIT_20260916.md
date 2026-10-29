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
