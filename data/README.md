# data/

Local data zones.

- raw/: immutable ingested source data
- curated/: cleaned and standardized tables in a canonical schema
- snapshots/: versioned bundles used for reproducible backtests and production runs

Do not edit files in raw/ manually. Prefer to re-ingest.
