# configs/

Configuration files for reproducible research and production runs.

Typical config types:
- Universe definitions and eligibility filters
- Strategy definitions (alpha model + risk model + portfolio construction)
- Rebalance schedules and calendars
- Transaction cost assumptions

Keep configs immutable for a given run, and record a hash of each config in run artifacts.
