# Production State Directory

This directory contains runtime state for the production system. Files here are **not tracked in git**.

## Files (Generated at Runtime)

| File | Purpose | Updated |
|------|---------|---------|
| `current_regime.json` | Current market regime state with hysteresis | Daily |
| `regime_history.parquet` | Historical regime changes log | Daily |
| `last_rebalance.json` | Most recent rebalance metadata | On rebalance |

## current_regime.json Format

```json
{
  "as_of": "2026-01-24",
  "regime": "uptrend_low_vol",
  "trend_state": "uptrend",
  "vol_state": "low_vol",
  "indicators": {
    "spy_price": 585.42,
    "spy_200ma": 542.18,
    "vix": 14.5
  },
  "previous_regime": "uptrend_low_vol",
  "regime_changed": false
}
```

## Backup Recommendations

While not tracked in git, this state should be backed up:
- Include in daily backup scripts
- Persist to cloud storage for disaster recovery
- Log regime changes to external monitoring system

## Recovery

If state files are lost:
1. The daily monitor will regenerate `current_regime.json` on next run
2. `regime_history.parquet` can be rebuilt from logs or recreated going forward
3. System continues functioning with fresh state (no historical context)

## Related Configuration

- Regime thresholds: `configs/regimes/thresholds.yaml`
- Default mapping: `configs/regimes/default_mapping.yaml`
- Architecture: `handoffs/architecture/ADR-001-regime-based-strategy-selection.md`
