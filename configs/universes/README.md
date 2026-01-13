# configs/universes/

**Universe configurations for ETF selection**

---

## Overview

This directory contains ETF universe definitions organized in **five tiers** from 20 to 300+ ETFs. Each tier builds on the previous one, balancing opportunity set size with liquidity, cost, and data quality.

---

## Available Universes

### Tier 1: Initial 20 ETFs
**File:** [tier1_initial_20.yaml](tier1_initial_20.yaml)
- **Size:** 20 ETFs
- **Focus:** Extremely liquid, low-cost baseline
- **Use:** Initial development, proof-of-concept
- **Liquidity:** ★★★★★ (>$50M daily volume)
- **Cost:** Ultra-low (<0.50% expense ratio)

### Tier 2: Core 50 ETFs
**File:** [tier2_core_50.yaml](tier2_core_50.yaml)
- **Size:** 50 ETFs
- **Focus:** Complete sector coverage + factors
- **Use:** Production strategies, sector rotation
- **Liquidity:** ★★★★☆ (>$20M daily volume)
- **Cost:** Low (<0.75% expense ratio)

### Tier 3: Expanded 100 ETFs
**File:** [tier3_expanded_100.yaml](tier3_expanded_100.yaml)
- **Size:** 100 ETFs
- **Focus:** Industry granularity + thematics
- **Use:** Industry rotation, thematic strategies
- **Liquidity:** ★★★☆☆ (>$10M daily volume)
- **Cost:** Moderate (<0.95% expense ratio)

### Tier 4: Broad 200 ETFs
**File:** [tier4_broad_200.yaml](tier4_broad_200.yaml)
- **Size:** 200 ETFs
- **Focus:** Global macro + hedging tools
- **Use:** Long/short, currency management, ESG
- **Liquidity:** ★★☆☆☆ (>$5M daily volume)
- **Cost:** Moderate-High (<1.50% expense ratio)

### Tier 5: Comprehensive 300+ ETFs
**File:** [tier5_comprehensive_300plus.yaml](tier5_comprehensive_300plus.yaml)
- **Size:** 300+ ETFs
- **Focus:** Maximum opportunity set
- **Use:** Alternatives, volatility, options-income
- **Liquidity:** ★☆☆☆☆ (>$1M daily volume)
- **Cost:** Varies widely (<2.00% expense ratio)

---

## Quick Selection Guide

**New to the project?** → Start with **Tier 1**

**Building production strategy?** → Use **Tier 2**

**Need all sectors or factor ETFs?** → Use **Tier 2+**

**Need industry detail or thematics?** → Use **Tier 3+**

**Need hedging or international variants?** → Use **Tier 4+**

**Research maximum opportunity?** → Use **Tier 5**

---

## Universe Configuration Format

Each universe config specifies:
- **Base list or source:** Static list, provider universe, or filter-driven dynamic set
- **Eligibility filters:** Liquidity, AUM, minimum history, domicile, etc.
- **Point-in-time rules:** Prevent lookahead bias
- **Tier metadata:** Size, liquidity profile, cost profile
- **Usage notes:** Recommendations and warnings

### Example Structure

```yaml
name: tier2_core_50_etfs
description: Core 50 ETF universe with complete sector coverage
tier: 2
size: 50
liquidity_profile: very_high
expense_ratio_profile: low_cost

source:
  type: static_list
  tickers:
    - SPY
    - QQQ
    # ... more tickers

eligibility:
  min_history_days: 252
  min_avg_dollar_volume: 20_000_000
  min_aum: 500_000_000
  max_expense_ratio: 0.75

notes: |
  Use this for production strategies...
```

---

## Usage Examples

### Running a backtest with a specific universe

```bash
# Run backtest on Tier 2
python scripts/run_backtest.py \
    --strategy residual_momentum \
    --universe tier2_core_50

# Compare strategy across multiple universes
python scripts/compare_strategies.py \
    --strategy momentum \
    --universes tier1_initial_20,tier2_core_50,tier3_expanded_100
```

### Ingesting data for a universe

```bash
# Ingest data for specific tier
python scripts/ingest_etf_data.py --universe tier2_core_50

# Create snapshot
python scripts/create_snapshot.py \
    --universe tier2_core_50 \
    --start 2020-01-01 \
    --end 2026-01-13
```

---

## Documentation

- **Full Guide:** [../docs/ETF_UNIVERSE_TIERS.md](../docs/ETF_UNIVERSE_TIERS.md)
- **Quick Reference:** [../docs/UNIVERSE_QUICK_REF.md](../docs/UNIVERSE_QUICK_REF.md)
- **Strategy Research Plan:** [../STRATEGY_RESEARCH_PLAN.md](../STRATEGY_RESEARCH_PLAN.md)

---

## Design Principles

1. **Progressive Expansion:** Each tier includes all ETFs from lower tiers
2. **Liquidity First:** Lower tiers prioritize extremely liquid ETFs
3. **Cost Conscious:** Avoid expensive/obscure ETFs until higher tiers
4. **Practical Testing:** Test on smaller universes before scaling
5. **Risk Management:** Higher tiers require more sophisticated controls

---

## Notes

- All ETFs avoid highly obscure or expensive instruments
- Focus on major providers: iShares, Vanguard, SPDR, Invesco, Schwab
- Each tier has explicit liquidity and AUM thresholds
- Higher tiers may have data gaps for newer ETFs (system handles gracefully)
- **Recommendation:** Always validate strategies on Tier 1-2 before live trading
