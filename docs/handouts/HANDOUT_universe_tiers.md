# ETF Universe Tiers - Implementation Handout

**Date:** 2026-01-13
**Topic:** Multi-Tier ETF Universe Design
**Agent:** Quant Research Agent

---

## What Was Built

A **5-tier ETF universe system** expanding from 20 to 300+ ETFs, designed for progressive strategy development with practical liquidity and cost constraints.

---

## Files Created

### Universe Configurations (YAML)
```
configs/universes/
├── tier1_initial_20.yaml          (20 ETFs - ultra-liquid baseline)
├── tier2_core_50.yaml             (50 ETFs - production ready)
├── tier3_expanded_100.yaml        (100 ETFs - industry granularity)
├── tier4_broad_200.yaml           (200 ETFs - global macro)
└── tier5_comprehensive_300plus.yaml (300+ ETFs - maximum coverage)
```

### Documentation
```
docs/
├── ETF_UNIVERSE_TIERS.md          (Complete guide with matrix)
└── UNIVERSE_QUICK_REF.md          (One-page cheat sheet)

configs/universes/
└── README.md                       (Updated with tier overview)
```

---

## Tier Overview

| Tier | Size | Liquidity | Expense Ratio | Primary Use |
|------|------|-----------|---------------|-------------|
| **1** | 20 | ★★★★★ | <0.50% | Initial development |
| **2** | 50 | ★★★★☆ | <0.75% | Production strategies |
| **3** | 100 | ★★★☆☆ | <0.95% | Industry rotation |
| **4** | 200 | ★★☆☆☆ | <1.50% | Global macro |
| **5** | 300+ | ★☆☆☆☆ | <2.00% | Research/alternatives |

---

## Key Features

### 1. Progressive Expansion
Each tier includes all ETFs from previous tiers, allowing seamless scaling

### 2. Liquidity-First Approach
Explicit minimum daily volume thresholds:
- Tier 1: >$50M/day
- Tier 2: >$20M/day
- Tier 3: >$10M/day
- Tier 4: >$5M/day
- Tier 5: >$1M/day

### 3. Cost-Conscious Design
Maximum expense ratio limits prevent expensive ETFs:
- Tier 1: 0.50% (ultra-low)
- Tier 2: 0.75% (low)
- Tier 3: 0.95% (moderate)
- Tier 4: 1.50% (mod-high, allows leveraged)
- Tier 5: 2.00% (varies, allows specialized)

### 4. Quality Filtering
- Minimum AUM requirements ($50M-$1B) prevent closure risk
- Focus on major providers (iShares, Vanguard, SPDR, Invesco, Schwab)
- Avoids obscure and expensive instruments

---

## Composition Highlights

### Tier 1 (20 ETFs)
- 4 US Equity (SPY, QQQ, IWM, DIA)
- 5 US Sectors (XLF, XLE, XLK, XLV, XLI)
- 3 International (EFA, EEM, VWO)
- 3 Fixed Income (AGG, TLT, LQD)
- 3 Real Assets (GLD, SLV, VNQ)
- 2 Alternatives (VIXY, USDU)

### Tier 2 Additions (+30)
- Complete 11-sector coverage (XLY, XLP, XLB, XLU)
- Regional equity (Europe, Japan, China, India)
- Style factors (value, growth, dividend)
- Extended fixed income (HY, munis, TIPS)
- Factor strategies (momentum, quality, low vol)

### Tier 3 Additions (+50)
- Industry ETFs (semiconductors, banks, biotech, homebuilders)
- Country exposures (Brazil, Korea, Taiwan, Germany, etc.)
- Thematics (ARKK, clean energy, AI, fintech, cybersecurity)
- EM fixed income (EMB, PCY)
- Precious metals (palladium, platinum, copper)

### Tier 4 Additions (+100)
- Frontier markets (15 countries)
- Currency hedged variants (10)
- Leveraged 2x (10)
- Inverse -1x for hedging (10)
- ESG variants (10)
- Dividend/income focus (10)
- Credit strategies (fallen angels, floating rate)

### Tier 5 Additions (+100-150)
- Options-based income (QYLD, JEPI, RYLD)
- Volatility strategies (managed futures, tail risk)
- Multi-factor combinations
- Comprehensive thematics (cannabis, space, gaming)
- Currency macro tools

---

## Usage Guide

### Development Workflow
```
1. Develop on Tier 1 (20 ETFs)
   └─> Fast iteration, proof-of-concept

2. Validate on Tier 2 (50 ETFs)
   └─> Check if strategy scales

3. Test on Tier 3 (100 ETFs)
   └─> Industry/thematic robustness

4. (Optional) Stress test on Tier 4-5
   └─> For comprehensive strategies only
```

### Strategy-Specific Recommendations

**Momentum/Trend Following:**
- Develop: Tier 1
- Production: Tier 2

**Sector Rotation:**
- Minimum: Tier 2 (need all 11 sectors)
- Optimal: Tier 2-3

**Factor Strategies:**
- Minimum: Tier 2 (factor ETFs)
- Optimal: Tier 2-3

**Industry Rotation:**
- Minimum: Tier 3
- Optimal: Tier 3

**Thematic Momentum:**
- Minimum: Tier 3
- Optimal: Tier 3-5

**Global Macro:**
- Minimum: Tier 4
- Optimal: Tier 4-5

**Long/Short Equity:**
- Minimum: Tier 4 (need inverse)
- Optimal: Tier 4-5

**Alternatives/Volatility:**
- Minimum: Tier 5

---

## Transaction Cost Estimates

| Tier | Spread (bps) | Total Cost/Trade |
|------|--------------|------------------|
| T1 | 3-5 | ~10 bps |
| T2 | 5-10 | ~15 bps |
| T3 | 10-15 | ~25 bps |
| T4 | 15-30 | ~40 bps |
| T5 | 30-50 | ~75 bps |

---

## Command Examples

### Run backtest on specific tier
```bash
python scripts/run_backtest.py \
    --strategy residual_momentum \
    --universe tier2_core_50
```

### Compare strategy across tiers
```bash
python scripts/compare_strategies.py \
    --strategy momentum \
    --universes tier1_initial_20,tier2_core_50,tier3_expanded_100
```

### Ingest data for tier
```bash
python scripts/ingest_etf_data.py --universe tier2_core_50

python scripts/create_snapshot.py \
    --universe tier2_core_50 \
    --start 2020-01-01 \
    --end 2026-01-13
```

---

## Risk Warnings by Tier

### Tiers 1-2: Low Risk
- Very liquid, tight spreads
- Well-established ETFs
- Low expense ratios
- **Recommended for live trading**

### Tier 3: Medium Risk
- Generally liquid but some wider spreads
- Some thematic ETFs have tracking issues
- Monitor transaction costs
- Validate on Tier 1-2 before live

### Tiers 4-5: Higher Risk
- Lower liquidity on some instruments
- Leveraged ETFs have decay risk
- Inverse ETFs not for long-term holds
- Higher expense ratios
- **Research only - validate thoroughly before live trading**

---

## Best Practices

✅ **Do:**
- Start development on Tier 1
- Validate scaling on Tier 2
- Test robustness across multiple tiers
- Monitor transaction costs carefully
- Use Tier 1-2 for live trading

❌ **Don't:**
- Skip Tier 1 validation
- Trade Tier 4-5 without Tier 1-2 validation
- Ignore transaction costs on higher tiers
- Assume results scale linearly
- Hold leveraged/inverse ETFs long-term

---

## Documentation References

- **Full Guide:** [`docs/ETF_UNIVERSE_TIERS.md`](../ETF_UNIVERSE_TIERS.md)
- **Quick Reference:** [`docs/UNIVERSE_QUICK_REF.md`](../UNIVERSE_QUICK_REF.md)
- **Config Files:** [`configs/universes/`](../../configs/universes/)
- **Strategy Plan:** [`STRATEGY_RESEARCH_PLAN.md`](../../STRATEGY_RESEARCH_PLAN.md)

---

## Implementation Notes

### What Works Now
- All 5 configuration files are ready to use
- Compatible with existing backtest infrastructure
- Can reference by name in strategy configs

### Data Availability
- Tier 1: Full historical data available
- Tier 2-3: Most ETFs have good history
- Tier 4-5: Some newer ETFs may have gaps
- System handles missing data gracefully

### Next Steps for Implementation
1. Test existing strategies on Tier 2
2. Compare results vs Tier 1
3. Implement industry rotation on Tier 3
4. Add transaction cost sensitivity analysis
5. Document performance by tier

---

## Git Commit

```
Commit: d07a1d8
Message: Add 5-tier ETF universe system (20 to 300+ ETFs)
Files: 8 changed, 1564 insertions(+)
```

---

**Questions or Issues?**
See full documentation in [`docs/ETF_UNIVERSE_TIERS.md`](../ETF_UNIVERSE_TIERS.md)
