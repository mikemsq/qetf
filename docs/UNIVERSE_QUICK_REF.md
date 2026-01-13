# ETF Universe Tiers - Quick Reference

**One-page guide to selecting the right universe tier**

---

## At a Glance

| Tier | ETFs | Best For | Liquidity | Cost |
|------|------|----------|-----------|------|
| **T1** | 20 | Prototyping | ★★★★★ | ★★★★★ |
| **T2** | 50 | Production | ★★★★☆ | ★★★★☆ |
| **T3** | 100 | Industry/Thematic | ★★★☆☆ | ★★★☆☆ |
| **T4** | 200 | Global Macro | ★★☆☆☆ | ★★☆☆☆ |
| **T5** | 300+ | Research | ★☆☆☆☆ | ★☆☆☆☆ |

---

## Decision Tree

```
Need hedging tools (inverse/leveraged)?
├─ YES → Use Tier 4+
└─ NO → Continue

Need thematic ETFs (AI, clean energy)?
├─ YES → Use Tier 3+
└─ NO → Continue

Need all 11 sectors?
├─ YES → Use Tier 2+
└─ NO → Use Tier 1

Strategy is production-ready?
├─ YES → Validate on Tier 1-2 only
└─ NO → Develop on Tier 1, test on Tier 2
```

---

## Common Use Cases

### Momentum Strategy
**Start:** Tier 1 (prove concept)
**Scale to:** Tier 2 (production)
**Max:** Tier 3 (if using industries)

### Sector Rotation
**Minimum:** Tier 2 (need all 11 sectors)
**Optimal:** Tier 2-3

### Factor Strategy (Value/Momentum/Quality)
**Minimum:** Tier 2 (factor ETFs available)
**Optimal:** Tier 2-3

### Global Macro
**Minimum:** Tier 4 (need country exposure + hedging)
**Optimal:** Tier 4-5

### Thematic Rotation
**Minimum:** Tier 3 (thematic ETFs)
**Optimal:** Tier 3-5

### Long/Short Equity
**Minimum:** Tier 4 (need inverse ETFs)
**Optimal:** Tier 4-5

### Options-Based Income
**Minimum:** Tier 5 (QYLD, JEPI, etc.)

---

## Cost Models by Tier

| Tier | Spread (bps) | Expense Ratio | Total Cost/Trade |
|------|--------------|---------------|------------------|
| T1 | 3-5 | 0.05-0.50% | ~10 bps |
| T2 | 5-10 | 0.05-0.75% | ~15 bps |
| T3 | 10-15 | 0.20-0.95% | ~25 bps |
| T4 | 15-30 | 0.30-1.50% | ~40 bps |
| T5 | 30-50 | 0.50-2.00% | ~75 bps |

*Note: Total cost = spread + commission proxy + expense ratio impact*

---

## Universe Composition

### Tier 1 (20 ETFs)
```
4  US Equity (SPY, QQQ, IWM, DIA)
5  US Sectors (XLF, XLE, XLK, XLV, XLI)
3  International (EFA, EEM, VWO)
3  Fixed Income (AGG, TLT, LQD)
3  Real Assets (GLD, SLV, VNQ)
2  Alternatives (VIXY, USDU)
```

### Tier 2 (50 ETFs) = T1 + 30
```
+ 4  More Sectors (XLY, XLP, XLB, XLU) → All 11 complete
+ 5  Regional (FEZ, EWJ, FXI, MCHI, INDA)
+ 5  Style Factors (VTV, VUG, VYM, IVV, VTI)
+ 7  Fixed Income (HYG, JNK, MUB, TIP, SHY, IEF, BND)
+ 4  Commodities (DBC, USO, UNG, PDBC)
+ 5  Factors (SPLV, MTUM, QUAL, SIZE, USMV)
```

### Tier 3 (100 ETFs) = T2 + 50
```
+ 10 Industries (banks, semis, biotech, homebuilders)
+ 10 Countries (Brazil, Korea, Taiwan, Germany, etc.)
+ 10 Thematics (ARKK, ICLN, TAN, BOTZ, HACK)
+ 10 EM/Int'l FI (EMB, PCY, BNDX, PFF)
+ 5  More factors (VLUE, IEMG, VEA, VO, VB)
+ 5  Precious Metals (IAU, PALL, PPLT, CPER)
```

### Tier 4 (200 ETFs) = T3 + 100
```
+ 15 Frontier Markets
+ 10 Currency Hedged
+ 10 Leveraged (2x)
+ 10 Inverse (-1x)
+ 10 ESG
+ 10 Dividend/Income
+ 15 More Industries
+ 10 Credit Strategies
+ 10 Commodities/Infrastructure
```

### Tier 5 (300+ ETFs) = T4 + 100+
```
+ 20 More Thematics
+ 15 Options-Based
+ 15 Multi-Factor
+ 10 Volatility Strategies
+ 10 Real Estate Detail
+ 15 Fixed Income Detail
+ 10 Small Cap International
+ 10 Currencies
```

---

## File Locations

```
configs/universes/tier1_initial_20.yaml
configs/universes/tier2_core_50.yaml
configs/universes/tier3_expanded_100.yaml
configs/universes/tier4_broad_200.yaml
configs/universes/tier5_comprehensive_300plus.yaml
```

---

## Commands

```bash
# List all available universes
ls configs/universes/tier*.yaml

# Run backtest on specific tier
python scripts/run_backtest.py \
    --universe tier2_core_50 \
    --strategy residual_momentum

# Compare same strategy across tiers
python scripts/compare_strategies.py \
    --strategy momentum \
    --universes tier1_initial_20,tier2_core_50,tier3_expanded_100
```

---

## Recommendations

✅ **Do:**
- Start small (Tier 1)
- Validate scaling (Tier 1 → Tier 2)
- Test robustness (same strategy, multiple tiers)
- Monitor costs (track slippage in higher tiers)
- Use Tier 1-2 for live trading

❌ **Don't:**
- Skip Tier 1 validation
- Trade Tier 4-5 strategies without Tier 1-2 validation
- Ignore transaction costs on higher tiers
- Assume results scale linearly
- Use leveraged ETFs without understanding decay

---

**Full Documentation:** [ETF_UNIVERSE_TIERS.md](./ETF_UNIVERSE_TIERS.md)
