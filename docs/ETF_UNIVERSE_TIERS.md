# ETF Universe Tiers - Complete Guide

**Last Updated:** 2026-01-13
**Purpose:** Multi-tier ETF universe design for progressive strategy development

---

## Overview

The QuantETF platform supports **five tiered universes** ranging from 20 to 300+ ETFs. Each tier is designed for specific use cases, balancing breadth of opportunity with practical considerations like liquidity, costs, and data availability.

### Design Philosophy

1. **Progressive Expansion**: Each tier builds on the previous tier
2. **Liquidity First**: Lower tiers prioritize extremely liquid ETFs
3. **Cost Conscious**: Avoid obscure/expensive ETFs until higher tiers
4. **Practical Testing**: Test strategies on smaller universes before scaling
5. **Risk Management**: Higher tiers require more sophisticated risk controls

---

## Tier Comparison Matrix

| Tier | Size | Liquidity | Expense Ratio | Primary Use Case |
|------|------|-----------|---------------|------------------|
| **Tier 1** | 20 ETFs | Extremely High | Ultra Low (≤0.50%) | Initial development, proof-of-concept |
| **Tier 2** | 50 ETFs | Very High | Low (≤0.75%) | Sector rotation, factor strategies |
| **Tier 3** | 100 ETFs | High | Moderate (≤0.95%) | Industry rotation, thematics |
| **Tier 4** | 200 ETFs | Med-High | Mod-High (≤1.50%) | Global macro, long/short, hedging |
| **Tier 5** | 300+ ETFs | Medium | Varies (≤2.00%) | Comprehensive research, alternatives |

---

## Tier 1: Initial 20 ETFs (Baseline)

**File:** [`configs/universes/tier1_initial_20.yaml`](../configs/universes/tier1_initial_20.yaml)

### Purpose
Foundation universe for initial strategy development and testing. Highly liquid, low-cost ETFs from top providers.

### Composition
- **US Equity** (4): SPY, QQQ, IWM, DIA
- **US Sectors** (5): XLF, XLE, XLK, XLV, XLI
- **International** (3): EFA, EEM, VWO
- **Fixed Income** (3): AGG, TLT, LQD
- **Real Assets** (3): GLD, SLV, VNQ
- **Alternatives** (2): VIXY, USDU

### Eligibility Criteria
- Min AUM: $1B
- Min Daily Volume: $50M
- Max Expense Ratio: 0.50%
- Min History: 1 year

### Use Cases
✅ Initial strategy prototyping
✅ Teaching and examples
✅ Quick backtests (fast execution)
✅ Proof-of-concept work
✅ Baseline for comparison

❌ Not suitable for: Granular sector strategies, country-specific plays, niche thematics

---

## Tier 2: Core 50 ETFs

**File:** [`configs/universes/tier2_core_50.yaml`](../configs/universes/tier2_core_50.yaml)

### Purpose
Expanded coverage for sophisticated cross-sectional and factor strategies while maintaining high liquidity.

### New Additions (+30 ETFs)
- **Additional Sectors** (4): XLY, XLP, XLB, XLU (complete SPDR sector coverage)
- **Regional Equity** (5): FEZ, EWJ, FXI, MCHI, INDA
- **Style Factors** (5): VTV, VUG, VYM, IVV, VTI
- **Extended Fixed Income** (7): HYG, JNK, MUB, TIP, SHY, IEF, BND
- **Commodities** (4): DBC, USO, UNG, PDBC
- **Factor Strategies** (5): SPLV, MTUM, QUAL, SIZE, USMV

### Eligibility Criteria
- Min AUM: $500M
- Min Daily Volume: $20M
- Max Expense Ratio: 0.75%
- Min History: 1 year

### Use Cases
✅ Complete sector rotation (all 11 sectors)
✅ Factor-based strategies (momentum, quality, low vol)
✅ Regional allocation strategies
✅ Full yield curve strategies
✅ High yield and municipal bond strategies

---

## Tier 3: Expanded 100 ETFs

**File:** [`configs/universes/tier3_expanded_100.yaml`](../configs/universes/tier3_expanded_100.yaml)

### Purpose
Industry-level granularity and thematic exposure for more sophisticated strategies.

### New Additions (+50 ETFs)
- **Industry ETFs** (10): XHB, KRE, KBE, XRT, SMH, IBB, IYR, ITB, SOXX, IHI
- **Country ETFs** (10): EWZ, EWT, EWY, EWG, EWU, EWC, EWA, THD, EPU, VGK
- **Thematic/Innovation** (10): ICLN, TAN, ARKK, ARKG, ARKW, BOTZ, SKYY, FINX, HACK, LIT
- **EM/Int'l Fixed Income** (10): EMB, PCY, VCIT, VCSH, BNDX, FLOT, PFF, SCHP, VMBS, SJNK
- **Additional Factors** (5): VLUE, IEMG, VEA, VO, VB
- **Precious Metals** (5): REET, IAU, PALL, PPLT, CPER

### Eligibility Criteria
- Min AUM: $250M
- Min Daily Volume: $10M
- Max Expense Ratio: 0.95%
- Min History: 6 months

### Use Cases
✅ Industry rotation strategies (banks, semiconductors, biotech)
✅ Country-specific macro strategies
✅ Thematic momentum (clean energy, AI, fintech)
✅ Emerging market debt strategies
✅ Preferred stock and floating rate strategies

---

## Tier 4: Broad 200 ETFs

**File:** [`configs/universes/tier4_broad_200.yaml`](../configs/universes/tier4_broad_200.yaml)

### Purpose
Comprehensive global coverage including hedging tools, currency management, and ESG variants.

### New Additions (+100 ETFs)
- **Frontier Markets** (15): EWH, EWI, EWL, EWP, EWQ, EWS, EWW, EZA, ECH, EIDO, EGPT, FM, RSX, TUR, VNM
- **Currency Hedged** (10): HEFA, HEWJ, DBEF, DBJP, HEWG, HEWU, HEEM, DBEM, DBEU, HEZU
- **Leveraged 2x** (10): SSO, QLD, UWM, DDM, MVV, UCC, UYG, URE, ROM, RXL
- **Inverse -1x** (10): SH, PSQ, RWM, DOG, MYY, TBF, TBT, SJB, EUM, EFZ
- **ESG/Sustainable** (10): ESGU, ESGV, VSGX, SUSL, USSG, SUSA, DSI, KRMA, ESGD, PBW
- **Dividend/Income** (10): VIG, SCHD, DVY, SDY, HDV, NOBL, DHS, FVD, DGRO, IDV
- **Industry Detail** (15): IYT, IYE, IYF, IYZ, IYC, IYK, IYM, IYW, IGV, IGN, IGM, IHF, IXC, IXN, IXJ
- **Credit Strategies** (10): ANGL, FALN, BKLN, SRLN, NEAR, JPST, MINT, SHYG, IGIB, SPIB
- **Commodities/Infra** (10): GSG, DJP, GCC, COMT, BCI, WOOD, GUNR, MOO, PAVE, FRAK

### Eligibility Criteria
- Min AUM: $100M
- Min Daily Volume: $5M
- Max Expense Ratio: 1.50%
- Min History: 3 months

### Use Cases
✅ Global macro strategies
✅ Long/short equity (leveraged/inverse)
✅ Currency risk management (hedged variants)
✅ ESG-integrated strategies
✅ Tactical hedging strategies
✅ Income-focused portfolios
✅ Frontier market diversification

⚠️ **Warnings:**
- Leveraged ETFs have daily reset risk and decay
- Inverse ETFs are for hedging, not long-term holds
- Currency hedged ETFs have basis risk
- Lower liquidity = wider spreads

---

## Tier 5: Comprehensive 300+ ETFs

**File:** [`configs/universes/tier5_comprehensive_300plus.yaml`](../configs/universes/tier5_comprehensive_300plus.yaml)

### Purpose
Maximum opportunity set for comprehensive research and specialized strategies.

### New Additions (+100-150 ETFs)
- **Thematic/Innovation** (20): ARKQ, ARKF, BLOK, BATT, DRIV, DTEC, GIGE, HERO, IPAY, JETS, MOON, MSOS, NERD, POTX, ROBO, SNSR, SRVR, UFO, WCLD, etc.
- **Options-Based Income** (15): QYLD, XYLD, RYLD, JEPI, JEPQ, NUSI, DIVO, SPYI, SWAN, PUTW, etc.
- **Multi-Factor Combos** (15): LRGF, SMLF, INTF, JPUS, JPGE, QMOM, QVAL, IVAL, IMOM, VFMF, etc.
- **Volatility Strategies** (10): SVXY, VXX, VIXM, TAIL, BTAL, CTA, KMLM, FMF, DBMF, EQLS
- **Real Estate Detail** (10): XLRE, RWR, SCHH, USRT, BBRE, REM, MORT, INDS, HOMZ, etc.
- **Fixed Income Detail** (15): GOVT, VGIT, VGSH, VGLT, SCHO, SCHR, SPTS, SPTL, SPTI, CMBS, MBB, GNMA, CWB, ICVT, LDUR
- **International Small** (10): ACWI, ACWX, IXUS, VXUS, VSS, SCHA, FNDA, FNDF, DLS, DFE
- **Currencies** (10): UUP, UDN, FXE, FXY, FXB, FXA, FXC, CYB, CEW, DBV

### Eligibility Criteria
- Min AUM: $50M
- Min Daily Volume: $1M
- Max Expense Ratio: 2.00%
- Min History: 1 month

### Use Cases
✅ Maximum cross-sectional opportunity
✅ Thematic rotation strategies
✅ Options-based income generation
✅ Volatility arbitrage
✅ Managed futures replication
✅ Currency macro strategies
✅ Academic research requiring broad coverage

⚠️ **Cautions:**
- Lower average liquidity (wider spreads, higher slippage)
- Higher expense ratios (drag on performance)
- Newer ETFs may lack sufficient history
- Survivorship bias risk (more ETF closures)
- Requires sophisticated risk management
- Not recommended for live trading without validation on smaller universe

---

## Usage Recommendations

### Strategy Development Workflow

```
Phase 1: Develop on Tier 1 (20 ETFs)
└─> Quick iteration, proof of concept

Phase 2: Validate on Tier 2 (50 ETFs)
└─> Check if strategy scales to more opportunities

Phase 3: Test on Tier 3 (100 ETFs)
└─> Validate with industry/thematic granularity

Phase 4: (Optional) Stress test on Tier 4-5
└─> Check robustness to universe expansion
```

### Selecting the Right Tier

**Use Tier 1 when:**
- Initial strategy development
- Teaching/documentation
- Need fast backtest execution
- Testing new backtest engine features

**Use Tier 2 when:**
- Building production strategies
- Factor-based approaches
- Sector rotation strategies
- Need high confidence in liquidity

**Use Tier 3 when:**
- Industry-level granularity needed
- Thematic strategies (clean energy, AI)
- Country-specific strategies
- Emerging market focus

**Use Tier 4 when:**
- Global macro strategies
- Long/short equity
- Currency management important
- ESG constraints
- Need hedging instruments

**Use Tier 5 when:**
- Maximum opportunity set required
- Thematic momentum
- Options-based income
- Alternatives/volatility strategies
- Academic research

---

## Risk Management by Tier

| Risk Factor | Tier 1-2 | Tier 3 | Tier 4-5 |
|-------------|----------|--------|----------|
| **Liquidity Risk** | Very Low | Low | Medium |
| **Spread Costs** | <5 bps | 5-10 bps | 10-30 bps |
| **Expense Ratio Drag** | Minimal | Low | Moderate |
| **Data Quality** | Excellent | Good | Variable |
| **Survivorship Bias** | Minimal | Low | Higher |
| **Position Sizing** | Flexible | Monitor | Constrain |

### Transaction Cost Assumptions

Recommended cost models by tier:

```yaml
Tier 1: 5 bps per trade (extremely tight spreads)
Tier 2: 10 bps per trade
Tier 3: 15 bps per trade
Tier 4: 20-30 bps per trade (leveraged/inverse higher)
Tier 5: 30-50 bps per trade (niche ETFs)
```

---

## Data Availability

All tiers can be backtested using the existing data pipeline:

```bash
# Ingest data for a specific tier
python scripts/ingest_etf_data.py --universe tier2_core_50

# Create snapshot
python scripts/create_snapshot.py --universe tier2_core_50 \
    --start 2020-01-01 --end 2026-01-13

# Run backtest
python scripts/run_backtest.py --strategy residual_momentum \
    --universe tier2_core_50
```

**Note:** Higher tiers may have gaps in historical data for newer ETFs. The system will automatically handle missing data and adjust the investable universe accordingly.

---

## Configuration Files

All universe configurations are located in:
```
configs/universes/
├── tier1_initial_20.yaml
├── tier2_core_50.yaml
├── tier3_expanded_100.yaml
├── tier4_broad_200.yaml
└── tier5_comprehensive_300plus.yaml
```

Each file includes:
- Complete ticker list
- Eligibility criteria
- Detailed notes on composition
- Use case recommendations
- Risk warnings

---

## Next Steps

1. **Ingest Data**: Start with Tier 1, then progressively add tiers as needed
2. **Test Strategy**: Develop strategies on Tier 1, validate on Tier 2-3
3. **Compare Results**: Run same strategy across multiple tiers to check scaling
4. **Monitor Costs**: Track transaction costs carefully on higher tiers
5. **Validate Live**: Only trade strategies validated on Tier 1-2 (highest liquidity)

---

## References

- [Initial 20 ETF Universe Design](../session-notes/2026-01-06-project-setup.md)
- [Strategy Research Plan](../STRATEGY_RESEARCH_PLAN.md)
- [Backtest Engine Documentation](../src/quantetf/backtest/README.md)
- [Universe Provider Implementation](../src/quantetf/universe/README.md)

---

**Questions or Issues?**
See [`configs/universes/README.md`](../configs/universes/README.md) or file an issue in the project repository.
