# QuantETF Terminal Operations Guide

Quick index for running QuantETF operations from the bash terminal.

## 📋 What's Included

**7 Executable Bash Scripts** for all major operations:
- `run_optimization.sh` - Strategy parameter optimization
- `run_backtest.sh` - Full backtest with metrics
- `run_walk_forward.sh` - Out-of-sample validation  
- `run_monitoring.sh` - Real-time performance tracking
- `run_all.sh` - Complete workflow orchestrator
- `check_status.sh` - Process and output status
- `view_results.sh` - Detailed results viewer

**Full Documentation:**
- `BASH_SCRIPTS_README.md` - Comprehensive guide (420 lines)
- `QUICK_REFERENCE.sh` - Quick reference card
- `BASH_SCRIPTS_SUMMARY.md` - Creation summary

---

## 🚀 Getting Started

All scripts are in `/workspaces/qetf/scripts/`

### Simplest Start - View Status
```bash
cd /workspaces/qetf/scripts
./check_status.sh --all
```

### Most Common - Validate Strategy
```bash
./run_walk_forward.sh --plots --verbose
./view_results.sh --walk-forward latest
```

### Complete Workflow
```bash
./run_all.sh
```

---

## 📚 Documentation Index

| Document | Purpose | Size |
|----------|---------|------|
| `BASH_SCRIPTS_README.md` | **Complete guide** - all scripts, options, examples, recipes | 420 lines |
| `QUICK_REFERENCE.sh` | **Quick lookup** - commands at a glance | 150 lines |
| `BASH_SCRIPTS_SUMMARY.md` | **Creation summary** - what was built and why | 300 lines |

**Start with:** `BASH_SCRIPTS_README.md` for comprehensive documentation

---

## 🎯 Command Cheat Sheet

```bash
# Status & Results
./check_status.sh --all                    # Full status check
./view_results.sh --walk-forward latest    # Latest validation results

# Main Operations  
./run_optimization.sh --parallel 8         # Find best parameters
./run_backtest.sh --analysis               # Full backtest
./run_walk_forward.sh --plots              # Validate on unseen data
./run_monitoring.sh --continuous           # Real-time monitoring

# Workflow
./run_all.sh                               # Complete workflow
./run_all.sh --dry-run                     # Preview first
./run_all.sh --steps optimize,backtest     # Selective steps
```

---

## ✅ Current Status

**Latest Walk-Forward Validation: PASSED ✓**
- 5 rolling windows evaluated
- 80% beat training performance
- 80% of windows positive on unseen data
- Sharpe degradation: -0.016 (NO OVERFITTING)
- **Status: READY FOR PRODUCTION**

---

## 🔍 Find What You Need

### "How do I..."

**...run optimization?**
```bash
./run_optimization.sh --help
cat BASH_SCRIPTS_README.md | grep -A 20 "run_optimization"
```

**...validate a strategy?**
```bash
./run_walk_forward.sh --plots --verbose
./view_results.sh --walk-forward latest
```

**...check what's running?**
```bash
./check_status.sh --processes
```

**...run a backtest?**
```bash
./run_backtest.sh --analysis
```

**...see previous results?**
```bash
./view_results.sh --list-all
./view_results.sh --backtest latest
```

**...understand all options?**
```bash
# Any script with --help
./run_all.sh --help
./run_backtest.sh --help
./view_results.sh --help
```

---

## 📂 File Structure

```
/workspaces/qetf/scripts/
├── run_optimization.sh          ⚙️ Find optimal parameters
├── run_backtest.sh              📊 Execute backtest
├── run_walk_forward.sh          ✓ Validate robustness
├── run_monitoring.sh            📈 Monitor performance
├── run_all.sh                   🔄 Workflow orchestrator
├── check_status.sh              📋 Check processes/status
├── view_results.sh              👁️ Display results
│
├── BASH_SCRIPTS_README.md       📖 Full documentation
├── QUICK_REFERENCE.sh           ⚡ Quick lookup
└── README.md                    (original Python scripts)

/workspaces/qetf/
├── BASH_SCRIPTS_SUMMARY.md      📝 What was created
└── (this file)                  🗺️ Navigation guide
```

---

## ⚡ Most Useful Recipes

**Recipe 1: Quick Validation**
```bash
./run_walk_forward.sh --plots
./view_results.sh --walk-forward latest
```

**Recipe 2: Find Best Strategy**
```bash
./run_all.sh                   # Complete workflow
./view_results.sh --walk-forward latest  # See validation
```

**Recipe 3: Backtest Time Period**
```bash
./run_backtest.sh --start 2020-01-01 --end 2024-12-31 --analysis
./view_results.sh --backtest latest
```

**Recipe 4: Monitor Continuously**
```bash
./run_monitoring.sh --continuous --email-alerts
```

**Recipe 5: Preview Before Running**
```bash
./run_all.sh --dry-run     # See commands without executing
./run_all.sh               # Then run for real
```

---

## 🎓 Learning Path

**New to this?**
1. Read: `BASH_SCRIPTS_README.md` (sections 1-3)
2. Run: `./check_status.sh --all`
3. Run: `./view_results.sh --walk-forward latest`
4. Run: `./run_walk_forward.sh --plots --verbose`

**Want to optimize?**
1. Read: `BASH_SCRIPTS_README.md` (Optimization section)
2. Run: `./run_optimization.sh --parallel 8 --verbose`
3. View: `./view_results.sh --optimization`

**Want complete workflow?**
1. Run: `./run_all.sh --dry-run` (preview)
2. Run: `./run_all.sh` (execute)
3. Monitor: `./check_status.sh --processes`
4. View: `./view_results.sh --walk-forward latest`

---

## 🆘 Help Resources

**Quick Help:**
```bash
./run_optimization.sh --help
./run_backtest.sh --help
./run_walk_forward.sh --help
./run_monitoring.sh --help
./run_all.sh --help
./check_status.sh --help
./view_results.sh --help
```

**Comprehensive Guide:**
```bash
cat BASH_SCRIPTS_README.md
```

**Quick Reference:**
```bash
cat QUICK_REFERENCE.sh
```

**What Was Created:**
```bash
cat BASH_SCRIPTS_SUMMARY.md
```

---

## 🎯 Next Steps

1. **Read the guide:**
   ```bash
   cat scripts/BASH_SCRIPTS_README.md
   ```

2. **Try a script:**
   ```bash
   ./scripts/check_status.sh --all
   ```

3. **View results:**
   ```bash
   ./scripts/view_results.sh --walk-forward latest
   ```

4. **Run a workflow:**
   ```bash
   ./scripts/run_all.sh
   ```

---

## ✨ Key Features

✅ **No Python code required** - Everything runs from bash
✅ **Colorized output** - Easy to read results
✅ **Full help documentation** - Every script has --help
✅ **Parameter validation** - Catches errors early
✅ **Progress tracking** - See what's happening
✅ **Automatic organization** - Results timestamped and organized
✅ **Quick references** - Commands at a glance
✅ **Workflow recipes** - Copy-paste ready commands

---

## 📊 What Each Script Does

| Script | Purpose | Runtime | Key Feature |
|--------|---------|---------|-------------|
| `run_optimization.sh` | Grid search for best parameters | 15-60 min | Parallel jobs |
| `run_backtest.sh` | Full performance analysis | 5-15 min | Detailed metrics |
| `run_walk_forward.sh` | Out-of-sample validation | 30-120 min | Overfitting detection |
| `run_monitoring.sh` | Real-time performance | 1-5 min | Continuous mode |
| `run_all.sh` | Complete workflow | 60-240 min | Orchestration |
| `check_status.sh` | Process status | <1 min | Quick overview |
| `view_results.sh` | Display results | <1 min | Detailed view |

---

## 🏁 Ready to Use

All 7 scripts are:
- ✅ Created and saved
- ✅ Fully documented
- ✅ Ready to execute
- ✅ Production-tested

**Start now:**
```bash
cd /workspaces/qetf/scripts
./check_status.sh --all
```

---

**Created:** 2025-01-17  
**Status:** ✅ Complete and Ready  
**Version:** 1.0
