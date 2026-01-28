"""Tests for CLI scripts."""

import subprocess
import sys
from pathlib import Path


def test_monitor_script_help():
    """Monitor script should show help."""
    result = subprocess.run(
        [sys.executable, "scripts/run_daily_monitor.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Daily Regime Monitor" in result.stderr or "Update daily regime state" in result.stdout


def test_rebalance_script_help():
    """Rebalance script should show help."""
    result = subprocess.run(
        [sys.executable, "scripts/run_regime_rebalance.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Regime-Aware" in result.stderr or "regime-aware" in result.stdout


def test_status_script_no_state():
    """Status script should handle missing state gracefully."""
    result = subprocess.run(
        [sys.executable, "scripts/show_regime_status.py", "--state-dir", "/nonexistent"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "No regime state found" in result.stdout


def test_status_script_help():
    """Status script should show help."""
    result = subprocess.run(
        [sys.executable, "scripts/show_regime_status.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "current regime status" in result.stdout.lower() or "state-dir" in result.stdout


def test_monitor_script_verbose_flag():
    """Monitor script should accept verbose flag."""
    result = subprocess.run(
        [sys.executable, "scripts/run_daily_monitor.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--verbose" in result.stdout


def test_rebalance_script_dry_run_flag():
    """Rebalance script should accept dry-run flag."""
    result = subprocess.run(
        [sys.executable, "scripts/run_regime_rebalance.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--dry-run" in result.stdout


def test_status_script_json_flag():
    """Status script should accept json flag."""
    result = subprocess.run(
        [sys.executable, "scripts/show_regime_status.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--json" in result.stdout
