#!/usr/bin/env python3
"""Git commit script for IMPL-019."""

import subprocess
import sys

def run_git_command(cmd):
    """Run a git command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def main():
    # Add all changes
    print("Adding changes...")
    code, out, err = run_git_command("git add -A")
    if code != 0:
        print(f"Error adding changes: {err}")
        return 1
    print(f"✓ Changes added\n{out}")
    
    # Commit with detailed message
    message = """IMPL-019: DAL Core Interfaces & Types

Implemented foundational Data Access Layer infrastructure:

New Files:
- src/quantetf/data/access/__init__.py: Package initialization and exports
- src/quantetf/data/access/types.py: Enums and dataclasses (Regime, TickerMetadata, ExchangeInfo, DataAccessMetadata)
- src/quantetf/data/access/abstract.py: Abstract base classes (PriceDataAccessor, MacroDataAccessor, UniverseDataAccessor, ReferenceDataAccessor)
- src/quantetf/data/access/context.py: DataAccessContext for unified accessor container
- src/quantetf/data/access/factory.py: DataAccessFactory for accessor instantiation
- tests/data/access/test_dal_interfaces.py: Comprehensive test suite (30 tests, 100% pass rate)

Key Features:
- 4 abstract accessor interfaces with complete docstrings
- Point-in-time safety with strict inequality on as_of dates
- Frozen dataclasses for immutability
- Clean factory pattern for extensibility
- 100% test coverage of types and interfaces

Tests: 30/30 PASSING ✓

Unblocks: IMPL-020, IMPL-021, IMPL-022, IMPL-023, IMPL-024"""
    
    print("\nCommitting...")
    code, out, err = run_git_command(f'git commit -m "{message}"')
    if code != 0:
        print(f"Error committing: {err}")
        return 1
    print(f"✓ Committed successfully")
    print(out)
    
    # Show commit info
    print("\nCommit details:")
    code, out, err = run_git_command("git log -1 --stat")
    print(out)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
