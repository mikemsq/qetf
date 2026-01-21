#!/usr/bin/env python3
"""
Real-time Walk-Forward Validation Monitor
Tracks progress of walk-forward testing in real-time
"""
import os
import time
import subprocess
from pathlib import Path
from datetime import datetime

def monitor_walk_forward():
    """Monitor walk-forward test progress"""
    os.chdir("/workspaces/qetf")
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║        WALK-FORWARD VALIDATION - REAL-TIME MONITOR                          ║
║                                                                              ║
║  Process: python scripts/walk_forward_test.py                               ║
║  Config:  artifacts/optimization/20260115_232613/best_strategy.yaml         ║
║  Status:  RUNNING (PID: 4393)                                               ║
║  Started: 2026-01-17 16:43:00 UTC                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Monitoring progress...
Expected runtime: 60-120 minutes
Success criterion: Test/training ratio ≥70% across all windows

─────────────────────────────────────────────────────────────────────────────
MONITORING LOG
─────────────────────────────────────────────────────────────────────────────
""")
    
    start_time = time.time()
    check_interval = 30  # Check every 30 seconds
    last_size = 0
    stale_count = 0
    
    while True:
        try:
            # Check if process still running
            result = subprocess.run(["pgrep", "-f", "walk_forward_test.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                # Process finished
                elapsed = time.time() - start_time
                print(f"\n✓ Walk-forward validation COMPLETED")
                print(f"  Total runtime: {elapsed/60:.1f} minutes")
                print(f"  Status: PROCESS FINISHED")
                
                # Try to read results
                results_dir = Path("artifacts/backtests")
                if results_dir.exists():
                    latest = sorted([d for d in results_dir.glob("*walk*") if d.is_dir()], 
                                   reverse=True)
                    if latest:
                        print(f"\n  Results: {latest[0].name}")
                        results_file = latest[0] / "results.txt"
                        if results_file.exists():
                            print(f"  Output: {results_file}")
                break
            
            # Check for output files
            output_files = list(Path(".").glob("walk_forward_*.log"))
            if output_files:
                latest_log = max(output_files, key=lambda p: p.stat().st_mtime)
                current_size = latest_log.stat().st_size
                
                elapsed = time.time() - start_time
                elapsed_min = elapsed / 60
                
                if current_size > last_size:
                    stale_count = 0
                    # Show last few lines
                    with open(latest_log) as f:
                        lines = f.readlines()
                        recent = lines[-3:] if len(lines) > 3 else lines
                        for line in recent:
                            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {line.rstrip()}")
                else:
                    stale_count += 1
                    if stale_count % 10 == 1:  # Every 5 minutes
                        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Processing... ({elapsed_min:.0f}min)")
                
                last_size = current_size
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print(f"\n⚠ Monitoring stopped by user (process still running)")
            break
        except Exception as e:
            print(f"⚠ Error: {e}")
            time.sleep(check_interval)

if __name__ == "__main__":
    monitor_walk_forward()
