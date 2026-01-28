#!/usr/bin/env python3
"""
Repository cleanup script for CWSN project.
Organizes root-level scripts into proper directories.
"""

import os
import shutil
from pathlib import Path

def main():
    repo_root = Path(__file__).parent
    
    # Create directories
    analysis_dir = repo_root / "scripts" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Created {analysis_dir}")
    
    tests_dir = repo_root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Created {tests_dir}")
    
    # Files to reorganize
    analysis_scripts = {
        "analyze_chain_detailed.py": analysis_dir,
        "master_analysis.py": analysis_dir,
        "improved_analysis.py": analysis_dir,
        "analyze_usage.py": analysis_dir,
        "check_logp.py": analysis_dir,
        "make_cov.py": analysis_dir,
    }
    
    test_scripts = {
        "test_likelihood.py": tests_dir,
    }
    
    # Move analysis scripts
    for script, dest in analysis_scripts.items():
        src = repo_root / script
        if src.exists():
            shutil.move(str(src), str(dest / script))
            print(f"[+] Moved {script} -> scripts/analysis/")
        else:
            print(f"[-] Not found: {script}")
    
    # Move test scripts
    for script, dest in test_scripts.items():
        src = repo_root / script
        if src.exists():
            shutil.move(str(src), str(dest / script))
            print(f"[+] Moved {script} -> tests/")
        else:
            print(f"[-] Not found: {script}")
    
    print("\n[+] Cleanup complete!")
    print("\nReminder: Update imports/paths in moved scripts if needed.")

if __name__ == "__main__":
    main()
