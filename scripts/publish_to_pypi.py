#!/usr/bin/env python3
"""
Script to publish py-statmatch to PyPI locally.

This is a fallback option when GitHub Actions publishing isn't working.

Prerequisites:
1. Install twine: pip install twine
2. Have PyPI credentials ready (username: __token__, password: your-api-token)
3. Build the package first: python -m build

Usage:
    python scripts/publish_to_pypi.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Check if dist directory exists
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("Error: dist/ directory not found. Run 'python -m build' first.")
        sys.exit(1)
    
    # Find the latest wheel and sdist
    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    
    if not wheels or not sdists:
        print("Error: No wheel or sdist files found in dist/")
        print("Run 'python -m build' to create distribution files.")
        sys.exit(1)
    
    # Get the latest files
    latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
    latest_sdist = max(sdists, key=lambda p: p.stat().st_mtime)
    
    print(f"Found distribution files:")
    print(f"  - {latest_wheel.name}")
    print(f"  - {latest_sdist.name}")
    
    # Confirm before uploading
    response = input("\nUpload to PyPI? [y/N] ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Upload using twine
    cmd = [
        sys.executable, "-m", "twine", "upload",
        "--skip-existing",
        str(latest_wheel),
        str(latest_sdist)
    ]
    
    print("\nRunning:", " ".join(cmd))
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\nSuccessfully uploaded to PyPI!")
        print("Install with: pip install py-statmatch")
    else:
        print("\nUpload failed. Please check your PyPI credentials.")
        print("You can set up credentials by creating ~/.pypirc or using environment variables.")
        sys.exit(1)


if __name__ == "__main__":
    main()