#!/usr/bin/env python3
"""Validate changelog entry format."""

import yaml
import sys

try:
    with open("changelog_entry.yaml", "r") as f:
        entry = yaml.safe_load(f)

    if not isinstance(entry, list):
        print("::error::changelog_entry.yaml must contain a list")
        sys.exit(1)

    for item in entry:
        if "bump" not in item:
            print(
                "::error::Each entry must have a bump field (major, minor, or patch)"
            )
            sys.exit(1)
        if item["bump"] not in ["major", "minor", "patch"]:
            print(f"::error::Invalid bump type: {item['bump']}")
            sys.exit(1)
        if "changes" not in item:
            print("::error::Each entry must have a changes field")
            sys.exit(1)

    print("✓ Changelog entry is valid")
except Exception as e:
    print(f"::error::Error parsing changelog_entry.yaml: {e}")
    sys.exit(1)
