from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    for section, rows in data.items():
        print("\n" + section)
        print("=" * len(section))
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
