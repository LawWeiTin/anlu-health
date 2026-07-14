import sys
from datetime import date
from pathlib import Path

import yaml


def main() -> int:
    registry = Path("data/source_registry.yaml")
    data = yaml.safe_load(registry.read_text(encoding="utf-8"))
    reviewed = date.fromisoformat(str(data["last_reviewed"]))
    age = (date.today() - reviewed).days
    print(f"source registry age: {age} days")
    if age > 90:
        print("source registry review is overdue", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
