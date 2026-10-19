"""Validate a results JSON against results/schema.json (uses jsonschema if installed, else a minimal check)."""
import json
import sys
from pathlib import Path

SCHEMA = Path(__file__).resolve().parent.parent / "results" / "schema.json"


def validate(result: dict) -> list[str]:
    """Return a list of error strings. Empty list = valid."""
    schema = json.loads(SCHEMA.read_text())
    try:
        import jsonschema
        v = jsonschema.Draft202012Validator(schema)
        return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(result)]
    except ImportError:
        errs = [f"missing key: {k}" for k in schema["required"] if k not in result]
        for i, lv in enumerate(result.get("levels", [])):
            errs += [f"levels/{i}: missing {k}" for k in schema["$defs"]["level"]["required"] if k not in lv]
        return errs


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        errs = validate(json.load(open(p)))
        print(p, "OK" if not errs else errs)
        bad += bool(errs)
    sys.exit(1 if bad else 0)
