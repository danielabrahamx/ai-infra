"""Write results JSON + markdown table."""
import json


def write_outputs(rows, stem, meta=None):
    json.dump({"meta": meta or {}, "results": rows}, open(f"{stem}.json", "w"), indent=2)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join("n/a" if r.get(c) is None else str(r[c]) for c in cols) + " |")
    open(f"{stem}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
