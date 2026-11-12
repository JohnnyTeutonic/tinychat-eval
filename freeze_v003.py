"""Offline v0.0.3 manifest freeze/check, by freeze.py's method (freeze.py itself is frozen in the v0.0.2 manifest).

    python freeze_v003.py --check evaluation_manifest_v003.json
"""
import argparse
import hashlib
import json
from pathlib import Path

VERSION = "v0.0.3"
STATUS = "FROZEN_RELEASE_SPECIFICATION"
FILES = ("evaluate.py", "integrations.py", "provenance.py", "suite_release_v002.json", "release_gate_v003.json",
         "gate_v003.py", "freeze_v003.py")


def manifest(root):
    return {"version": VERSION, "files": {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in FILES},
            "status": STATUS}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out")
    p.add_argument("--check")
    a = p.parse_args(argv)
    if bool(a.out) == bool(a.check):
        p.error("exactly one of --out/--check")
    m = manifest(Path(__file__).parent)
    if a.check:
        if json.loads(Path(a.check).read_text(encoding="utf-8")) != m:
            raise SystemExit("manifest drift")
        print("Frozen v0.0.3 evaluation manifest verified")
    else:
        with Path(a.out).open("x", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
