#!/usr/bin/env python3
"""Zips data/submission/*.csv into submission.zip with the submission/
folder itself INSIDE the archive (not just the CSVs at the zip root) --
this is a spec requirement, and getting it wrong burns an attempt for a
reason that has nothing to do with your model's quality.

Runs validate_submission.py automatically before zipping and refuses to
produce a zip if there are blocking errors (use --force to override, not
recommended).

Usage:
    python scripts/package_submission.py
    python scripts/package_submission.py --src data/submission --out submission.zip
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT_DIR / "data" / "submission"))
    ap.add_argument("--out", default=str(ROOT_DIR / "submission.zip"))
    ap.add_argument("--force", action="store_true",
                     help="zip even if validate_submission.py reports errors (not recommended)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    validate_script = ROOT_DIR / "scripts" / "validate_submission.py"
    result = subprocess.run(
        [sys.executable, str(validate_script), str(src)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0 and not args.force:
        print("Validation failed -- refusing to package. Fix errors or pass --force.")
        sys.exit(1)

    csv_files = sorted(src.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {src}, nothing to package.")
        sys.exit(1)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for csv_path in csv_files:
            # arcname places every file under submission/<name>.csv inside
            # the zip, regardless of the --src path on disk.
            arcname = f"submission/{csv_path.name}"
            zf.write(csv_path, arcname=arcname)

    print(f"Wrote {out} with {len(csv_files)} CSV file(s) under submission/.")

    # Final sanity check on the zip itself.
    result = subprocess.run(
        [sys.executable, str(validate_script), str(out)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("WARNING: zip failed post-packaging validation -- inspect before uploading.")
        sys.exit(1)


if __name__ == "__main__":
    main()
