#!/usr/bin/env python3
"""Validates a submission directory (or zip) against the competition's
exact CSV format BEFORE you upload it to Codabench -- a malformed
submission still burns one of your 3 attempts per query package, so this
check is meant to be run every single time, with zero exceptions.

Usage:
    python scripts/validate_submission.py data/submission/
    python scripts/validate_submission.py data/submission.zip

Checks performed:
  - every *.csv is UTF-8 decodable
  - no header row (heuristic: first row's 2nd column isn't a non-numeric
    word like "frame" -- real header detection isn't perfectly reliable,
    treat this as a smell test, not a proof)
  - correct column count per inferred task (kis=2, qa=3, trake>=2)
  - frame IDs parse as integers
  - video filename has no .mp4/.mkv/.avi extension
  - <= 100 rows per file
  - Q&A: answer length <= 100 chars, answer field quoting looks sane
  - TRAKE: frame columns are non-decreasing (chronological order) --
    warning only, since we can't know the true event mapping locally
  - zip (if given a .zip): submission/ is the top-level dir inside it,
    not the CSVs sitting at the zip root
"""
from __future__ import annotations

import csv
import re
import sys
import zipfile
from pathlib import Path

MAX_ROWS = 100
ANSWER_MAX_CHARS = 100
VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".MP4", ".MKV", ".AVI")

_TASK_RE = re.compile(r"-(kis|qa|trake)\b", re.IGNORECASE)


class Issue:
    def __init__(self, level: str, file: str, msg: str):
        self.level = level  # "ERROR" | "WARN"
        self.file = file
        self.msg = msg

    def __str__(self):
        return f"[{self.level}] {self.file}: {self.msg}"


def infer_task(filename: str) -> str | None:
    m = _TASK_RE.search(Path(filename).stem)
    return m.group(1).lower() if m else None


def validate_csv_text(filename: str, text: str) -> list[Issue]:
    issues: list[Issue] = []
    task = infer_task(filename)
    if task is None:
        issues.append(Issue("ERROR", filename,
            "cannot infer task type (-kis/-qa/-trake) from filename"))
        return issues

    reader = csv.reader(text.splitlines())
    rows = [r for r in reader if r]

    if len(rows) == 0:
        issues.append(Issue("WARN", filename, "file is empty (0 candidate rows)"))
        return issues

    if len(rows) > MAX_ROWS:
        issues.append(Issue("ERROR", filename,
            f"{len(rows)} rows > max {MAX_ROWS} allowed"))

    # Header smell test: does the first row's frame-column look non-numeric?
    first = rows[0]
    if len(first) >= 2:
        try:
            int(first[1])
        except ValueError:
            issues.append(Issue("WARN", filename,
                f"first row's 2nd column ({first[1]!r}) isn't an integer -- "
                f"possible leftover header row (spec requires NO header)"))

    for i, row in enumerate(rows, start=1):
        video = row[0] if row else ""
        if video.endswith(VIDEO_EXTS):
            issues.append(Issue("ERROR", filename,
                f"row {i}: video filename includes an extension ({video!r}); "
                f"spec requires no .mp4/.mkv/.avi suffix"))

        if task == "kis":
            if len(row) != 2:
                issues.append(Issue("ERROR", filename,
                    f"row {i}: expected 2 columns (video,frame), got {len(row)}"))
                continue
            _check_int_frame(row[1], i, filename, issues)

        elif task == "qa":
            if len(row) != 3:
                issues.append(Issue("ERROR", filename,
                    f"row {i}: expected 3 columns (video,frame,answer), got {len(row)}"))
                continue
            _check_int_frame(row[1], i, filename, issues)
            answer = row[2]
            if len(answer) > ANSWER_MAX_CHARS:
                issues.append(Issue("ERROR", filename,
                    f"row {i}: answer exceeds {ANSWER_MAX_CHARS} chars "
                    f"({len(answer)})"))
            if answer == "":
                issues.append(Issue("WARN", filename, f"row {i}: empty answer"))

        elif task == "trake":
            if len(row) < 2:
                issues.append(Issue("ERROR", filename,
                    f"row {i}: expected video + >=1 frame columns, got {len(row)}"))
                continue
            frame_strs = row[1:]
            frames = []
            ok = True
            for fs in frame_strs:
                if not _check_int_frame(fs, i, filename, issues):
                    ok = False
            if ok:
                frames = [int(x) for x in frame_strs]
                if frames != sorted(frames):
                    issues.append(Issue("WARN", filename,
                        f"row {i}: frames not in non-decreasing (chronological) "
                        f"order: {frames}"))

        # cross-row event-count consistency for TRAKE (same query -> same N)
    if task == "trake":
        lengths = {len(r) for r in rows}
        if len(lengths) > 1:
            issues.append(Issue("WARN", filename,
                f"inconsistent column counts across rows in the same file: "
                f"{sorted(lengths)} -- expected same N (event count) throughout "
                f"since one file = one query"))

    return issues


def _check_int_frame(value: str, row_idx: int, filename: str, issues: list[Issue]) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        issues.append(Issue("ERROR", filename,
            f"row {row_idx}: frame value {value!r} is not an integer"))
        return False


def validate_dir(dir_path: Path) -> list[Issue]:
    issues: list[Issue] = []
    csv_files = sorted(dir_path.glob("*.csv"))
    if not csv_files:
        issues.append(Issue("ERROR", str(dir_path), "no .csv files found"))
        return issues
    for path in csv_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            issues.append(Issue("ERROR", path.name, f"not valid UTF-8: {e}"))
            continue
        issues.extend(validate_csv_text(path.name, text))
    return issues


def validate_zip(zip_path: Path) -> list[Issue]:
    issues: list[Issue] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        top_level_dirs = {n.split("/")[0] for n in names if "/" in n}
        if "submission" not in top_level_dirs:
            issues.append(Issue("ERROR", zip_path.name,
                "zip does not contain a top-level 'submission/' directory -- "
                "spec requires the submission folder itself to be inside the zip"))
        csv_names = [n for n in names if n.endswith(".csv")]
        if not csv_names:
            issues.append(Issue("ERROR", zip_path.name, "no .csv files found inside zip"))
        for name in csv_names:
            try:
                text = zf.read(name).decode("utf-8")
            except UnicodeDecodeError as e:
                issues.append(Issue("ERROR", name, f"not valid UTF-8: {e}"))
                continue
            issues.extend(validate_csv_text(Path(name).name, text))
    return issues


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"ERROR: path does not exist: {target}")
        sys.exit(1)

    if target.is_dir():
        issues = validate_dir(target)
    elif target.suffix == ".zip":
        issues = validate_zip(target)
    else:
        print(f"ERROR: expected a directory or .zip file, got {target}")
        sys.exit(1)

    errors = [i for i in issues if i.level == "ERROR"]
    warns = [i for i in issues if i.level == "WARN"]

    for i in issues:
        print(str(i))

    print("")
    print(f"{len(errors)} error(s), {len(warns)} warning(s).")
    if errors:
        print("DO NOT SUBMIT -- fix errors above first.")
        sys.exit(1)
    else:
        print("No blocking errors found. (Warnings are worth a look, not necessarily blockers.)")
        sys.exit(0)


if __name__ == "__main__":
    main()
