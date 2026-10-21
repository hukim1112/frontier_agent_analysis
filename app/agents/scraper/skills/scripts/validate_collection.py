"""Validate a JSON array or JSONL collection without modifying the input."""

import argparse
import json
import math
from pathlib import Path


TYPE_CHECKS = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: (
        isinstance(value, int) and not isinstance(value, bool)
        or isinstance(value, float) and math.isfinite(value)
    ),
    "boolean": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, list),
    "object": lambda value: isinstance(value, dict),
}


def reject_constant(value):
    raise ValueError(f"Non-JSON numeric constant: {value}")


def load_records(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            return [json.loads(line, parse_constant=reject_constant) for line in stream if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)
    if not isinstance(value, list):
        raise ValueError("The input must be a JSON array of records, or a .jsonl file.")
    return value


def missing(value):
    return value is None or (isinstance(value, str) and not value.strip()) or value == [] or value == {}


def validate(records, *, required=(), unique_key=(), field_types=None, min_count=1, expected_count=None):
    field_types = field_types or {}
    issues = {}
    seen = set()
    duplicate_count = 0
    duplicate_examples = []

    def issue(kind, index):
        detail = issues.setdefault(kind, {"count": 0, "row_indexes": []})
        detail["count"] += 1
        if len(detail["row_indexes"]) < 5:
            detail["row_indexes"].append(index)

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            issue("record_not_object", index)
            continue
        for field in required:
            if field not in record or missing(record[field]):
                issue(f"missing_required:{field}", index)
        for field, type_name in field_types.items():
            if field not in record or not TYPE_CHECKS[type_name](record[field]):
                issue(f"invalid_type:{field}:{type_name}", index)
        if unique_key:
            if any(field not in record or missing(record[field]) for field in unique_key):
                issue("missing_unique_key", index)
                continue
            key = json.dumps([record[field] for field in unique_key], ensure_ascii=False, sort_keys=True)
            if key in seen:
                duplicate_count += 1
                if len(duplicate_examples) < 5:
                    duplicate_examples.append(index)
            seen.add(key)

    count = len(records)
    count_errors = []
    if count < min_count:
        count_errors.append(f"Expected at least {min_count} records; found {count}.")
    if expected_count is not None and count != expected_count:
        count_errors.append(f"Expected exactly {expected_count} records; found {count}.")
    return {
        "checks_passed": not issues and not count_errors and not duplicate_count,
        "record_count": count,
        "unique_count": len(seen) if unique_key else None,
        "duplicate_count": duplicate_count if unique_key else None,
        "duplicate_row_indexes": duplicate_examples,
        "field_issues": issues,
        "count_errors": count_errors,
        "coverage": (
            "not_established" if expected_count is None else
            "count_matches_expected" if count == expected_count else "count_mismatch"
        ),
        "note": "Checks apply only to this file. Verify source coverage, pagination termination and value accuracy separately.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--required", nargs="+", default=[])
    parser.add_argument("--unique-key", nargs="+", default=[])
    parser.add_argument("--type", action="append", default=[], metavar="FIELD:TYPE")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.min_count < 0 or (args.expected_count is not None and args.expected_count < 0):
        parser.error("Counts cannot be negative.")
    field_types = {}
    for specification in args.type:
        field, separator, type_name = specification.partition(":")
        if not field or not separator or type_name not in TYPE_CHECKS:
            parser.error(f"Invalid type {specification!r}; use FIELD: one of {', '.join(TYPE_CHECKS)}.")
        field_types[field] = type_name
    try:
        result = validate(
            load_records(args.input), required=args.required, unique_key=args.unique_key,
            field_types=field_types, min_count=args.min_count, expected_count=args.expected_count,
        )
        exit_code = 0 if result["checks_passed"] else 1
    except (OSError, ValueError) as exc:
        result = {"checks_passed": False, "error": str(exc)}
        exit_code = 2
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        if args.report.resolve() == args.input.resolve():
            parser.error("--report must differ from the input file.")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(content + "\n", encoding="utf-8")
    print(content)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
