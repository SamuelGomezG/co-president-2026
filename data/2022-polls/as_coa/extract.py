"""SPEC: Extract clean CSVs from AS/COA Infogram payloads for both 2022 rounds.

Source: AS/COA Poll Tracker pages
  - 1st round: https://www.as-coa.org/articles/poll-tracker-colombias-2022-presidential-election
  - Runoff:    https://www.as-coa.org/articles/poll-tracker-colombias-2022-presidential-runoff

Method: Infogram inlines `window.infographicData` in the public viewer HTML.
        We grab the page, extract that JSON object, and parse the chart entities.

Re-run: `uv run python data/2022-polls/as_coa/extract.py`
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import re
from typing import Any

HERE = Path(__file__).resolve().parent

_DATE_EXPECTED_PARTS: int = 2

MONTH_TO_NUM = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def parse_asc_date(label: str, default_year: int) -> date | None:
    """Parse labels like 'March 18', 'April 21', 'May 1', 'June 6' to ISO dates.

    Bare month names ('February', 'March') return None — caller decides bucket.
    """
    s = label.strip()
    parts = s.split()
    if len(parts) == _DATE_EXPECTED_PARTS and parts[0].lower() in MONTH_TO_NUM:
        try:
            day = int(parts[1])
            return date(default_year, MONTH_TO_NUM[parts[0].lower()], day)
        except ValueError:
            return None
    return None


def _to_float(s: object) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s).strip().rstrip("%").replace(",", ".")
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _entities(raw: dict[str, Any]) -> dict[str, Any]:
    return raw["elements"]["content"]["content"]["entities"]


def _entity_data(entities: dict[str, Any], key: str) -> list[list[Any]]:
    raw = entities[key]["data"]
    return raw[0] if raw else []


def extract_round1(raw_path: Path) -> list[dict[str, Any]]:
    """Build the round-1 timeline: one row per published aggregated value.

    Source entity `6b352ba4-abdb-44fa-955e-56c3852c03cb` has poll-by-poll dates.
    """
    raw = json.loads(raw_path.read_text())
    entities = _entities(raw)
    table = _entity_data(entities, "6b352ba4-2e91-43a1-b659-aa5b60dbdb81")
    header_row = table[0]
    columns = [str(c).strip() for c in header_row[1:]]
    rows: list[dict[str, Any]] = []
    for row in table[1:]:
        date_label = str(row[0]).strip()
        d = parse_asc_date(date_label, 2022)
        if d is None:
            continue
        out: dict[str, Any] = {
            "fecha": d.isoformat(),
            "fuente": "AS/COA",
            "encuestadora": "AGREGADO",
            "ambito": "nacional",
        }
        for col, val in zip(columns, row[1:], strict=False):
            key = _slug_col(col)
            out[key] = _to_float(val)
        rows.append(out)
    rows.sort(key=lambda r: r["fecha"])
    return rows


def extract_runoff(raw_path: Path) -> list[dict[str, Any]]:
    """Build the runoff timeline: one row per published aggregated value.

    Source entity `86cf9962-e39d-422d-8eaa-b26e7350e792` has the date-keyed table.
    """
    raw = json.loads(raw_path.read_text())
    entities = _entities(raw)
    table = _entity_data(entities, "86cf9962-e39d-422d-8eaa-b26e7350e792")
    header_row = table[0]
    columns = [str(c).strip() for c in header_row[1:]]
    rows: list[dict[str, Any]] = []
    for row in table[1:]:
        date_label = str(row[0]).strip()
        d = parse_asc_date(date_label, 2022)
        if d is None:
            continue
        out: dict[str, Any] = {
            "fecha": d.isoformat(),
            "fuente": "AS/COA",
            "encuestadora": "AGREGADO",
            "ambito": "nacional",
        }
        for col, val in zip(columns, row[1:], strict=False):
            key = _slug_col(col)
            out[key] = _to_float(val)
        rows.append(out)
    rows.sort(key=lambda r: r["fecha"])
    return rows


def extract_transfer_matrix(raw_path: Path) -> list[dict[str, Any]]:
    """Build the 1R-vote → 2R-candidate transfer table.

    Source entity `38e6e966-88e3-4924-95cf-60ab29827ec3` from the runoff tracker.
    """
    raw = json.loads(raw_path.read_text())
    entities = _entities(raw)
    table = _entity_data(entities, "38e6e966-88e3-4924-95cf-60ab29827ec3")
    table[0]
    out_rows: list[dict[str, Any]] = []
    for row in table[1:]:
        label = str(row[0])
        match = re.match(r"^(.*?)\s*\((\d+(?:\.\d+)?)%\)\s*$", label)
        if not match:
            continue
        candidato_1r = match.group(1).strip()
        pct_1r = float(match.group(2))
        out_rows.append(
            {
                "candidato_primera_vuelta": candidato_1r,
                "pct_primera_vuelta": pct_1r,
                "petro_runoff": _to_float(row[1]),
                "hernandez_runoff": _to_float(row[2]),
                "blanco_runoff": _to_float(row[3]),
                "ns_nr_runoff": _to_float(row[4]),
                "fuente": "AS/COA",
            }
        )
    return out_rows


def _slug_col(label: str) -> str:
    s = label.strip().lower()
    s = s.replace("'", "").replace("\u2019", "")
    s = (
        s.replace("\u00e9", "e")
        .replace("\u00e1", "a")
        .replace("\u00ed", "i")
        .replace("\u00f3", "o")
        .replace("\u00fa", "u")
        .replace("\u00f1", "n")
    )
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    mapping = {
        "petro": "petro",
        "gustavo_petro": "petro",
        "gutierrez": "fico_gutierrez",
        "federico_gutierrez": "fico_gutierrez",
        "hernandez": "rodolfo_hernandez",
        "rodolfo_hernandez": "rodolfo_hernandez",
        "fajardo": "sergio_fajardo",
        "sergio_fajardo": "sergio_fajardo",
        "betancourt": "ingrid_betancourt",
        "ingrid_betancourt": "ingrid_betancourt",
        "rodriguez": "john_milton_rodriguez",
        "john_milton_rodriguez": "john_milton_rodriguez",
        "blanco": "blanco",
        "blank": "blanco",
        "blank_vote": "blanco",
        "ninguno": "ninguno",
        "none": "ninguno",
        "ns_nr": "ns_nr",
        "not_sure": "ns_nr",
        "undecided": "ns_nr",
        "neither": "ns_nr",
        "otro": "otro",
        "otros": "otro",
    }
    return mapping.get(s, s)


def _all_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts as a CSV to *path*.

    Args:
        path: Destination file path.
        rows: List of dicts with uniform keys.

    """
    if not rows:
        path.write_text("")
        return
    keys = _all_keys(rows)
    lines = [",".join(keys)]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k)
            if v is None:
                cells.append("")
            else:
                cells.append(str(v))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Extract clean CSVs from AS/COA Infogram payloads for both rounds."""
    round1 = extract_round1(HERE / "round1_raw.json")
    runoff = extract_runoff(HERE / "runoff_raw.json")
    transfer = extract_transfer_matrix(HERE / "runoff_raw.json")

    write_csv(HERE / "round1.csv", round1)
    write_csv(HERE / "runoff.csv", runoff)
    write_csv(HERE / "transfer_matrix.csv", transfer)


if __name__ == "__main__":
    main()
