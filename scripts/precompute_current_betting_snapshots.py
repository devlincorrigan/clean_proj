#!/usr/bin/env python3
"""Precompute current scored betting snapshots for fast Streamlit loading."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamlit_app.points_ou_model import (  # noqa: E402
    CURRENT_SCORED_ROWS_CSV as POINTS_OU_CURRENT_SCORED_ROWS_CSV,
    score_current_points_ou,
)
from streamlit_app.score_live_odds import (  # noqa: E402
    CURRENT_SCORED_ROWS_CSV as POINTS_ALTERNATE_CURRENT_SCORED_ROWS_CSV,
    score_current_points_ladders,
)


def write_snapshot(label: str, rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False)
    print(f"{label}: saved {len(rows):,} rows to {output_path}")


def main() -> None:
    points_ou_rows = score_current_points_ou()
    if points_ou_rows.empty:
        print("Points O/U: no scored rows were generated.")
    else:
        write_snapshot("Points O/U", points_ou_rows, POINTS_OU_CURRENT_SCORED_ROWS_CSV)

    points_alternate_rows = score_current_points_ladders()
    if points_alternate_rows.empty:
        print("Points Alternate: no scored rows were generated.")
    else:
        write_snapshot(
            "Points Alternate",
            points_alternate_rows,
            POINTS_ALTERNATE_CURRENT_SCORED_ROWS_CSV,
        )


if __name__ == "__main__":
    main()
