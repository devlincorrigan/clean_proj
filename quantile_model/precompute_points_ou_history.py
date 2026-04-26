#!/usr/bin/env python3
"""Precompute historical Points O/U recommendations for fast app loading."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamlit_app.points_ou_model import (  # noqa: E402
    HISTORICAL_SELECTED_BETS_CSV,
    score_historical_points_ou,
)


def build_selected_bets() -> pd.DataFrame:
    historical_rows = score_historical_points_ou()
    if historical_rows.empty:
        return historical_rows

    selected = historical_rows[
        historical_rows["is_recommended"].fillna(False)
        & historical_rows["actual_side_calc"].isin(["under", "over"])
    ].copy()
    if selected.empty:
        return selected

    selected["game_date_sort"] = pd.to_datetime(selected["game_date"], errors="coerce")
    selected["bookmaker_last_update_sort"] = pd.to_datetime(
        selected["bookmaker_last_update"], utc=True, errors="coerce"
    )
    selected = selected.sort_values(
        [
            "game_date_sort",
            "bookmaker_last_update_sort",
            "player_name_odds",
            "line_points",
        ],
        ascending=[True, True, True, True],
    ).drop(columns=["game_date_sort", "bookmaker_last_update_sort"])
    return selected.reset_index(drop=True)


def main() -> None:
    selected = build_selected_bets()
    if selected.empty:
        print("No historical Points O/U recommendations were generated.")
        return

    HISTORICAL_SELECTED_BETS_CSV.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(HISTORICAL_SELECTED_BETS_CSV, index=False)

    game_dates = pd.to_datetime(selected["game_date"], errors="coerce").dropna()
    start_label = game_dates.min().date().isoformat() if not game_dates.empty else "N/A"
    end_label = game_dates.max().date().isoformat() if not game_dates.empty else "N/A"
    correct = int(selected["is_correct"].fillna(False).sum())
    total = int(len(selected))
    accuracy = correct / total if total else float("nan")

    print(f"Saved historical Points O/U recommendations to {HISTORICAL_SELECTED_BETS_CSV}")
    print(f"rows: {total}")
    print(f"correct: {correct}")
    print(f"accuracy: {accuracy:.4f}" if pd.notna(accuracy) else "accuracy: nan")
    print(f"test_window: {start_label} to {end_label}")


if __name__ == "__main__":
    main()
