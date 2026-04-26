#!/usr/bin/env python3
"""Precompute historical Points O/U recommendations for fast app loading."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamlit_app.points_ou_model import (  # noqa: E402
    HISTORICAL_BUCKET_SUMMARY_CSV,
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


def build_bucket_summary() -> pd.DataFrame:
    historical_rows = score_historical_points_ou()
    if historical_rows.empty:
        return historical_rows

    bucket_rows = historical_rows[
        historical_rows["actual_side_calc"].isin(["under", "over"])
    ].copy()
    if bucket_rows.empty:
        return bucket_rows

    bucket_rows["line_numeric"] = pd.to_numeric(bucket_rows["line"], errors="coerce")
    bucket_rows["q50_numeric"] = pd.to_numeric(bucket_rows["q50"], errors="coerce")
    bucket_rows["over_price_numeric"] = pd.to_numeric(bucket_rows["over_price"], errors="coerce")
    bucket_rows["under_price_numeric"] = pd.to_numeric(bucket_rows["under_price"], errors="coerce")
    bucket_rows["raw_selection_side"] = bucket_rows["model_recommendation"].where(
        bucket_rows["model_recommendation"].isin(["over", "under"]),
        pd.NA,
    )
    bucket_rows["raw_pick_odds"] = pd.NA
    bucket_rows.loc[bucket_rows["raw_selection_side"].eq("over"), "raw_pick_odds"] = (
        bucket_rows.loc[bucket_rows["raw_selection_side"].eq("over"), "over_price_numeric"]
    )
    bucket_rows.loc[bucket_rows["raw_selection_side"].eq("under"), "raw_pick_odds"] = (
        bucket_rows.loc[bucket_rows["raw_selection_side"].eq("under"), "under_price_numeric"]
    )
    bucket_rows["raw_is_correct"] = (
        bucket_rows["raw_selection_side"].fillna("").astype(str)
        == bucket_rows["actual_side_calc"].fillna("").astype(str)
    )
    bucket_rows["raw_profit"] = np.where(
        bucket_rows["raw_selection_side"].isin(["over", "under"])
        & bucket_rows["raw_is_correct"]
        & pd.to_numeric(bucket_rows["raw_pick_odds"], errors="coerce").gt(1.0),
        pd.to_numeric(bucket_rows["raw_pick_odds"], errors="coerce") - 1.0,
        np.where(bucket_rows["raw_selection_side"].isin(["over", "under"]), -1.0, np.nan),
    )
    bucket_rows["abs_q50_line_gap"] = (
        bucket_rows["q50_numeric"] - bucket_rows["line_numeric"]
    ).abs()
    bucket_rows["gap_bucket"] = pd.cut(
        bucket_rows["abs_q50_line_gap"],
        bins=[-float("inf"), 1, 2, 3, 4, 5, 7, 10, float("inf")],
        labels=["<1", "1-2", "2-3", "3-4", "4-5", "5-7", "7-10", "10+"],
        right=False,
    )
    summary = (
        bucket_rows.dropna(subset=["gap_bucket", "raw_profit"])
        .groupby("gap_bucket", observed=True)
        .agg(
            bets=("raw_profit", "size"),
            roi=("raw_profit", "mean"),
            accuracy=("raw_is_correct", "mean"),
        )
        .reset_index()
    )
    summary["bucket_label"] = summary["gap_bucket"].astype(str)
    return summary


def main() -> None:
    selected = build_selected_bets()
    if selected.empty:
        print("No historical Points O/U recommendations were generated.")
        return

    HISTORICAL_SELECTED_BETS_CSV.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(HISTORICAL_SELECTED_BETS_CSV, index=False)
    bucket_summary = build_bucket_summary()
    if not bucket_summary.empty:
        bucket_summary.to_csv(HISTORICAL_BUCKET_SUMMARY_CSV, index=False)

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
    if not bucket_summary.empty:
        print(f"Saved bucket summary to {HISTORICAL_BUCKET_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
