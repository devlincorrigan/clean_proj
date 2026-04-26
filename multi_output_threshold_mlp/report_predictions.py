#!/usr/bin/env python3
"""Generate side, bookmaker, probability, and dedup diagnostics for the pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from multi_output_threshold_mlp.odds_processing import (
        dedup_player_points_alternate_rows,
        load_odds_event,
        normalize_player_name,
    )
except ImportError:
    from odds_processing import (
        dedup_player_points_alternate_rows,
        load_odds_event,
        normalize_player_name,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_STAGE1 = SCRIPT_DIR / "multi_output_threshold_mlp_predictions.csv"
DEFAULT_STAGE2 = SCRIPT_DIR / "stacked_bet_quality_predictions.csv"
DEFAULT_SELECTED = SCRIPT_DIR / "walk_forward_stacked_selected_bets.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "diagnostics"
DEFAULT_MAPPING_CSV = PROJECT_ROOT / "data" / "game_event_bijection.csv"
DEFAULT_ODDS_DIR = PROJECT_ROOT / "data" / "historical_odds_api" / "historical_player_points_alternate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-input", default=str(DEFAULT_STAGE1))
    parser.add_argument("--stage2-input", default=str(DEFAULT_STAGE2))
    parser.add_argument("--selected-input", default=str(DEFAULT_SELECTED))
    parser.add_argument("--mapping-csv", default=str(DEFAULT_MAPPING_CSV))
    parser.add_argument("--odds-dir", default=str(DEFAULT_ODDS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def odds_band(price: float) -> str:
    if not np.isfinite(price):
        return "missing"
    if price < 2.0:
        return "[0, 2)"
    if price < 3.0:
        return "[2, 3)"
    if price < 5.0:
        return "[3, 5)"
    return "[5, inf)"


def probability_band(probability: float) -> str:
    if not np.isfinite(probability):
        return "missing"
    if probability < 0.40:
        return "low(<0.40)"
    if probability < 0.60:
        return "medium([0.40,0.60))"
    return "high(>=0.60)"


def availability_type(row: pd.Series) -> str:
    has_over = pd.notna(row.get("over_price"))
    has_under = pd.notna(row.get("under_price"))
    if has_over and has_under:
        return "both"
    if has_over:
        return "over_only"
    if has_under:
        return "under_only"
    return "neither"


def consensus_depth_band(value) -> str:
    count = safe_float(value)
    if count is None:
        return "missing"
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    return "4+"


def profit_from_rows(df: pd.DataFrame, price_col: str = "bet_price") -> pd.Series:
    prices = pd.to_numeric(df[price_col], errors="coerce")
    return pd.Series(
        np.where(
            df["bet_result"] == "win",
            prices - 1.0,
            np.where(df["bet_result"] == "loss", -1.0, np.nan),
        ),
        index=df.index,
    )


def load_mapping_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bookmaker_specific_line_entries(event: dict) -> list[dict]:
    entries = {}
    for bookmaker in event.get("bookmakers", []):
        bookmaker_key = str(bookmaker.get("key", "") or "")
        for market in bookmaker.get("markets", []):
            if market.get("key") != "player_points_alternate":
                continue
            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description")
                point = outcome.get("point")
                side = (outcome.get("name") or "").strip().lower()
                if player_name is None or point is None or side not in {"over", "under"}:
                    continue
                key = (bookmaker_key, normalize_player_name(player_name), float(point))
                if not key[1]:
                    continue
                entry = entries.setdefault(
                    key,
                    {
                        "bookmaker_key": bookmaker_key,
                        "line_points": float(point),
                        "has_over": False,
                        "has_under": False,
                    },
                )
                entry[f"has_{side}"] = True
    return list(entries.values())


def build_source_inventory(mapping_rows: list[dict], odds_dir: Path) -> tuple[dict, pd.DataFrame]:
    unique_event_ids = sorted({row["event_id"] for row in mapping_rows if row.get("event_id")})
    inventory = Counter()
    bookmaker_counter = defaultdict(Counter)

    for event_id in unique_event_ids:
        event_path = odds_dir / f"{event_id}.json"
        if not event_path.exists():
            inventory["missing_event_file"] += 1
            continue

        event = load_odds_event(event_path)
        raw_entries = bookmaker_specific_line_entries(event)
        deduped_entries = dedup_player_points_alternate_rows(event)
        inventory["events_seen"] += 1
        inventory["raw_bookmaker_line_entries"] += len(raw_entries)
        inventory["deduped_player_line_entries"] += len(deduped_entries)
        inventory["collapsed_duplicates"] += len(raw_entries) - len(deduped_entries)

        for bookmaker in event.get("bookmakers", []):
            bookmaker_key = str(bookmaker.get("key", "") or "")
            for market in bookmaker.get("markets", []):
                if market.get("key") != "player_points_alternate":
                    continue
                for outcome in market.get("outcomes", []):
                    side = (outcome.get("name") or "").strip().lower()
                    if side in {"over", "under"}:
                        inventory[f"raw_{side}_outcomes"] += 1
                        bookmaker_counter[bookmaker_key][f"raw_{side}_outcomes"] += 1

        for entry in raw_entries:
            bookmaker_key = entry["bookmaker_key"]
            bookmaker_counter[bookmaker_key]["raw_bookmaker_line_entries"] += 1
            if entry["has_over"]:
                bookmaker_counter[bookmaker_key]["raw_lines_with_over"] += 1
            if entry["has_under"]:
                bookmaker_counter[bookmaker_key]["raw_lines_with_under"] += 1
            if entry["has_over"] and entry["has_under"]:
                bookmaker_counter[bookmaker_key]["raw_both_sides"] += 1

        for entry in deduped_entries:
            inventory["deduped_rows_with_over"] += int(entry.get("over_price") is not None)
            inventory["deduped_rows_with_under"] += int(entry.get("under_price") is not None)
            inventory["deduped_both_sides"] += int(
                entry.get("over_price") is not None and entry.get("under_price") is not None
            )
            inventory["deduped_over_only"] += int(
                entry.get("over_price") is not None and entry.get("under_price") is None
            )
            inventory["deduped_under_only"] += int(
                entry.get("over_price") is None and entry.get("under_price") is not None
            )
            bookmaker_counter[entry["bookmaker_key"]]["dedup_wins"] += 1

    bookmaker_rows = [
        {"bookmaker_key": bookmaker_key, **dict(counter)}
        for bookmaker_key, counter in sorted(bookmaker_counter.items())
    ]
    bookmaker_df = pd.DataFrame(bookmaker_rows).fillna(0)
    if not bookmaker_df.empty:
        for column in bookmaker_df.columns:
            if column != "bookmaker_key":
                bookmaker_df[column] = bookmaker_df[column].astype(int)
    return dict(inventory), bookmaker_df


def load_stage_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for column in [
        "bet_price",
        "bet_ev",
        "raw_bet_win_prob",
        "trust_win_prob",
        "bookmaker_count",
        "over_bookmaker_count",
        "under_bookmaker_count",
        "over_price",
        "under_price",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "raw_bet_win_prob" not in df.columns and {
        "bet_side",
        "model_over_prob",
        "model_under_prob",
    }.issubset(df.columns):
        df["raw_bet_win_prob"] = np.where(
            df["bet_side"] == "over",
            pd.to_numeric(df["model_over_prob"], errors="coerce"),
            np.where(
                df["bet_side"] == "under",
                pd.to_numeric(df["model_under_prob"], errors="coerce"),
                np.nan,
            ),
        )
    if "bet_win_target" not in df.columns:
        df["bet_win_target"] = np.where(df["bet_result"] == "win", 1.0, np.where(df["bet_result"] == "loss", 0.0, np.nan))
    if "profit" not in df.columns:
        df["profit"] = profit_from_rows(df)
    df["availability_type"] = df.apply(availability_type, axis=1)
    if "bookmaker_count" in df.columns:
        df["consensus_depth_band"] = df["bookmaker_count"].map(consensus_depth_band)
    else:
        df["consensus_depth_band"] = "missing"
    if "bet_price" in df.columns:
        df["odds_band"] = pd.to_numeric(df["bet_price"], errors="coerce").map(
            lambda value: odds_band(float(value)) if pd.notna(value) else "missing"
        )
    return df


def summarize_groups(df: pd.DataFrame, group_col: str, prob_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col, "rows", "win_rate", "roi", "mean_price", "mean_prob"])
    working = df.copy()
    working["group_key"] = working[group_col].astype(str)
    summary = (
        working.groupby("group_key", dropna=False)
        .agg(
            rows=("group_key", "size"),
            win_rate=("bet_win_target", "mean"),
            roi=("profit", "mean"),
            mean_price=("bet_price", "mean"),
            mean_prob=(prob_col, "mean"),
        )
        .reset_index()
        .rename(columns={"group_key": group_col})
        .sort_values("rows", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def write_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame], inventory: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "source_inventory.json").open("w", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2, sort_keys=True)
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    mapping_rows = load_mapping_rows(Path(args.mapping_csv))
    inventory, source_bookmakers = build_source_inventory(mapping_rows, Path(args.odds_dir))

    stage1 = load_stage_frame(Path(args.stage1_input))
    stage2 = load_stage_frame(Path(args.stage2_input))
    selected = load_stage_frame(Path(args.selected_input))

    stage1_candidates = stage1[stage1["bet_side"].fillna("").ne("") & stage1["profit"].notna()].copy()
    stage1_candidates["probability_band"] = stage1_candidates["raw_bet_win_prob"].map(
        lambda value: probability_band(float(value)) if pd.notna(value) else "missing"
    )
    selected["probability_band"] = selected["trust_win_prob"].map(
        lambda value: probability_band(float(value)) if pd.notna(value) else "missing"
    )

    outputs = {
        "source_bookmaker_summary.csv": source_bookmakers,
        "stage1_candidate_side_summary.csv": summarize_groups(stage1_candidates, "bet_side", "raw_bet_win_prob"),
        "stage1_candidate_probability_band_summary.csv": summarize_groups(stage1_candidates, "probability_band", "raw_bet_win_prob"),
        "stage1_candidate_odds_band_summary.csv": summarize_groups(stage1_candidates, "odds_band", "raw_bet_win_prob"),
        "stage1_candidate_availability_summary.csv": summarize_groups(stage1_candidates, "availability_type", "raw_bet_win_prob"),
        "stage1_candidate_bookmaker_summary.csv": summarize_groups(stage1_candidates, "bookmaker_key", "raw_bet_win_prob"),
        "selected_side_summary.csv": summarize_groups(selected, "bet_side", "trust_win_prob"),
        "selected_probability_band_summary.csv": summarize_groups(selected, "probability_band", "trust_win_prob"),
        "selected_odds_band_summary.csv": summarize_groups(selected, "odds_band", "trust_win_prob"),
        "selected_availability_summary.csv": summarize_groups(selected, "availability_type", "trust_win_prob"),
        "selected_consensus_depth_summary.csv": summarize_groups(selected, "consensus_depth_band", "trust_win_prob"),
        "selected_bookmaker_summary.csv": summarize_groups(selected, "bookmaker_key", "trust_win_prob"),
        "selected_side_by_bookmaker_summary.csv": (
            selected.groupby(["bookmaker_key", "bet_side"], dropna=False)
            .agg(
                rows=("bet_side", "size"),
                win_rate=("bet_win_target", "mean"),
                roi=("profit", "mean"),
                mean_price=("bet_price", "mean"),
                mean_prob=("trust_win_prob", "mean"),
            )
            .reset_index()
            .sort_values(["rows", "bookmaker_key"], ascending=[False, True])
            .reset_index(drop=True)
        ),
        "stage2_candidate_side_summary.csv": summarize_groups(
            stage2[stage2["bet_side"].fillna("").ne("") & stage2["profit"].notna()].copy(),
            "bet_side",
            "trust_win_prob",
        ),
    }
    write_outputs(output_dir, outputs, inventory)

    print(f"Wrote diagnostics to {output_dir}")
    print(json.dumps(inventory, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
