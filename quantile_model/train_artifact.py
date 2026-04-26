#!/usr/bin/env python3
"""Train and save a local points O/U quantile-model artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantile_model.data import build_local_datasets
from quantile_model.service import (
    artifact_summary,
    evaluate_test_set,
    save_artifact,
    train_production_artifact,
    train_split_artifact,
)

DEFAULT_PLAYERS_CSV = REPO_ROOT / "data" / "box_scores" / "players.csv"
DEFAULT_TEAMS_CSV = REPO_ROOT / "data" / "box_scores" / "teams.csv"
DEFAULT_OUTPUT_PRODUCTION_PATH = (
    REPO_ROOT / "quantile_model" / "artifacts" / "points_ou_quantile_artifact_production.pt"
)
DEFAULT_OUTPUT_EVALUATION_PATH = (
    REPO_ROOT / "quantile_model" / "artifacts" / "points_ou_quantile_artifact_split.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local points O/U quantile-model artifact")
    parser.add_argument("--players-csv", type=Path, default=DEFAULT_PLAYERS_CSV)
    parser.add_argument("--teams-csv", type=Path, default=DEFAULT_TEAMS_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PRODUCTION_PATH)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If positive, limit training to the most recent N rows for a faster smoke run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print epoch-by-epoch training progress.",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default="",
        help="If provided, train a split-based artifact using rows on/before this date and evaluate on later rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output == DEFAULT_OUTPUT_PRODUCTION_PATH and args.split_date:
        args.output = DEFAULT_OUTPUT_EVALUATION_PATH
    history_df, _, _ = build_local_datasets(
        players_csv=args.players_csv,
        teams_csv=args.teams_csv,
    )
    if args.max_rows and args.max_rows > 0:
        history_df = (
            history_df.assign(_game_date_sort=pd.to_datetime(history_df["GAME_DATE"], errors="coerce"))
            .sort_values(["_game_date_sort", "GAME_ID", "PLAYER_ID"])
            .tail(args.max_rows)
            .drop(columns="_game_date_sort")
            .reset_index(drop=True)
        )
    progress_callback = None
    if args.verbose:
        def progress_callback(update: dict[str, float | int]) -> None:
            best_val = update.get("best_val_loss")
            best_text = (
                f"{float(best_val):.6f}"
                if best_val is not None and not pd.isna(best_val)
                else "inf"
            )
            print(
                "epoch "
                f"{int(update['epoch'])}: "
                f"train_loss={float(update['train_loss']):.6f} "
                f"val_loss={float(update['val_loss']):.6f} "
                f"best_val_before_epoch={best_text}",
                flush=True,
            )

    if args.split_date:
        artifact = train_split_artifact(
            history_df,
            split_date=args.split_date,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.patience,
            sequence_length=args.sequence_length,
            val_fraction=args.validation_fraction,
            progress_callback=progress_callback,
        )
    else:
        artifact = train_production_artifact(
            history_df,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.patience,
            sequence_length=args.sequence_length,
            val_fraction=args.validation_fraction,
            progress_callback=progress_callback,
        )
    save_artifact(artifact, args.output)
    print(f"Saved artifact to {args.output}")
    for key, value in artifact_summary(artifact).items():
        print(f"{key}: {value}")
    if args.split_date:
        evaluation = evaluate_test_set(history_df, artifact)
        print("test_summary:")
        for key, value in evaluation.summary.items():
            print(f"  {key}: {value}")
        print("quantile_metrics:")
        print(evaluation.quantile_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
