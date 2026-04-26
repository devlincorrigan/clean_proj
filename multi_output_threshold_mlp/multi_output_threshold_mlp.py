#!/usr/bin/env python3
"""Train and score a multi-output threshold MLP for player-points ladders.

This script:
1. Builds a player-game feature matrix using the existing rolling/context stack.
2. Trains a PyTorch MLP to predict tail probabilities P(points >= k | x) jointly.
3. Enforces monotonicity across thresholds with a cumulative minimum pass.
4. Scores the actual sportsbook-offered alternate ladder rows only.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from multi_output_threshold_mlp.odds_processing import (
        BOOKMAKER_PRIORITY,
        bookmaker_title,
        dedup_player_points_alternate_rows,
        line_key,
        load_odds_event,
        required_threshold_from_line,
    )
except ImportError:
    from odds_processing import (
        BOOKMAKER_PRIORITY,
        bookmaker_title,
        dedup_player_points_alternate_rows,
        line_key,
        load_odds_event,
        required_threshold_from_line,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}
MANUAL_NAME_ALIASES = {
    "carlton carrington": ["bub carrington"],
}

FEATURE_COLS = [
    "rolling_mean_points",
    "player_points_median_roll",
    "player_points_std_roll",
    "player_minutes_roll",
    "player_fga_roll",
    "player_3pa_roll",
    "player_fta_roll",
    "player_rebounds_roll",
    "player_assists_roll",
    "player_usage_roll",
    "player_points_recent_roll",
    "player_minutes_recent_roll",
    "player_fga_recent_roll",
    "player_3pa_recent_roll",
    "player_fta_recent_roll",
    "player_usage_recent_roll",
    "player_points_trend",
    "player_minutes_trend",
    "player_fga_trend",
    "player_3pa_trend",
    "player_fta_trend",
    "player_usage_trend",
    "player_share_team_points_roll",
    "player_share_team_fga_roll",
    "is_home_int",
    "team_pace_roll",
    "opp_points_roll",
    "opp_pace_roll",
    "opp_net_rating_roll",
    "opp_defensive_rating_roll",
]

DEFAULT_CONFIG = {
    "mapping_csv": str(PROJECT_ROOT / "data/game_event_bijection.csv"),
    "players_csv": str(PROJECT_ROOT / "data/box_scores/players.csv"),
    "teams_csv": str(PROJECT_ROOT / "data/box_scores/teams.csv"),
    "odds_dir": str(PROJECT_ROOT / "data/historical_odds_api" / "historical_player_points_alternate"),
    "output": str(SCRIPT_DIR / "multi_output_threshold_mlp_predictions.csv"),
    "artifact_output": str(ARTIFACTS_DIR / "threshold_mlp_artifact.pt"),
    "player_window": 60,
    "recent_window": 5,
    "team_window": 40,
    "min_threshold": 1,
    "max_threshold": 65,
    "train_frac": 0.70,
    "val_frac": 0.15,
    "test_frac": 0.15,
    "hidden_dim": 192,
    "num_layers": 3,
    "dropout": 0.15,
    "batch_size": 1024,
    "epochs": 80,
    "lr": 3e-4,
    "weight_decay": 5e-4,
    "patience": 10,
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "min_lr": 1e-5,
    "grad_clip_norm": 1.0,
    "threshold_weighting": "none",
    "threshold_weight_power": 0.5,
    "use_pos_weight": 0,
    "selection_metric": "sportsbook_log_loss",
    "seed": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a multi-output threshold MLP for player-points ladders and "
            "score actual sportsbook alternate lines."
        )
    )
    parser.add_argument("--mapping-csv", default=DEFAULT_CONFIG["mapping_csv"])
    parser.add_argument("--players-csv", default=DEFAULT_CONFIG["players_csv"])
    parser.add_argument("--teams-csv", default=DEFAULT_CONFIG["teams_csv"])
    parser.add_argument("--odds-dir", default=DEFAULT_CONFIG["odds_dir"])
    parser.add_argument("--output", default=DEFAULT_CONFIG["output"])
    parser.add_argument("--artifact-output", default=DEFAULT_CONFIG["artifact_output"])
    parser.add_argument("--player-window", type=int, default=DEFAULT_CONFIG["player_window"])
    parser.add_argument("--recent-window", type=int, default=DEFAULT_CONFIG["recent_window"])
    parser.add_argument("--team-window", type=int, default=DEFAULT_CONFIG["team_window"])
    parser.add_argument("--min-threshold", type=int, default=DEFAULT_CONFIG["min_threshold"])
    parser.add_argument("--max-threshold", type=int, default=DEFAULT_CONFIG["max_threshold"])
    parser.add_argument("--train-frac", type=float, default=DEFAULT_CONFIG["train_frac"])
    parser.add_argument("--val-frac", type=float, default=DEFAULT_CONFIG["val_frac"])
    parser.add_argument("--test-frac", type=float, default=DEFAULT_CONFIG["test_frac"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG["num_layers"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=DEFAULT_CONFIG["scheduler_factor"],
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=DEFAULT_CONFIG["scheduler_patience"],
    )
    parser.add_argument("--min-lr", type=float, default=DEFAULT_CONFIG["min_lr"])
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=DEFAULT_CONFIG["grad_clip_norm"],
    )
    parser.add_argument(
        "--threshold-weighting",
        choices=["none", "sportsbook_frequency"],
        default=DEFAULT_CONFIG["threshold_weighting"],
        help="How to weight threshold losses during training.",
    )
    parser.add_argument(
        "--threshold-weight-power",
        type=float,
        default=DEFAULT_CONFIG["threshold_weight_power"],
        help="Exponent applied to sportsbook threshold frequencies before normalization.",
    )
    parser.add_argument(
        "--use-pos-weight",
        type=int,
        default=DEFAULT_CONFIG["use_pos_weight"],
        help="Whether to apply per-threshold positive-class reweighting (1=yes, 0=no).",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["threshold_bce", "sportsbook_brier", "sportsbook_log_loss"],
        default=DEFAULT_CONFIG["selection_metric"],
        help="Validation metric used for checkpoint selection and early stopping.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def canonical_game_id(game_id: str) -> str:
    return str(int(game_id))


def infer_season(game_date) -> str:
    start_year = game_date.year if game_date.month >= 7 else game_date.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def parse_minutes(value) -> float:
    if pd.isna(value) or value in (None, ""):
        return np.nan
    text = str(value)
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) + float(seconds) / 60.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def normalize_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(
        char for char in ascii_text if not unicodedata.combining(char)
    )
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return " ".join(ascii_text.split())


def name_tokens(name: str) -> tuple[list[str], list[str]]:
    tokens = normalize_text(name).split()
    if tokens and tokens[-1] in SUFFIX_TOKENS:
        core_tokens = tokens[:-1]
    else:
        core_tokens = tokens
    return tokens, core_tokens


def alias_keys(name: str) -> list[str]:
    tokens, core_tokens = name_tokens(name)
    aliases = []

    if tokens:
        aliases.append(" ".join(tokens))
    if core_tokens and core_tokens != tokens:
        aliases.append(" ".join(core_tokens))
    if len(core_tokens) >= 2:
        family_name = " ".join(core_tokens[1:])
        aliases.append(f"{core_tokens[0][0]} {family_name}")

    initials = []
    index = 0
    while index < len(core_tokens) - 1 and len(core_tokens[index]) == 1:
        initials.append(core_tokens[index])
        index += 1
    if len(initials) >= 2:
        aliases.append(f"{''.join(initials)} {' '.join(core_tokens[index:])}")

    seen = set()
    ordered = []
    for alias in aliases:
        if alias and alias not in seen:
            ordered.append(alias)
            seen.add(alias)

    for manual_alias in MANUAL_NAME_ALIASES.get(normalize_text(name), []):
        if manual_alias not in seen:
            ordered.append(manual_alias)
            seen.add(manual_alias)

    return ordered


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value) -> int:
    return int(float(value))


def add_grouped_shifted_rolling_mean(
    frame: pd.DataFrame,
    group_col: str,
    source_col: str,
    target_col: str,
    window: int,
) -> pd.DataFrame:
    frame[target_col] = frame.groupby(group_col)[source_col].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    return frame


def add_grouped_shifted_rolling_std(
    frame: pd.DataFrame,
    group_col: str,
    source_col: str,
    target_col: str,
    window: int,
) -> pd.DataFrame:
    frame[target_col] = frame.groupby(group_col)[source_col].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).std(ddof=0)
    )
    return frame


def load_players(players_csv: str) -> pd.DataFrame:
    players = pd.read_csv(
        players_csv,
        dtype={"gameId": str, "teamId": str, "personId": str},
    )
    players["gameId"] = players["gameId"].map(canonical_game_id)
    players["gameDate"] = pd.to_datetime(players["gameDate"])
    players["player_name_box_score"] = (
        players["firstName"].fillna("").str.strip()
        + " "
        + players["familyName"].fillna("").str.strip()
    ).str.strip()
    players["minutes_float"] = (
        players["minutesFloat"].map(parse_minutes)
        if "minutesFloat" in players.columns
        else players["minutes"].map(parse_minutes)
    )
    players["is_home_int"] = players["isHome"].astype(int)

    numeric_columns = [
        "points",
        "fieldGoalsAttempted",
        "threePointersAttempted",
        "freeThrowsAttempted",
        "reboundsTotal",
        "assists",
        "usagePercentage",
    ]
    for column in numeric_columns:
        players[column] = pd.to_numeric(players[column], errors="coerce")

    players = players[players["minutes_float"] >= 1.0].copy()
    players = players.sort_values(["personId", "gameDate", "gameId"]).copy()
    return players


def load_teams(teams_csv: str) -> pd.DataFrame:
    teams = pd.read_csv(
        teams_csv,
        dtype={"gameId": str, "teamId": str},
    )
    teams["gameId"] = teams["gameId"].map(canonical_game_id)
    teams["gameDate"] = pd.to_datetime(teams["gameDate"])

    numeric_columns = [
        "points",
        "fieldGoalsAttempted",
        "pace",
        "netRating",
        "defensiveRating",
    ]
    for column in numeric_columns:
        teams[column] = pd.to_numeric(teams[column], errors="coerce")

    teams = teams.sort_values(["teamId", "gameDate", "gameId"]).copy()
    return teams


def build_roster_indexes(players: pd.DataFrame):
    roster_index = defaultdict(lambda: defaultdict(list))
    roster_rows = defaultdict(list)
    box_rows_by_key = {}

    for row in players.to_dict("records"):
        game_id = row["gameId"]
        person_id = row["personId"]
        roster_rows[game_id].append(row)
        box_rows_by_key[(game_id, person_id)] = row

        aliases = set(alias_keys(row["player_name_box_score"]))
        if row.get("playerSlug"):
            aliases.update(alias_keys(str(row["playerSlug"]).replace("-", " ")))

        for alias in aliases:
            roster_index[game_id][alias].append(row)

    return roster_index, roster_rows, box_rows_by_key


def unique_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    person_ids = {row["personId"] for row in rows}
    if len(person_ids) == 1:
        return rows[0]
    return None


def family_name_from_core(core_tokens: list[str]) -> str:
    if len(core_tokens) < 2:
        return ""
    return " ".join(core_tokens[1:])


def match_current_player(
    game_id: str,
    odds_player_name: str,
    roster_rows,
    roster_index,
):
    canonical_id = canonical_game_id(game_id)
    for alias in alias_keys(odds_player_name):
        match = unique_row(roster_index[canonical_id].get(alias, []))
        if match is not None:
            return match

    _, core_tokens = name_tokens(odds_player_name)
    family_name = family_name_from_core(core_tokens)
    if not family_name:
        return None

    first_initial = core_tokens[0][0]
    candidates = []
    for row in roster_rows[canonical_id]:
        _, record_core = name_tokens(row["player_name_box_score"])
        if not record_core:
            continue
        if family_name_from_core(record_core) != family_name:
            continue
        if record_core[0][0] != first_initial:
            continue
        candidates.append(row)

    return unique_row(candidates)


def build_player_feature_rows(
    players: pd.DataFrame,
    player_window: int,
    recent_window: int,
) -> pd.DataFrame:
    working = players.copy()
    working["season"] = working["gameDate"].map(infer_season)

    working = add_grouped_shifted_rolling_mean(
        working,
        group_col="personId",
        source_col="points",
        target_col="rolling_mean_points",
        window=player_window,
    )
    working["player_points_median_roll"] = working.groupby("personId")["points"].transform(
        lambda series: series.shift(1).rolling(player_window, min_periods=player_window).median()
    )
    working = add_grouped_shifted_rolling_std(
        working,
        group_col="personId",
        source_col="points",
        target_col="player_points_std_roll",
        window=player_window,
    )

    long_sources = {
        "player_minutes_roll": "minutes_float",
        "player_fga_roll": "fieldGoalsAttempted",
        "player_3pa_roll": "threePointersAttempted",
        "player_fta_roll": "freeThrowsAttempted",
        "player_rebounds_roll": "reboundsTotal",
        "player_assists_roll": "assists",
        "player_usage_roll": "usagePercentage",
    }
    for target_col, source_col in long_sources.items():
        working = add_grouped_shifted_rolling_mean(
            working,
            group_col="personId",
            source_col=source_col,
            target_col=target_col,
            window=player_window,
        )

    recent_sources = {
        "player_points_recent_roll": "points",
        "player_minutes_recent_roll": "minutes_float",
        "player_fga_recent_roll": "fieldGoalsAttempted",
        "player_3pa_recent_roll": "threePointersAttempted",
        "player_fta_recent_roll": "freeThrowsAttempted",
        "player_usage_recent_roll": "usagePercentage",
    }
    for target_col, source_col in recent_sources.items():
        working = add_grouped_shifted_rolling_mean(
            working,
            group_col="personId",
            source_col=source_col,
            target_col=target_col,
            window=recent_window,
        )

    return working[
        [
            "gameId",
            "gameDate",
            "season",
            "personId",
            "teamId",
            "player_name_box_score",
            "points",
            "is_home_int",
            "rolling_mean_points",
            "player_points_median_roll",
            "player_points_std_roll",
            "player_minutes_roll",
            "player_fga_roll",
            "player_3pa_roll",
            "player_fta_roll",
            "player_rebounds_roll",
            "player_assists_roll",
            "player_usage_roll",
            "player_points_recent_roll",
            "player_minutes_recent_roll",
            "player_fga_recent_roll",
            "player_3pa_recent_roll",
            "player_fta_recent_roll",
            "player_usage_recent_roll",
        ]
    ].rename(
        columns={
            "gameId": "game_id",
            "gameDate": "game_date",
            "personId": "person_id",
            "teamId": "team_id",
            "points": "actual_points",
        }
    )


def build_team_feature_lookups(teams: pd.DataFrame, team_window: int):
    working = teams.copy()
    rolling_specs = {
        "team_points_roll": "points",
        "team_fga_roll": "fieldGoalsAttempted",
        "team_pace_roll": "pace",
        "opp_points_roll": "points",
        "opp_pace_roll": "pace",
        "opp_net_rating_roll": "netRating",
        "opp_defensive_rating_roll": "defensiveRating",
    }
    for target_col, source_col in rolling_specs.items():
        working = add_grouped_shifted_rolling_mean(
            working,
            group_col="teamId",
            source_col=source_col,
            target_col=target_col,
            window=team_window,
        )

    own_lookup = working[
        [
            "gameId",
            "teamId",
            "team_points_roll",
            "team_fga_roll",
            "team_pace_roll",
        ]
    ].rename(columns={"gameId": "game_id", "teamId": "team_id"})

    opponent_lookup = working[
        [
            "gameId",
            "teamId",
            "opp_points_roll",
            "opp_pace_roll",
            "opp_net_rating_roll",
            "opp_defensive_rating_roll",
        ]
    ].rename(columns={"gameId": "game_id", "teamId": "opponent_team_id"})

    game_team_ids = working[["gameId", "teamId"]].rename(
        columns={"gameId": "game_id", "teamId": "opponent_team_id"}
    )

    return own_lookup, opponent_lookup, game_team_ids


def enrich_with_team_features(
    merged: pd.DataFrame,
    own_lookup: pd.DataFrame,
    opponent_lookup: pd.DataFrame,
    game_team_ids: pd.DataFrame,
) -> pd.DataFrame:
    enriched = merged.merge(own_lookup, on=["game_id", "team_id"], how="left")
    enriched = enriched.merge(game_team_ids, on="game_id", how="left")
    enriched = enriched[enriched["team_id"] != enriched["opponent_team_id"]].copy()
    enriched = enriched.merge(opponent_lookup, on=["game_id", "opponent_team_id"], how="left")

    enriched["player_share_team_points_roll"] = (
        enriched["rolling_mean_points"] / enriched["team_points_roll"]
    )
    enriched["player_share_team_fga_roll"] = (
        enriched["player_fga_roll"] / enriched["team_fga_roll"]
    )

    enriched["player_points_trend"] = (
        enriched["player_points_recent_roll"] - enriched["rolling_mean_points"]
    )
    enriched["player_minutes_trend"] = (
        enriched["player_minutes_recent_roll"] - enriched["player_minutes_roll"]
    )
    enriched["player_fga_trend"] = (
        enriched["player_fga_recent_roll"] - enriched["player_fga_roll"]
    )
    enriched["player_3pa_trend"] = (
        enriched["player_3pa_recent_roll"] - enriched["player_3pa_roll"]
    )
    enriched["player_fta_trend"] = (
        enriched["player_fta_recent_roll"] - enriched["player_fta_roll"]
    )
    enriched["player_usage_trend"] = (
        enriched["player_usage_recent_roll"] - enriched["player_usage_roll"]
    )

    for column in [
        "player_share_team_points_roll",
        "player_share_team_fga_roll",
    ]:
        enriched.loc[~np.isfinite(enriched[column]), column] = np.nan

    return enriched


def build_feature_dataset(
    players_csv: str,
    teams_csv: str,
    player_window: int,
    recent_window: int,
    team_window: int,
):
    players = load_players(players_csv)
    teams = load_teams(teams_csv)
    roster_index, roster_rows, box_rows_by_key = build_roster_indexes(players)

    player_features = build_player_feature_rows(
        players,
        player_window=player_window,
        recent_window=recent_window,
    )
    own_lookup, opponent_lookup, game_team_ids = build_team_feature_lookups(
        teams,
        team_window=team_window,
    )
    dataset = enrich_with_team_features(
        player_features,
        own_lookup=own_lookup,
        opponent_lookup=opponent_lookup,
        game_team_ids=game_team_ids,
    )
    dataset = dataset.sort_values(["game_date", "game_id", "person_id"]).reset_index(drop=True)
    return dataset, roster_index, roster_rows, box_rows_by_key


def chronological_split_by_date(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> pd.DataFrame:
    total = train_frac + val_frac + test_frac
    if total <= 0:
        raise ValueError("train/val/test fractions must sum to a positive value.")

    train_frac /= total
    val_frac /= total
    test_frac /= total

    unique_dates = np.array(sorted(df["game_date"].drop_duplicates()))
    if len(unique_dates) < 3:
        raise ValueError("Need at least 3 unique game dates to form train/val/test splits.")

    train_end = int(len(unique_dates) * train_frac)
    val_end = train_end + int(len(unique_dates) * val_frac)

    train_end = max(1, min(train_end, len(unique_dates) - 2))
    val_end = max(train_end + 1, min(val_end, len(unique_dates) - 1))

    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates = set(unique_dates[val_end:])

    split = np.where(
        df["game_date"].isin(train_dates),
        "train",
        np.where(df["game_date"].isin(val_dates), "val", "test"),
    )
    split_df = df.copy()
    split_df["dataset_split"] = split
    return split_df


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


def fit_standardizer(values: np.ndarray) -> Standardizer:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0.0] = 1.0
    return Standardizer(mean=mean, std=std)


class ThresholdMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def build_threshold_targets(actual_points: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (actual_points[:, None] >= thresholds[None, :]).astype(np.float32)


def make_loader(features: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool):
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def monotone_tail_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(probabilities, axis=1)


def build_prediction_lookup_from_frame(frame: pd.DataFrame, probs: np.ndarray) -> dict:
    lookup = {}
    for row, tail_probs in zip(frame.to_dict("records"), probs, strict=True):
        lookup[(row["game_id"], row["person_id"])] = {
            "game_date": pd.Timestamp(row["game_date"]).strftime("%Y-%m-%d"),
            "season": row["season"],
            "dataset_split": row["dataset_split"],
            "team_id": row["team_id"],
            "player_name_box_score": row["player_name_box_score"],
            "actual_points": float(row["actual_points"]),
            "tail_probabilities": tail_probs,
        }
    return lookup


def sportsbook_probability_metrics(rows: list[dict]) -> dict | None:
    probabilities = []
    actuals = []
    for row in rows:
        if row["actual_result"] == "push":
            continue
        probability = safe_float(row["model_over_prob"])
        if probability is None:
            continue
        probabilities.append(float(probability))
        actuals.append(1.0 if row["actual_result"] == "over" else 0.0)

    if not probabilities:
        return None

    probs = np.asarray(probabilities, dtype=np.float64)
    actual = np.asarray(actuals, dtype=np.float64)
    clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return {
        "rows": int(len(probs)),
        "brier": float(np.mean((probs - actual) ** 2)),
        "log_loss": float(
            -np.mean(actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped))
        ),
    }


def train_threshold_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_frame: pd.DataFrame,
    val_mapping_rows: list[dict],
    odds_dir: str,
    roster_rows,
    roster_index,
    thresholds: np.ndarray,
    threshold_weights: np.ndarray,
    pos_weights: np.ndarray | None,
    args: argparse.Namespace,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True)
    val_loader = make_loader(val_x, val_y, args.batch_size, shuffle=False)

    model = ThresholdMLP(
        input_dim=train_x.shape[1],
        output_dim=train_y.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    threshold_weight_tensor = torch.tensor(
        threshold_weights,
        dtype=torch.float32,
        device=device,
    )
    pos_weight_tensor = None
    if pos_weights is not None:
        pos_weight_tensor = torch.tensor(
            pos_weights,
            dtype=torch.float32,
            device=device,
        )
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor)

    best_state = None
    best_threshold_bce = float("inf")
    best_selection_metric = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_metric_name = args.selection_metric
    selection_metric_history = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss_matrix = criterion(logits, batch_y)
            loss = (loss_matrix * threshold_weight_tensor.unsqueeze(0)).mean()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        val_prob_batches = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss_matrix = criterion(logits, batch_y)
                loss = (loss_matrix * threshold_weight_tensor.unsqueeze(0)).mean()
                val_losses.append(float(loss.item()))
                val_prob_batches.append(torch.sigmoid(logits).cpu().numpy())

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        val_probs = monotone_tail_probabilities(np.concatenate(val_prob_batches, axis=0))

        selection_metric_value = mean_val_loss
        selection_metric_label = "val_bce"
        sportsbook_metrics = None
        if args.selection_metric != "threshold_bce":
            val_prediction_lookup = build_prediction_lookup_from_frame(val_frame, val_probs)
            val_prediction_rows, _ = build_prediction_rows(
                mapping_rows=val_mapping_rows,
                odds_dir=odds_dir,
                prediction_lookup=val_prediction_lookup,
                roster_rows=roster_rows,
                roster_index=roster_index,
                thresholds=thresholds,
            )
            sportsbook_metrics = sportsbook_probability_metrics(val_prediction_rows)
            if sportsbook_metrics is not None:
                if args.selection_metric == "sportsbook_brier":
                    selection_metric_value = sportsbook_metrics["brier"]
                    selection_metric_label = "val_sportsbook_brier"
                elif args.selection_metric == "sportsbook_log_loss":
                    selection_metric_value = sportsbook_metrics["log_loss"]
                    selection_metric_label = "val_sportsbook_log_loss"

        scheduler.step(selection_metric_value)
        current_lr = optimizer.param_groups[0]["lr"]
        log_parts = [
            f"epoch {epoch:02d}",
            f"train_bce={mean_train_loss:.5f}",
            f"val_bce={mean_val_loss:.5f}",
            f"{selection_metric_label}={selection_metric_value:.5f}",
        ]
        if sportsbook_metrics is not None:
            log_parts.append(f"val_books_rows={sportsbook_metrics['rows']}")
            log_parts.append(f"val_books_brier={sportsbook_metrics['brier']:.5f}")
            log_parts.append(f"val_books_logloss={sportsbook_metrics['log_loss']:.5f}")
        log_parts.append(f"lr={current_lr:.6g}")
        print(" ".join(log_parts))

        if selection_metric_value < best_selection_metric:
            best_selection_metric = selection_metric_value
            best_threshold_bce = mean_val_loss
            best_epoch = epoch
            stale_epochs = 0
            selection_metric_history = sportsbook_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(
                    f"early stopping after epoch {epoch} "
                    f"(best epoch {best_epoch}, best {selection_metric_label} {best_selection_metric:.5f})"
                )
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    return model.cpu(), {
        "best_epoch": best_epoch,
        "best_threshold_bce": best_threshold_bce,
        "best_selection_metric": best_selection_metric,
        "selection_metric_name": best_metric_name,
        "best_sportsbook_metrics": selection_metric_history,
    }


def predict_tail_probabilities(model: ThresholdMLP, values: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(values, dtype=torch.float32, device=device)
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    return monotone_tail_probabilities(probs)


def load_mapping_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_event(path: Path) -> dict:
    return load_odds_event(path)


def player_ladder_rows(event: dict) -> list[dict]:
    return dedup_player_points_alternate_rows(event)


def compute_sportsbook_threshold_weights(
    mapping_rows: list[dict],
    odds_dir: str,
    train_game_ids: set[str],
    thresholds: np.ndarray,
    power: float,
) -> np.ndarray:
    counts = np.zeros(len(thresholds), dtype=np.float64)
    threshold_min = int(thresholds.min())
    threshold_max = int(thresholds.max())
    odds_path = Path(odds_dir)

    for mapping_row in mapping_rows:
        game_id = canonical_game_id(mapping_row["game_id"])
        if game_id not in train_game_ids:
            continue

        event_path = odds_path / f"{mapping_row['event_id']}.json"
        if not event_path.exists():
            continue

        event = load_event(event_path)
        for ladder_row in player_ladder_rows(event):
            required_threshold = required_threshold_from_line(ladder_row["line_points"])
            if threshold_min <= required_threshold <= threshold_max:
                counts[required_threshold - threshold_min] += 1.0

    if counts.sum() <= 0:
        return np.ones(len(thresholds), dtype=np.float32)

    smoothed = np.power(np.maximum(counts, 1.0), max(power, 0.0))
    normalized = smoothed / smoothed.mean()
    return normalized.astype(np.float32)


def compute_positive_class_weights(targets: np.ndarray) -> np.ndarray:
    positives = targets.sum(axis=0, dtype=np.float64)
    negatives = targets.shape[0] - positives
    pos_weight = np.ones(targets.shape[1], dtype=np.float32)

    valid = positives > 0
    pos_weight[valid] = np.clip(negatives[valid] / positives[valid], 1.0, 50.0)
    return pos_weight


def summarize_threshold_weights(
    thresholds: np.ndarray,
    threshold_weights: np.ndarray,
    pos_weights: np.ndarray | None,
) -> list[str]:
    focus_thresholds = sorted(
        set(
            int(value)
            for value in np.linspace(int(thresholds.min()), int(thresholds.max()), num=7)
        )
    )
    lines = ["threshold loss weighting summary:"]
    for threshold in focus_thresholds:
        index = threshold - int(thresholds.min())
        pieces = [f"k={threshold}", f"threshold_weight={threshold_weights[index]:.3f}"]
        if pos_weights is not None:
            pieces.append(f"pos_weight={float(pos_weights[index]):.3f}")
        lines.append("  " + ", ".join(pieces))
    return lines


def build_prediction_rows(
    mapping_rows: list[dict],
    odds_dir: str,
    prediction_lookup: dict,
    roster_rows,
    roster_index,
    thresholds: np.ndarray,
) -> tuple[list[dict], Counter]:
    odds_path = Path(odds_dir)
    stats = Counter()
    threshold_min = int(thresholds.min())
    threshold_max = int(thresholds.max())

    rows = []
    for mapping_row in mapping_rows:
        game_id = canonical_game_id(mapping_row["game_id"])
        event_id = mapping_row["event_id"]
        event_path = odds_path / f"{event_id}.json"
        if not event_path.exists():
            stats["missing_event_file"] += 1
            continue

        event = load_event(event_path)
        ladder_rows = player_ladder_rows(event)
        stats["ladder_rows_seen"] += len(ladder_rows)

        for ladder_row in ladder_rows:
            matched = match_current_player(
                game_id=game_id,
                odds_player_name=ladder_row["player_name_odds"],
                roster_rows=roster_rows,
                roster_index=roster_index,
            )
            if matched is None:
                stats["unmatched_player_rows"] += 1
                continue

            person_id = matched["personId"]
            prediction_row = prediction_lookup.get((game_id, person_id))
            if prediction_row is None:
                stats["missing_prediction_rows"] += 1
                continue

            required_threshold = required_threshold_from_line(ladder_row["line_points"])
            if required_threshold < threshold_min or required_threshold > threshold_max:
                stats["out_of_range_threshold_rows"] += 1
                continue

            over_prob = float(
                prediction_row["tail_probabilities"][required_threshold - threshold_min]
            )
            under_prob = 1.0 - over_prob

            over_price = safe_float(ladder_row["over_price"])
            under_price = safe_float(ladder_row["under_price"])
            over_ev = over_prob * over_price - 1.0 if over_price is not None else None
            under_ev = under_prob * under_price - 1.0 if under_price is not None else None

            bet_side = ""
            bet_price = None
            bet_ev = None
            if over_ev is not None and over_ev > 0.0:
                bet_side = "over"
                bet_price = over_price
                bet_ev = over_ev
            if under_ev is not None and under_ev > 0.0:
                if bet_ev is None or under_ev > bet_ev:
                    bet_side = "under"
                    bet_price = under_price
                    bet_ev = under_ev

            actual_points = float(prediction_row["actual_points"])
            line_points = float(ladder_row["line_points"])
            if actual_points > line_points:
                actual_result = "over"
            elif actual_points < line_points:
                actual_result = "under"
            else:
                actual_result = "push"

            if bet_side == "":
                bet_result = ""
            elif actual_result == "push":
                bet_result = "push"
            elif bet_side == actual_result:
                bet_result = "win"
            else:
                bet_result = "loss"

            chosen_bookmaker_key = ladder_row["bookmaker_key"]
            chosen_bookmaker_title = ladder_row["bookmaker_title"]
            chosen_bookmaker_last_update = ladder_row["bookmaker_last_update"]
            chosen_market_last_update = ladder_row["market_last_update"]
            if bet_side in {"over", "under"}:
                chosen_bookmaker_key = (
                    ladder_row.get(f"{bet_side}_bookmaker_key") or chosen_bookmaker_key
                )
                chosen_bookmaker_title = (
                    ladder_row.get(f"{bet_side}_bookmaker_title") or chosen_bookmaker_title
                )
                chosen_bookmaker_last_update = (
                    ladder_row.get(f"{bet_side}_bookmaker_last_update")
                    or chosen_bookmaker_last_update
                )
                chosen_market_last_update = (
                    ladder_row.get(f"{bet_side}_market_last_update")
                    or chosen_market_last_update
                )

            rows.append(
                {
                    "game_id": game_id,
                    "event_id": event_id,
                    "game_date": prediction_row["game_date"],
                    "season": prediction_row["season"],
                    "dataset_split": prediction_row["dataset_split"],
                    "person_id": person_id,
                    "team_id": prediction_row["team_id"],
                    "player_name_odds": ladder_row["player_name_odds"],
                    "player_name_box_score": prediction_row["player_name_box_score"],
                    "bookmaker_key": chosen_bookmaker_key,
                    "bookmaker_title": chosen_bookmaker_title,
                    "bookmaker_last_update": chosen_bookmaker_last_update,
                    "market_last_update": chosen_market_last_update,
                    "over_bookmaker_key": ladder_row.get("over_bookmaker_key", ""),
                    "over_bookmaker_title": ladder_row.get("over_bookmaker_title", ""),
                    "under_bookmaker_key": ladder_row.get("under_bookmaker_key", ""),
                    "under_bookmaker_title": ladder_row.get("under_bookmaker_title", ""),
                    "bookmaker_count": ladder_row.get("bookmaker_count", 0),
                    "over_bookmaker_count": ladder_row.get("over_bookmaker_count", 0),
                    "under_bookmaker_count": ladder_row.get("under_bookmaker_count", 0),
                    "line_points": line_points,
                    "line_key": line_key(line_points),
                    "required_threshold": required_threshold,
                    "over_price": over_price,
                    "under_price": under_price,
                    "model_over_prob": over_prob,
                    "model_under_prob": under_prob,
                    "over_ev": over_ev,
                    "under_ev": under_ev,
                    "bet_side": bet_side,
                    "bet_price": bet_price,
                    "bet_ev": bet_ev,
                    "actual_points": actual_points,
                    "actual_result": actual_result,
                    "bet_result": bet_result,
                }
            )
            stats["scored_rows"] += 1

    return rows, stats


def save_artifact(
    path: str,
    model: ThresholdMLP,
    standardizer: Standardizer,
    thresholds: np.ndarray,
    threshold_weights: np.ndarray,
    pos_weights: np.ndarray | None,
    args: argparse.Namespace,
    training_summary: dict,
    train_rows: int,
    val_rows: int,
    test_rows: int,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "feature_cols": FEATURE_COLS,
        "thresholds": thresholds.tolist(),
        "standardizer_mean": standardizer.mean.astype(np.float32),
        "standardizer_std": standardizer.std.astype(np.float32),
        "threshold_weights": threshold_weights.astype(np.float32),
        "pos_weights": None if pos_weights is None else pos_weights.astype(np.float32),
        "config": vars(args),
        "best_val_bce": float(training_summary["best_threshold_bce"]),
        "best_epoch": int(training_summary["best_epoch"]),
        "best_selection_metric": float(training_summary["best_selection_metric"]),
        "selection_metric_name": training_summary["selection_metric_name"],
        "best_sportsbook_metrics": training_summary["best_sportsbook_metrics"],
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "test_rows": int(test_rows),
    }
    torch.save(payload, path)


def write_prediction_rows(path: str, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "game_id",
        "event_id",
        "game_date",
        "season",
        "dataset_split",
        "person_id",
        "team_id",
        "player_name_odds",
        "player_name_box_score",
        "bookmaker_key",
        "bookmaker_title",
        "bookmaker_last_update",
        "market_last_update",
        "over_bookmaker_key",
        "over_bookmaker_title",
        "under_bookmaker_key",
        "under_bookmaker_title",
        "bookmaker_count",
        "over_bookmaker_count",
        "under_bookmaker_count",
        "line_points",
        "line_key",
        "required_threshold",
        "over_price",
        "under_price",
        "model_over_prob",
        "model_under_prob",
        "over_ev",
        "under_ev",
        "bet_side",
        "bet_price",
        "bet_ev",
        "actual_points",
        "actual_result",
        "bet_result",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.min_threshold > args.max_threshold:
        raise ValueError("--min-threshold must be <= --max-threshold.")

    set_seed(args.seed)
    thresholds = np.arange(args.min_threshold, args.max_threshold + 1, dtype=np.int32)

    dataset, roster_index, roster_rows, _ = build_feature_dataset(
        players_csv=args.players_csv,
        teams_csv=args.teams_csv,
        player_window=args.player_window,
        recent_window=args.recent_window,
        team_window=args.team_window,
    )
    dataset = chronological_split_by_date(
        dataset,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )

    usable = dataset.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)
    usable["row_id"] = np.arange(len(usable))
    print(f"all player-game rows: {len(dataset)}")
    print(f"usable player-game rows: {len(usable)}")

    train_df = usable[usable["dataset_split"] == "train"].copy()
    val_df = usable[usable["dataset_split"] == "val"].copy()
    test_df = usable[usable["dataset_split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One of the chronological splits is empty after dropping missing features.")

    train_x = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    val_x = val_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    test_x = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)

    train_y = build_threshold_targets(
        train_df["actual_points"].to_numpy(dtype=np.float32),
        thresholds=thresholds,
    )
    val_y = build_threshold_targets(
        val_df["actual_points"].to_numpy(dtype=np.float32),
        thresholds=thresholds,
    )
    test_y = build_threshold_targets(
        test_df["actual_points"].to_numpy(dtype=np.float32),
        thresholds=thresholds,
    )

    standardizer = fit_standardizer(train_x)
    train_x_std = standardizer.transform(train_x)
    val_x_std = standardizer.transform(val_x)
    test_x_std = standardizer.transform(test_x)

    mapping_rows = load_mapping_rows(args.mapping_csv)
    val_mapping_rows = [
        row for row in mapping_rows if canonical_game_id(row["game_id"]) in set(val_df["game_id"])
    ]
    if args.threshold_weighting == "sportsbook_frequency":
        threshold_weights = compute_sportsbook_threshold_weights(
            mapping_rows=mapping_rows,
            odds_dir=args.odds_dir,
            train_game_ids=set(train_df["game_id"]),
            thresholds=thresholds,
            power=args.threshold_weight_power,
        )
    else:
        threshold_weights = np.ones(len(thresholds), dtype=np.float32)

    pos_weights = compute_positive_class_weights(train_y) if args.use_pos_weight else None
    for line in summarize_threshold_weights(thresholds, threshold_weights, pos_weights):
        print(line)

    model, training_summary = train_threshold_model(
        train_x=train_x_std,
        train_y=train_y,
        val_x=val_x_std,
        val_y=val_y,
        val_frame=val_df,
        val_mapping_rows=val_mapping_rows,
        odds_dir=args.odds_dir,
        roster_rows=roster_rows,
        roster_index=roster_index,
        thresholds=thresholds,
        threshold_weights=threshold_weights,
        pos_weights=pos_weights,
        args=args,
    )

    val_probs = predict_tail_probabilities(model, val_x_std)
    test_probs = predict_tail_probabilities(model, test_x_std)
    print(
        "best checkpoint: "
        f"epoch={training_summary['best_epoch']}, "
        f"selection_metric={training_summary['selection_metric_name']}, "
        f"value={training_summary['best_selection_metric']:.5f}, "
        f"val_bce={training_summary['best_threshold_bce']:.5f}"
    )
    if training_summary["best_sportsbook_metrics"] is not None:
        print(
            "best checkpoint sportsbook metrics: "
            f"rows={training_summary['best_sportsbook_metrics']['rows']}, "
            f"brier={training_summary['best_sportsbook_metrics']['brier']:.5f}, "
            f"log_loss={training_summary['best_sportsbook_metrics']['log_loss']:.5f}"
        )
    print(
        "sample monotone check:",
        bool(np.all(np.diff(test_probs[:100], axis=1) <= 1e-9)) if len(test_probs) else True,
    )

    prediction_lookup = {}
    for frame, probs in ((val_df, val_probs), (test_df, test_probs)):
        prediction_lookup.update(build_prediction_lookup_from_frame(frame, probs))

    prediction_rows, stats = build_prediction_rows(
        mapping_rows=mapping_rows,
        odds_dir=args.odds_dir,
        prediction_lookup=prediction_lookup,
        roster_rows=roster_rows,
        roster_index=roster_index,
        thresholds=thresholds,
    )
    write_prediction_rows(args.output, prediction_rows)
    save_artifact(
        path=args.artifact_output,
        model=model,
        standardizer=standardizer,
        thresholds=thresholds,
        threshold_weights=threshold_weights,
        pos_weights=pos_weights,
        args=args,
        training_summary=training_summary,
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
    )

    output_counter = Counter(row["dataset_split"] for row in prediction_rows)
    print(f"wrote {len(prediction_rows)} sportsbook rows to {args.output}")
    print(f"split counts in sportsbook output: {dict(sorted(output_counter.items()))}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
