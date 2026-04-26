#!/usr/bin/env python3
"""Fit a simple Stage 2 bet-quality model on top of the threshold MLP outputs.

This script treats the existing threshold MLP as Stage 1:
- Stage 1 models basketball tail probabilities.
- Stage 2 learns when a positive-EV sportsbook row is actually trustworthy.

The Stage 2 model is intentionally simple and regularized:
- fit on historical validation sportsbook rows only
- evaluate/score on both validation and test rows
- output a corrected bet-level win probability and corrected EV
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = SCRIPT_DIR / "multi_output_threshold_mlp_predictions.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "stacked_bet_quality_predictions.csv"
DEFAULT_ARTIFACT = SCRIPT_DIR / "artifacts" / "stacked_bet_quality_artifact.joblib"
DEFAULT_PLAYERS = PROJECT_ROOT / "data/box_scores/players.csv"
DEFAULT_TEAMS = PROJECT_ROOT / "data/box_scores/teams.csv"
DEFAULT_LONGSHOT_CUTOFF = 3.0
DEFAULT_ODDS_BANDS = (2.0, 3.0, 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a Stage 2 bet-quality model on sportsbook rows using the "
            "existing threshold MLP predictions as Stage 1 inputs."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--artifact-output", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--players-csv", default=str(DEFAULT_PLAYERS))
    parser.add_argument("--teams-csv", default=str(DEFAULT_TEAMS))
    parser.add_argument("--player-window", type=int, default=60)
    parser.add_argument("--recent-window", type=int, default=5)
    parser.add_argument("--team-window", type=int, default=40)
    parser.add_argument("--fit-split", default="val")
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument("--longshot-cutoff", type=float, default=DEFAULT_LONGSHOT_CUTOFF)
    parser.add_argument(
        "--fit-min-bet-price",
        type=float,
        default=0.0,
        help="Optional minimum decimal price to include in Stage 2 fit rows. 0 disables.",
    )
    parser.add_argument(
        "--fit-max-bet-price",
        type=float,
        default=0.0,
        help="Optional maximum decimal price to include in Stage 2 fit rows. 0 disables.",
    )
    parser.add_argument(
        "--recency-weighting",
        choices=["none", "exponential"],
        default="none",
        help="Optional date-based weighting scheme for Stage 2 fit rows.",
    )
    parser.add_argument(
        "--recency-half-life-days",
        type=float,
        default=30.0,
        help="Half-life in days for exponential recency weighting.",
    )
    parser.add_argument(
        "--regime",
        choices=["global", "odds_band", "longshot"],
        default="global",
        help="Fit separate Stage 2 models by regime when enough rows exist.",
    )
    parser.add_argument(
        "--min-regime-rows",
        type=int,
        default=5000,
        help="Minimum resolved fit rows required to train a separate regime Stage 2 model.",
    )
    parser.add_argument("--min-bet-price", type=float, default=1.01)
    parser.add_argument(
        "--model-type",
        choices=["logistic", "hist_gbdt"],
        default="logistic",
        help="Stage 2 model family.",
    )
    parser.add_argument("--c", type=float, default=0.03)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--ridge-alpha", type=float, default=30.0)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    return parser.parse_args()


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_band_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def odds_band_label(price: float, cutoffs: tuple[float, ...]) -> str:
    lower = 0.0
    for upper in cutoffs:
        if price < upper:
            return f"[{format_band_label(lower)}, {format_band_label(upper)})"
        lower = upper
    return f"[{format_band_label(lower)}, inf)"


def load_stage1_module():
    module_path = SCRIPT_DIR / "multi_output_threshold_mlp.py"
    module_name = "stage1_threshold_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Stage 1 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        low_memory=False,
        dtype={
            "game_id": str,
            "event_id": str,
            "person_id": str,
            "team_id": str,
            "bookmaker_key": str,
            "bookmaker_title": str,
            "bookmaker_last_update": str,
            "market_last_update": str,
            "over_bookmaker_key": str,
            "over_bookmaker_title": str,
            "under_bookmaker_key": str,
            "under_bookmaker_title": str,
            "dataset_split": str,
        },
    )
    for column in ["bookmaker_count", "over_bookmaker_count", "under_bookmaker_count"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["line_points"] = pd.to_numeric(df["line_points"], errors="coerce")
    df["required_threshold"] = pd.to_numeric(df["required_threshold"], errors="coerce")
    df["over_price"] = pd.to_numeric(df["over_price"], errors="coerce")
    df["under_price"] = pd.to_numeric(df["under_price"], errors="coerce")
    df["model_over_prob"] = pd.to_numeric(df["model_over_prob"], errors="coerce")
    df["model_under_prob"] = pd.to_numeric(df["model_under_prob"], errors="coerce")
    df["over_ev"] = pd.to_numeric(df["over_ev"], errors="coerce")
    df["under_ev"] = pd.to_numeric(df["under_ev"], errors="coerce")
    df["bet_price"] = pd.to_numeric(df["bet_price"], errors="coerce")
    df["bet_ev"] = pd.to_numeric(df["bet_ev"], errors="coerce")
    df["actual_points"] = pd.to_numeric(df["actual_points"], errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def build_stage2_base_frame(
    predictions: pd.DataFrame,
    players_csv: str,
    teams_csv: str,
    player_window: int,
    recent_window: int,
    team_window: int,
) -> pd.DataFrame:
    stage1_module = load_stage1_module()
    feature_dataset, _, _, _ = stage1_module.build_feature_dataset(
        players_csv=players_csv,
        teams_csv=teams_csv,
        player_window=player_window,
        recent_window=recent_window,
        team_window=team_window,
    )

    feature_cols = [
        "game_id",
        "person_id",
        "team_id",
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
        "team_pace_roll",
        "opp_points_roll",
        "opp_pace_roll",
        "opp_net_rating_roll",
        "opp_defensive_rating_roll",
    ]
    feature_frame = feature_dataset[feature_cols].copy()
    feature_frame["game_id"] = feature_frame["game_id"].astype(str)
    feature_frame["person_id"] = feature_frame["person_id"].astype(str)
    feature_frame["team_id"] = feature_frame["team_id"].astype(str)
    feature_frame = feature_frame.drop_duplicates(subset=["game_id", "person_id"])

    merged = predictions.merge(
        feature_frame,
        on=["game_id", "person_id", "team_id"],
        how="left",
    )
    return merged


def add_stage2_features(df: pd.DataFrame, longshot_cutoff: float) -> pd.DataFrame:
    output = df.copy()
    output["bet_side"] = output["bet_side"].fillna("")
    output["bet_result"] = output["bet_result"].fillna("")

    output["implied_prob"] = np.where(
        output["bet_price"].notna() & (output["bet_price"] > 0.0),
        1.0 / output["bet_price"],
        np.nan,
    )
    output["raw_edge"] = output["model_over_prob"] - output["implied_prob"]
    output.loc[output["bet_side"] == "under", "raw_edge"] = (
        output.loc[output["bet_side"] == "under", "model_under_prob"]
        - output.loc[output["bet_side"] == "under", "implied_prob"]
    )
    output["raw_bet_win_prob"] = np.where(
        output["bet_side"] == "over",
        output["model_over_prob"],
        np.where(output["bet_side"] == "under", output["model_under_prob"], np.nan),
    )
    output["side_is_over"] = (output["bet_side"] == "over").astype(int)
    output["longshot_flag"] = (
        output["bet_price"].notna() & (output["bet_price"] >= longshot_cutoff)
    ).astype(int)
    output["odds_band"] = output["bet_price"].map(
        lambda value: odds_band_label(float(value), DEFAULT_ODDS_BANDS)
        if pd.notna(value)
        else "missing"
    )
    output["longshot_regime"] = output["bet_price"].map(
        lambda value: (
            f"longshot(>={longshot_cutoff:.2f})"
            if float(value) >= longshot_cutoff
            else f"non_longshot(<{longshot_cutoff:.2f})"
        )
        if pd.notna(value)
        else "missing"
    )
    output["bookmaker_side_key"] = (
        output["bookmaker_key"].fillna("missing").astype(str)
        + "__"
        + output["bet_side"].fillna("missing").astype(str)
    )
    output["bookmaker_odds_band_key"] = (
        output["bookmaker_key"].fillna("missing").astype(str)
        + "__"
        + output["odds_band"].fillna("missing").astype(str)
    )
    output["bookmaker_longshot_key"] = (
        output["bookmaker_key"].fillna("missing").astype(str)
        + "__"
        + output["longshot_regime"].fillna("missing").astype(str)
    )
    output["side_odds_band_key"] = (
        output["bet_side"].fillna("missing").astype(str)
        + "__"
        + output["odds_band"].fillna("missing").astype(str)
    )
    output["log_bet_price"] = np.where(
        output["bet_price"].notna() & (output["bet_price"] > 0.0),
        np.log(output["bet_price"]),
        np.nan,
    )
    output["line_minus_roll_mean"] = output["line_points"] - output["rolling_mean_points"]
    output["line_minus_roll_median"] = (
        output["line_points"] - output["player_points_median_roll"]
    )
    output["line_minus_recent_mean"] = (
        output["line_points"] - output["player_points_recent_roll"]
    )
    output["line_z_from_std"] = (
        output["line_points"] - output["rolling_mean_points"]
    ) / output["player_points_std_roll"]
    output.loc[~np.isfinite(output["line_z_from_std"]), "line_z_from_std"] = np.nan
    output["bet_ev_abs"] = output["bet_ev"].abs()
    output["line_depth_over_mean"] = np.maximum(output["line_minus_roll_mean"], 0.0)
    output["line_depth_under_mean"] = np.maximum(-output["line_minus_roll_mean"], 0.0)

    candidate_mask = output["bet_side"] != ""
    group_keys = ["game_id", "team_id"]
    output["same_team_candidate_count"] = (
        output[candidate_mask]
        .groupby(group_keys)["person_id"]
        .transform("size")
        .reindex(output.index)
    )
    output["same_team_candidate_count"] = output["same_team_candidate_count"].fillna(0.0)

    bookmaker_group_keys = ["game_id", "team_id", "bookmaker_key"]
    output["same_team_book_candidate_count"] = (
        output[candidate_mask]
        .groupby(bookmaker_group_keys)["person_id"]
        .transform("size")
        .reindex(output.index)
    )
    output["same_team_book_candidate_count"] = output["same_team_book_candidate_count"].fillna(0.0)

    output["same_team_higher_ev_count"] = 0.0
    output["same_team_book_higher_ev_count"] = 0.0
    output["same_team_max_other_ev"] = np.nan

    candidate_rows = output[candidate_mask].copy()
    if not candidate_rows.empty:
        candidate_rows["same_team_higher_ev_count"] = (
            candidate_rows.groupby(group_keys)["bet_ev"]
            .rank(method="min", ascending=False)
            .sub(1.0)
        )
        candidate_rows["same_team_book_higher_ev_count"] = (
            candidate_rows.groupby(bookmaker_group_keys)["bet_ev"]
            .rank(method="min", ascending=False)
            .sub(1.0)
        )
        candidate_rows["same_team_max_other_ev"] = candidate_rows.groupby(group_keys)["bet_ev"].transform("max")
        output.loc[candidate_rows.index, "same_team_higher_ev_count"] = candidate_rows["same_team_higher_ev_count"]
        output.loc[candidate_rows.index, "same_team_book_higher_ev_count"] = candidate_rows[
            "same_team_book_higher_ev_count"
        ]
        output.loc[candidate_rows.index, "same_team_max_other_ev"] = candidate_rows[
            "same_team_max_other_ev"
        ]

    player_ladder_keys = ["game_id", "person_id", "bookmaker_key"]
    output["player_book_line_rank"] = (
        output.groupby(player_ladder_keys)["line_points"]
        .rank(method="dense", ascending=True)
        .astype(float)
    )
    output["player_book_line_count"] = (
        output.groupby(player_ladder_keys)["line_points"].transform("size").astype(float)
    )

    output["bet_win_target"] = np.where(output["bet_result"] == "win", 1.0, 0.0)
    output["bet_resolved"] = output["bet_result"].isin(["win", "loss"])
    output["profit_target"] = np.where(
        output["bet_result"] == "win",
        output["bet_price"] - 1.0,
        np.where(output["bet_result"] == "loss", -1.0, np.nan),
    )
    output["candidate_row"] = candidate_mask
    return output


def stage2_feature_lists(df: pd.DataFrame | None = None) -> tuple[list[str], list[str]]:
    numeric_features = [
        "model_over_prob",
        "model_under_prob",
        "over_ev",
        "under_ev",
        "bet_ev",
        "bet_price",
        "implied_prob",
        "raw_edge",
        "log_bet_price",
        "line_points",
        "required_threshold",
        "line_minus_roll_mean",
        "line_minus_roll_median",
        "line_minus_recent_mean",
        "line_z_from_std",
        "bet_ev_abs",
        "line_depth_over_mean",
        "line_depth_under_mean",
        "rolling_mean_points",
        "player_points_median_roll",
        "player_points_std_roll",
        "player_minutes_roll",
        "player_usage_roll",
        "player_points_recent_roll",
        "player_minutes_recent_roll",
        "player_usage_recent_roll",
        "player_points_trend",
        "player_minutes_trend",
        "player_usage_trend",
        "player_share_team_points_roll",
        "player_share_team_fga_roll",
        "team_pace_roll",
        "opp_points_roll",
        "opp_pace_roll",
        "opp_net_rating_roll",
        "opp_defensive_rating_roll",
        "same_team_candidate_count",
        "same_team_book_candidate_count",
        "same_team_higher_ev_count",
        "same_team_book_higher_ev_count",
        "same_team_max_other_ev",
        "player_book_line_rank",
        "player_book_line_count",
    ]
    categorical_features = [
        "bet_side",
        "bookmaker_key",
        "bookmaker_side_key",
        "bookmaker_odds_band_key",
        "bookmaker_longshot_key",
        "side_odds_band_key",
        "season",
        "odds_band",
        "longshot_regime",
        "longshot_flag",
        "side_is_over",
    ]
    if df is not None:
        numeric_features = [
            column
            for column in numeric_features
            if column in df.columns and df[column].notna().any()
        ]
        categorical_features = [
            column for column in categorical_features if column in df.columns
        ]
    return numeric_features, categorical_features


def build_stage2_pipeline(
    *,
    model_type: str,
    c_value: float,
    max_iter: int,
    learning_rate: float,
    max_depth: int,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    numeric_features,
    categorical_features,
):
    if model_type == "logistic":
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ]
        )
        model = LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            solver="lbfgs",
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    if model_type == "hist_gbdt":
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ]
        )
        model = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=42,
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    raise ValueError(f"Unsupported Stage 2 model_type: {model_type}")


def build_return_pipeline(
    *,
    ridge_alpha: float,
    numeric_features,
    categorical_features,
):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    model = Ridge(alpha=ridge_alpha)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def compute_metrics(df: pd.DataFrame, prob_col: str, ev_col: str) -> dict:
    working = df.copy()
    working["profit"] = np.where(
        working["bet_result"] == "win",
        working["bet_price"] - 1.0,
        np.where(working["bet_result"] == "loss", -1.0, np.nan),
    )
    resolved = working[working["bet_resolved"]].copy()
    if resolved.empty:
        return {
            "rows": 0,
            "brier": np.nan,
            "log_loss": np.nan,
            "win_rate": np.nan,
            "roi": np.nan,
            "expected_profit_per_stake": np.nan,
        }

    probs = np.clip(resolved[prob_col].to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
    actual = resolved["bet_win_target"].to_numpy(dtype=np.float64)
    brier = float(np.mean((probs - actual) ** 2))
    log_loss = float(-np.mean(actual * np.log(probs) + (1.0 - actual) * np.log(1.0 - probs)))
    return {
        "rows": int(len(resolved)),
        "brier": brier,
        "log_loss": log_loss,
        "win_rate": float(resolved["bet_win_target"].mean()),
        "roi": float(resolved["profit"].mean()),
        "expected_profit_per_stake": float(resolved[ev_col].mean()),
    }


def compute_return_metrics(df: pd.DataFrame, pred_col: str) -> dict:
    resolved = df[df["bet_resolved"] & df[pred_col].notna() & df["profit_target"].notna()].copy()
    if resolved.empty:
        return {
            "rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "pred_mean": np.nan,
            "actual_mean": np.nan,
        }

    pred = resolved[pred_col].to_numpy(dtype=np.float64)
    actual = resolved["profit_target"].to_numpy(dtype=np.float64)
    return {
        "rows": int(len(resolved)),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(mean_squared_error(actual, pred) ** 0.5),
        "pred_mean": float(np.mean(pred)),
        "actual_mean": float(np.mean(actual)),
    }


def build_regime_labels(df: pd.DataFrame, regime: str) -> pd.Series:
    if regime == "global":
        return pd.Series(["global"] * len(df), index=df.index, dtype=object)
    if regime == "odds_band":
        return df["odds_band"].astype(object)
    if regime == "longshot":
        return df["longshot_regime"].astype(object)
    raise ValueError(f"Unsupported Stage 2 regime: {regime}")


def build_recency_weights(
    df: pd.DataFrame,
    *,
    weighting: str,
    half_life_days: float,
) -> np.ndarray:
    if weighting == "none" or df.empty:
        return np.ones(len(df), dtype=np.float64)

    if weighting != "exponential":
        raise ValueError(f"Unsupported recency weighting: {weighting}")

    if half_life_days <= 0.0:
        raise ValueError("recency half-life must be positive")

    game_dates = pd.to_datetime(df["game_date"], errors="coerce")
    if game_dates.notna().sum() == 0:
        return np.ones(len(df), dtype=np.float64)

    max_date = game_dates.max()
    age_days = (max_date - game_dates).dt.total_seconds() / 86400.0
    age_days = age_days.fillna(age_days.max() if np.isfinite(age_days.max()) else 0.0)
    weights = np.power(0.5, age_days.to_numpy(dtype=np.float64) / half_life_days)
    mean_weight = float(np.mean(weights))
    if not np.isfinite(mean_weight) or mean_weight <= 0.0:
        return np.ones(len(df), dtype=np.float64)
    return weights / mean_weight


def fit_stage2_models(
    fit_df: pd.DataFrame,
    *,
    args,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict:
    if fit_df.empty:
        raise ValueError("fit_df must not be empty.")

    fit_weights = build_recency_weights(
        fit_df,
        weighting=args.recency_weighting,
        half_life_days=args.recency_half_life_days,
    )

    pipeline = build_stage2_pipeline(
        model_type=args.model_type,
        c_value=args.c,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    return_pipeline = build_return_pipeline(
        ridge_alpha=args.ridge_alpha,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    global_pipeline = clone(pipeline)
    global_pipeline.fit(
        fit_df[numeric_features + categorical_features],
        fit_df["bet_win_target"],
        model__sample_weight=fit_weights,
    )
    global_return_pipeline = clone(return_pipeline)
    global_return_pipeline.fit(
        fit_df[numeric_features + categorical_features],
        fit_df["profit_target"],
        model__sample_weight=fit_weights,
    )

    regime_models = {}
    regime_return_models = {}
    regime_counts = {}
    regime_labels = build_regime_labels(fit_df, args.regime)
    for label in pd.unique(regime_labels):
        label_mask = regime_labels == label
        label_rows = fit_df.loc[label_mask].copy()
        regime_counts[str(label)] = int(len(label_rows))
        if args.regime == "global" or len(label_rows) < args.min_regime_rows:
            continue
        label_weights = fit_weights[np.asarray(label_mask)]
        regime_pipeline = clone(pipeline)
        regime_pipeline.fit(
            label_rows[numeric_features + categorical_features],
            label_rows["bet_win_target"],
            model__sample_weight=label_weights,
        )
        regime_models[str(label)] = regime_pipeline
        regime_return_pipeline = clone(return_pipeline)
        regime_return_pipeline.fit(
            label_rows[numeric_features + categorical_features],
            label_rows["profit_target"],
            model__sample_weight=label_weights,
        )
        regime_return_models[str(label)] = regime_return_pipeline

    return {
        "pipeline": global_pipeline,
        "return_pipeline": global_return_pipeline,
        "regime_models": regime_models,
        "regime_return_models": regime_return_models,
        "regime_counts": regime_counts,
        "fit_weight_summary": {
            "min": float(np.min(fit_weights)),
            "mean": float(np.mean(fit_weights)),
            "max": float(np.max(fit_weights)),
        },
    }


def add_trust_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["trust_bet_side"] = output["bet_side"]
    output["trust_bet_price"] = output["bet_price"]
    output["trust_bet_ev"] = np.where(
        output["trust_win_prob"].notna() & output["bet_price"].notna(),
        output["trust_win_prob"] * output["bet_price"] - 1.0,
        np.nan,
    )
    clipped_raw_prob = np.clip(output["raw_bet_win_prob"], 1e-6, np.inf)
    output["trust_prob_ratio"] = np.where(
        output["trust_win_prob"].notna() & output["raw_bet_win_prob"].notna(),
        output["trust_win_prob"] / clipped_raw_prob,
        np.nan,
    )
    output["trust_weighted_raw_ev"] = np.where(
        output["trust_prob_ratio"].notna() & output["bet_ev"].notna(),
        output["bet_ev"] * output["trust_prob_ratio"],
        np.nan,
    )
    output["trust_blend_score"] = np.where(
        output["trust_bet_ev"].notna() & output["trust_return_pred"].notna(),
        0.5 * output["trust_bet_ev"] + 0.5 * output["trust_return_pred"],
        np.nan,
    )
    output["trust_edge"] = np.where(
        output["trust_win_prob"].notna() & output["implied_prob"].notna(),
        output["trust_win_prob"] - output["implied_prob"],
        np.nan,
    )
    output["trust_score"] = np.where(
        output["trust_win_prob"].notna() & output["bet_ev"].notna(),
        output["trust_win_prob"] * output["bet_ev"],
        np.nan,
    )
    return output


def score_candidate_rows(
    rows: pd.DataFrame,
    *,
    regime: str,
    fitted: dict,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    scored = rows.copy()
    scored["trust_win_prob"] = np.nan
    scored["trust_return_pred"] = np.nan
    scored["trust_model_source"] = ""

    candidate_mask = scored["candidate_row"].fillna(False).astype(bool)
    if candidate_mask.any():
        candidate_df = scored.loc[candidate_mask].copy()
        candidate_labels = build_regime_labels(candidate_df, regime)
        for label in pd.unique(candidate_labels):
            label_mask = candidate_labels == label
            label_index = candidate_df.index[label_mask]
            model_key = str(label)
            model = fitted["regime_models"].get(model_key, fitted["pipeline"])
            return_model = fitted["regime_return_models"].get(
                model_key,
                fitted["return_pipeline"],
            )
            source = str(label) if str(label) in fitted["regime_models"] else "global"
            features = scored.loc[label_index, numeric_features + categorical_features]
            scored.loc[label_index, "trust_win_prob"] = model.predict_proba(features)[:, 1]
            scored.loc[label_index, "trust_return_pred"] = return_model.predict(features)
            scored.loc[label_index, "trust_model_source"] = source

    return add_trust_derived_columns(scored)


def score_rows_with_artifact(rows: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    fitted = {
        "pipeline": artifact["pipeline"],
        "return_pipeline": artifact["return_pipeline"],
        "regime_models": artifact.get("regime_models", {}),
        "regime_return_models": artifact.get("regime_return_models", {}),
    }
    return score_candidate_rows(
        rows,
        regime=str(artifact.get("regime", "global")),
        fitted=fitted,
        numeric_features=artifact["numeric_features"],
        categorical_features=artifact["categorical_features"],
    )


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.input)
    base = build_stage2_base_frame(
        predictions=predictions,
        players_csv=args.players_csv,
        teams_csv=args.teams_csv,
        player_window=args.player_window,
        recent_window=args.recent_window,
        team_window=args.team_window,
    )
    stage2 = add_stage2_features(base, longshot_cutoff=args.longshot_cutoff)

    numeric_features, categorical_features = stage2_feature_lists(stage2)
    fit_mask = (
        (stage2["dataset_split"] == args.fit_split)
        & stage2["candidate_row"]
        & stage2["bet_resolved"]
    )
    if args.fit_min_bet_price > 0.0:
        fit_mask &= stage2["bet_price"].ge(args.fit_min_bet_price)
    if args.fit_max_bet_price > 0.0:
        fit_mask &= stage2["bet_price"].le(args.fit_max_bet_price)
    fit_df = stage2.loc[fit_mask].copy()
    if fit_df.empty:
        raise ValueError(f"No resolved candidate rows found for fit split={args.fit_split}")
    fitted = fit_stage2_models(
        fit_df,
        args=args,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    stage2 = score_candidate_rows(
        stage2,
        regime=args.regime,
        fitted=fitted,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    eval_splits = {split.strip() for split in args.eval_splits.split(",") if split.strip()}
    metrics = {}
    for split in sorted(eval_splits):
        split_rows = stage2[
            (stage2["dataset_split"] == split)
            & stage2["candidate_row"]
            & stage2["bet_resolved"]
        ].copy()
        if split_rows.empty:
            continue
        metrics[split] = {
            "raw": compute_metrics(split_rows, prob_col="raw_bet_win_prob", ev_col="bet_ev"),
            "stage2": compute_metrics(split_rows, prob_col="trust_win_prob", ev_col="trust_bet_ev"),
            "return_model": compute_return_metrics(split_rows, pred_col="trust_return_pred"),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage2.to_csv(output_path, index=False)

    artifact = {
        "pipeline": fitted["pipeline"],
        "return_pipeline": fitted["return_pipeline"],
        "regime_models": fitted["regime_models"],
        "regime_return_models": fitted["regime_return_models"],
        "regime": args.regime,
        "regime_counts": fitted["regime_counts"],
        "model_type": args.model_type,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "fit_split": args.fit_split,
        "config": vars(args),
        "metrics": metrics,
        "fit_weight_summary": {
            "scheme": args.recency_weighting,
            "half_life_days": args.recency_half_life_days,
            **fitted["fit_weight_summary"],
        },
    }
    artifact_path = Path(args.artifact_output)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    print(f"Wrote stacked predictions to {output_path}")
    print(f"Saved Stage 2 artifact to {artifact_path}")
    print(
        f"Stage 2 model type={args.model_type}, regime={args.regime}, "
        f"fit_split={args.fit_split}, min_regime_rows={args.min_regime_rows}"
    )
    print(
        "fit recency weights: "
        f"scheme={args.recency_weighting}, "
        f"half_life_days={args.recency_half_life_days:.1f}, "
        f"min={fitted['fit_weight_summary']['min']:.4f}, "
        f"mean={fitted['fit_weight_summary']['mean']:.4f}, "
        f"max={fitted['fit_weight_summary']['max']:.4f}"
    )
    if fitted["regime_counts"]:
        regime_summary = ", ".join(
            f"{label}: {count}"
            for label, count in sorted(fitted["regime_counts"].items())
        )
        print(f"fit regime counts: {regime_summary}")
    for split, split_metrics in metrics.items():
        raw = split_metrics["raw"]
        trust = split_metrics["stage2"]
        print(
            f"{split}: raw rows={raw['rows']}, raw_win_rate={raw['win_rate']:.4%}, "
            f"raw_roi={raw['roi']:.4%}, raw_brier={raw['brier']:.5f}, raw_log_loss={raw['log_loss']:.5f}"
        )
        print(
            f"{split}: stage2 rows={trust['rows']}, trust_win_rate={trust['win_rate']:.4%}, "
            f"trust_roi={trust['roi']:.4%}, trust_brier={trust['brier']:.5f}, trust_log_loss={trust['log_loss']:.5f}"
        )
        return_model = split_metrics["return_model"]
        print(
            f"{split}: return-model rows={return_model['rows']}, "
            f"return_mae={return_model['mae']:.5f}, return_rmse={return_model['rmse']:.5f}, "
            f"pred_mean={return_model['pred_mean']:.4%}, actual_mean={return_model['actual_mean']:.4%}"
        )


if __name__ == "__main__":
    main()
