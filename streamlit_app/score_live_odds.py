#!/usr/bin/env python3
"""Score current live player-points alternate ladders with saved model artifacts."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from multi_output_threshold_mlp.odds_processing import (
    line_key,
    load_odds_event,
    required_threshold_from_line,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
ROSTERS_CSV = DATA_DIR / "rosters.csv"
PLAYERS_CSV = DATA_DIR / "box_scores" / "players.csv"
TEAMS_CSV = DATA_DIR / "box_scores" / "teams.csv"
LIVE_ALT_DIR = DATA_DIR / "current_odds_api" / "player_points_alternate"
STAGE1_PATH = PROJECT_ROOT / "multi_output_threshold_mlp" / "multi_output_threshold_mlp.py"
STAGE2_PATH = PROJECT_ROOT / "multi_output_threshold_mlp" / "stacked_bet_quality_model.py"
STAGE1_ARTIFACT = PROJECT_ROOT / "multi_output_threshold_mlp" / "artifacts" / "threshold_mlp_artifact.pt"
STAGE2_ARTIFACT = (
    PROJECT_ROOT / "multi_output_threshold_mlp" / "artifacts" / "stacked_bet_quality_artifact.joblib"
)


@lru_cache(maxsize=4)
def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_rosters() -> pd.DataFrame:
    rosters = pd.read_csv(ROSTERS_CSV, dtype={"PLAYER_ID": str, "TEAM_ID": str, "TeamID": str})
    rosters["PLAYER_ID"] = rosters["PLAYER_ID"].astype(str)
    rosters["TEAM_ID"] = rosters["TEAM_ID"].fillna(rosters["TeamID"]).astype(str)
    rosters["TEAM_NAME"] = rosters["TEAM_NAME"].fillna("").astype(str).str.strip()
    rosters["PLAYER"] = rosters["PLAYER"].fillna("").astype(str).str.strip()
    return rosters


@lru_cache(maxsize=8)
def load_live_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def unique_live_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    person_ids = {str(row["personId"]) for row in rows}
    if len(person_ids) == 1:
        return rows[0]
    return None


def match_live_player(stage1_module, game_id: str, odds_player_name: str, roster_rows, roster_index):
    for alias in stage1_module.alias_keys(odds_player_name):
        match = unique_live_row(roster_index[game_id].get(alias, []))
        if match is not None:
            return match

    _, core_tokens = stage1_module.name_tokens(odds_player_name)
    family_name = stage1_module.family_name_from_core(core_tokens)
    if not family_name:
        return None

    first_initial = core_tokens[0][0]
    candidates = []
    for row in roster_rows[game_id]:
        _, record_core = stage1_module.name_tokens(row["player_name_box_score"])
        if not record_core:
            continue
        if stage1_module.family_name_from_core(record_core) != family_name:
            continue
        if record_core[0][0] != first_initial:
            continue
        candidates.append(row)
    return unique_live_row(candidates)


def latest_team_feature_frames(stage1_module, team_window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = stage1_module.load_teams(str(TEAMS_CSV))
    own_lookup, opponent_lookup, _ = stage1_module.build_team_feature_lookups(
        teams,
        team_window=team_window,
    )

    team_dates = teams[["gameId", "teamId", "gameDate"]].rename(
        columns={"gameId": "game_id", "teamId": "team_id", "gameDate": "game_date"}
    )
    own_latest = (
        own_lookup.merge(team_dates, on=["game_id", "team_id"], how="left")
        .sort_values(["team_id", "game_date", "game_id"])
        .drop_duplicates(subset=["team_id"], keep="last")
        .reset_index(drop=True)
    )

    opp_dates = team_dates.rename(columns={"team_id": "opponent_team_id"})
    opp_latest = (
        opponent_lookup.merge(opp_dates, on=["game_id", "opponent_team_id"], how="left")
        .sort_values(["opponent_team_id", "game_date", "game_id"])
        .drop_duplicates(subset=["opponent_team_id"], keep="last")
        .reset_index(drop=True)
    )
    return own_latest, opp_latest


@lru_cache(maxsize=4)
def build_live_feature_rows(
    player_window: int,
    recent_window: int,
    team_window: int,
) -> tuple[pd.DataFrame, dict, dict]:
    stage1_module = load_module(STAGE1_PATH, "live_stage1_module")
    dataset, _, _, _ = stage1_module.build_feature_dataset(
        players_csv=str(PLAYERS_CSV),
        teams_csv=str(TEAMS_CSV),
        player_window=player_window,
        recent_window=recent_window,
        team_window=team_window,
    )
    latest_player_rows = (
        dataset.sort_values(["person_id", "game_date", "game_id"])
        .drop_duplicates(subset=["person_id"], keep="last")
        .reset_index(drop=True)
    )
    latest_player_by_id = {
        str(row["person_id"]): row for row in latest_player_rows.to_dict("records")
    }

    rosters = load_rosters()
    team_name_to_id = (
        rosters[["TEAM_NAME", "TEAM_ID"]]
        .drop_duplicates()
        .set_index("TEAM_NAME")["TEAM_ID"]
        .to_dict()
    )
    team_id_to_name = (
        rosters[["TEAM_ID", "TEAM_NAME"]]
        .drop_duplicates()
        .set_index("TEAM_ID")["TEAM_NAME"]
        .to_dict()
    )

    own_latest, opp_latest = latest_team_feature_frames(stage1_module, team_window=team_window)
    own_by_team = {
        str(row["team_id"]): row for row in own_latest.to_dict("records")
    }
    opp_by_team = {
        str(row["opponent_team_id"]): row for row in opp_latest.to_dict("records")
    }

    live_rows = []
    events = load_live_json(LIVE_ALT_DIR / "events.json")
    for event in events:
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            continue

        home_team_name = str(event.get("home_team", "")).strip()
        away_team_name = str(event.get("away_team", "")).strip()
        home_team_id = team_name_to_id.get(home_team_name)
        away_team_id = team_name_to_id.get(away_team_name)
        if not home_team_id or not away_team_id:
            continue

        event_ts = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(event_ts):
            continue
        event_date = event_ts.tz_convert(None)
        synthetic_game_id = f"live::{event_id}"
        season = stage1_module.infer_season(event_date)

        for team_id, team_name, opponent_team_id, is_home_int in [
            (home_team_id, home_team_name, away_team_id, 1),
            (away_team_id, away_team_name, home_team_id, 0),
        ]:
            own_row = own_by_team.get(str(team_id))
            opp_row = opp_by_team.get(str(opponent_team_id))
            if own_row is None or opp_row is None:
                continue

            team_roster = rosters[rosters["TEAM_ID"] == str(team_id)]
            for _, roster_row in team_roster.iterrows():
                person_id = str(roster_row["PLAYER_ID"])
                latest_row = latest_player_by_id.get(person_id)
                if latest_row is None:
                    continue

                row = {
                    "game_id": synthetic_game_id,
                    "game_date": event_date,
                    "season": season,
                    "dataset_split": "live",
                    "person_id": person_id,
                    "team_id": str(team_id),
                    "player_name_box_score": latest_row["player_name_box_score"] or roster_row["PLAYER"],
                    "actual_points": np.nan,
                    "rolling_mean_points": latest_row["rolling_mean_points"],
                    "player_points_median_roll": latest_row["player_points_median_roll"],
                    "player_points_std_roll": latest_row["player_points_std_roll"],
                    "player_minutes_roll": latest_row["player_minutes_roll"],
                    "player_fga_roll": latest_row["player_fga_roll"],
                    "player_3pa_roll": latest_row["player_3pa_roll"],
                    "player_fta_roll": latest_row["player_fta_roll"],
                    "player_rebounds_roll": latest_row["player_rebounds_roll"],
                    "player_assists_roll": latest_row["player_assists_roll"],
                    "player_usage_roll": latest_row["player_usage_roll"],
                    "player_points_recent_roll": latest_row["player_points_recent_roll"],
                    "player_minutes_recent_roll": latest_row["player_minutes_recent_roll"],
                    "player_fga_recent_roll": latest_row["player_fga_recent_roll"],
                    "player_3pa_recent_roll": latest_row["player_3pa_recent_roll"],
                    "player_fta_recent_roll": latest_row["player_fta_recent_roll"],
                    "player_usage_recent_roll": latest_row["player_usage_recent_roll"],
                    "player_points_trend": latest_row["player_points_trend"],
                    "player_minutes_trend": latest_row["player_minutes_trend"],
                    "player_fga_trend": latest_row["player_fga_trend"],
                    "player_3pa_trend": latest_row["player_3pa_trend"],
                    "player_fta_trend": latest_row["player_fta_trend"],
                    "player_usage_trend": latest_row["player_usage_trend"],
                    "is_home_int": is_home_int,
                    "team_pace_roll": own_row["team_pace_roll"],
                    "opp_points_roll": opp_row["opp_points_roll"],
                    "opp_pace_roll": opp_row["opp_pace_roll"],
                    "opp_net_rating_roll": opp_row["opp_net_rating_roll"],
                    "opp_defensive_rating_roll": opp_row["opp_defensive_rating_roll"],
                    "event_id": event_id,
                    "event_home_team": home_team_name,
                    "event_away_team": away_team_name,
                    "commence_time": event.get("commence_time", ""),
                    "team_name": team_name,
                    "opponent_team_name": team_id_to_name.get(str(opponent_team_id), ""),
                }
                team_points_roll = own_row["team_points_roll"]
                team_fga_roll = own_row["team_fga_roll"]
                row["player_share_team_points_roll"] = (
                    row["rolling_mean_points"] / team_points_roll if pd.notna(team_points_roll) else np.nan
                )
                row["player_share_team_fga_roll"] = (
                    row["player_fga_roll"] / team_fga_roll if pd.notna(team_fga_roll) else np.nan
                )
                live_rows.append(row)

    live_frame = pd.DataFrame(live_rows)
    if live_frame.empty:
        return live_frame, {}, {}

    for column in ["player_share_team_points_roll", "player_share_team_fga_roll"]:
        live_frame.loc[~np.isfinite(live_frame[column]), column] = np.nan

    usable = live_frame.dropna(subset=stage1_module.FEATURE_COLS).copy().reset_index(drop=True)
    roster_index = defaultdict(lambda: defaultdict(list))
    roster_rows = defaultdict(list)
    for row in usable.to_dict("records"):
        game_id = str(row["game_id"])
        roster_rows[game_id].append(
            {
                "personId": row["person_id"],
                "player_name_box_score": row["player_name_box_score"],
            }
        )
        aliases = set(stage1_module.alias_keys(row["player_name_box_score"]))
        for alias in aliases:
            roster_index[game_id][alias].append(
                {
                    "personId": row["person_id"],
                    "player_name_box_score": row["player_name_box_score"],
                }
            )

    return usable, roster_rows, roster_index


@lru_cache(maxsize=1)
def load_stage1_model():
    stage1_module = load_module(STAGE1_PATH, "live_stage1_module")
    artifact = torch.load(STAGE1_ARTIFACT, map_location="cpu", weights_only=False)
    config = artifact["config"]
    thresholds = np.asarray(artifact["thresholds"], dtype=np.int32)
    model = stage1_module.ThresholdMLP(
        input_dim=len(artifact["feature_cols"]),
        output_dim=len(thresholds),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    standardizer = stage1_module.Standardizer(
        mean=np.asarray(artifact["standardizer_mean"], dtype=np.float32),
        std=np.asarray(artifact["standardizer_std"], dtype=np.float32),
    )
    return stage1_module, artifact, model, standardizer, thresholds


@lru_cache(maxsize=1)
def load_stage2_runtime():
    stage2_module = load_module(STAGE2_PATH, "live_stage2_module")
    artifact = joblib.load(STAGE2_ARTIFACT)
    return stage2_module, artifact


def build_live_prediction_rows(
    stage1_module,
    live_features: pd.DataFrame,
    roster_rows,
    roster_index,
) -> pd.DataFrame:
    if live_features.empty:
        return pd.DataFrame()

    _, artifact, model, standardizer, thresholds = load_stage1_model()
    feature_values = live_features[artifact["feature_cols"]].to_numpy(dtype=np.float32)
    standardized = standardizer.transform(feature_values)

    with torch.no_grad():
        probs = torch.sigmoid(torch.tensor(model(torch.tensor(standardized)).numpy())).numpy()
    probs = stage1_module.monotone_tail_probabilities(probs)

    prediction_lookup = {}
    for row, tail_probs in zip(live_features.to_dict("records"), probs, strict=True):
        prediction_lookup[(str(row["game_id"]), str(row["person_id"]))] = {
            "game_date": pd.Timestamp(row["game_date"]).strftime("%Y-%m-%d"),
            "season": row["season"],
            "dataset_split": row["dataset_split"],
            "team_id": row["team_id"],
            "player_name_box_score": row["player_name_box_score"],
            "tail_probabilities": tail_probs,
            "event_id": row["event_id"],
            "event_home_team": row["event_home_team"],
            "event_away_team": row["event_away_team"],
            "commence_time": row["commence_time"],
        }

    threshold_min = int(thresholds.min())
    threshold_max = int(thresholds.max())
    rows = []
    for event_file in sorted(LIVE_ALT_DIR.glob("*.json")):
        if event_file.name == "events.json":
            continue

        event = load_odds_event(event_file)
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            continue
        synthetic_game_id = f"live::{event_id}"

        for ladder_row in stage1_module.player_ladder_rows(event):
            matched = match_live_player(
                stage1_module,
                game_id=synthetic_game_id,
                odds_player_name=ladder_row["player_name_odds"],
                roster_rows=roster_rows,
                roster_index=roster_index,
            )
            if matched is None:
                continue

            person_id = str(matched["personId"])
            prediction_row = prediction_lookup.get((synthetic_game_id, person_id))
            if prediction_row is None:
                continue

            required_threshold = required_threshold_from_line(ladder_row["line_points"])
            if required_threshold < threshold_min or required_threshold > threshold_max:
                continue

            over_prob = float(prediction_row["tail_probabilities"][required_threshold - threshold_min])
            under_prob = 1.0 - over_prob
            over_price = stage1_module.safe_float(ladder_row["over_price"])
            under_price = stage1_module.safe_float(ladder_row["under_price"])
            over_ev = over_prob * over_price - 1.0 if over_price is not None else None
            under_ev = under_prob * under_price - 1.0 if under_price is not None else None

            bet_side = ""
            bet_price = None
            bet_ev = None
            if over_ev is not None and over_ev > 0.0:
                bet_side = "over"
                bet_price = over_price
                bet_ev = over_ev
            if under_ev is not None and under_ev > 0.0 and (bet_ev is None or under_ev > bet_ev):
                bet_side = "under"
                bet_price = under_price
                bet_ev = under_ev

            rows.append(
                {
                    "game_id": synthetic_game_id,
                    "event_id": event_id,
                    "game_date": prediction_row["game_date"],
                    "season": prediction_row["season"],
                    "dataset_split": "live",
                    "person_id": person_id,
                    "team_id": prediction_row["team_id"],
                    "player_name_odds": ladder_row["player_name_odds"],
                    "player_name_box_score": prediction_row["player_name_box_score"],
                    "bookmaker_key": ladder_row["bookmaker_key"],
                    "bookmaker_title": ladder_row["bookmaker_title"],
                    "bookmaker_last_update": ladder_row["bookmaker_last_update"],
                    "market_last_update": ladder_row["market_last_update"],
                    "over_bookmaker_key": ladder_row.get("over_bookmaker_key", ""),
                    "over_bookmaker_title": ladder_row.get("over_bookmaker_title", ""),
                    "under_bookmaker_key": ladder_row.get("under_bookmaker_key", ""),
                    "under_bookmaker_title": ladder_row.get("under_bookmaker_title", ""),
                    "bookmaker_count": ladder_row.get("bookmaker_count", 0),
                    "over_bookmaker_count": ladder_row.get("over_bookmaker_count", 0),
                    "under_bookmaker_count": ladder_row.get("under_bookmaker_count", 0),
                    "all_bookmaker_titles": ladder_row.get("all_bookmaker_titles", ""),
                    "line_points": float(ladder_row["line_points"]),
                    "line_key": line_key(ladder_row["line_points"]),
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
                    "actual_points": np.nan,
                    "actual_result": "",
                    "bet_result": "",
                    "event_home_team": prediction_row["event_home_team"],
                    "event_away_team": prediction_row["event_away_team"],
                    "commence_time": prediction_row["commence_time"],
                }
            )

    return pd.DataFrame(rows)


def apply_stage2_scores(stage2_module, predictions: pd.DataFrame, live_features: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions

    base_feature_cols = [
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
    base = predictions.merge(
        live_features[base_feature_cols],
        on=["game_id", "person_id", "team_id"],
        how="left",
    )

    _, artifact = load_stage2_runtime()
    longshot_cutoff = float(artifact["config"].get("longshot_cutoff", 3.0))
    stage2 = stage2_module.add_stage2_features(base, longshot_cutoff=longshot_cutoff)
    return stage2_module.score_rows_with_artifact(stage2, artifact)


def score_current_points_ladders() -> pd.DataFrame:
    stage1_module, stage1_artifact, _, _, _ = load_stage1_model()
    stage2_module, _ = load_stage2_runtime()
    config = stage1_artifact["config"]

    live_features, roster_rows, roster_index = build_live_feature_rows(
        player_window=int(config["player_window"]),
        recent_window=int(config["recent_window"]),
        team_window=int(config["team_window"]),
    )
    if live_features.empty:
        return pd.DataFrame()

    predictions = build_live_prediction_rows(
        stage1_module,
        live_features,
        roster_rows,
        roster_index,
    )
    if predictions.empty:
        return predictions

    scored = apply_stage2_scores(stage2_module, predictions, live_features)
    sort_cols = [col for col in ["trust_blend_score", "bet_ev", "player_name_odds"] if col in scored.columns]
    if sort_cols:
        scored = scored.sort_values(sort_cols, ascending=[False, False, True]).reset_index(drop=True)
    return scored
