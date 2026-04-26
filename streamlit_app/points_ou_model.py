"""Live points O/U scoring using the local quantile-model artifact."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from multi_output_threshold_mlp.odds_processing import (
    dedup_player_points_alternate_rows,
    load_odds_event,
    normalize_player_name,
)
from quantile_model.data import build_local_datasets
from quantile_model.service import evaluate_test_set, load_artifact, predict_matchup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PLAYERS_CSV = DATA_DIR / "box_scores" / "players.csv"
TEAMS_CSV = DATA_DIR / "box_scores" / "teams.csv"
ROSTERS_CSV = DATA_DIR / "rosters.csv"
CURRENT_POINTS_DIR = DATA_DIR / "current_odds_api" / "player_points"
HISTORICAL_POINTS_DIR = DATA_DIR / "historical_odds_api" / "historical_player_points"
GAME_EVENT_BIJECTION_CSV = DATA_DIR / "game_event_bijection.csv"
ARTIFACT_PATH = (
    PROJECT_ROOT / "quantile_model" / "artifacts" / "points_ou_quantile_artifact_split.pt"
)
MIN_ABS_Q50_LINE_DISTANCE = 7.0


def _clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


@lru_cache(maxsize=1)
def _load_history_state():
    history_df, current_players, current_teams = build_local_datasets(
        players_csv=PLAYERS_CSV,
        teams_csv=TEAMS_CSV,
    )
    return history_df, current_players, current_teams


@lru_cache(maxsize=1)
def _load_rosters() -> pd.DataFrame:
    rosters = pd.read_csv(
        ROSTERS_CSV,
        dtype={"PLAYER_ID": str, "TEAM_ID": str, "TeamID": str},
    )
    rosters["PLAYER_ID"] = rosters["PLAYER_ID"].astype(str)
    rosters["TEAM_ID"] = rosters["TEAM_ID"].fillna(rosters["TeamID"]).astype(str)
    rosters["PLAYER"] = rosters["PLAYER"].fillna("").astype(str).str.strip()
    rosters["PLAYER_KEY"] = rosters["PLAYER"].map(normalize_player_name)
    return rosters


def _team_name_resolution(current_teams: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for _, row in current_teams.iterrows():
        team_id = str(row["TEAM_ID"])
        city = _clean_text(row.get("TEAM_CITY"))
        nickname = _clean_text(row.get("TEAM_NAME"))
        abbreviation = _clean_text(row.get("TEAM_ABBREVIATION")).upper()
        full_name = f"{city} {nickname}".strip()
        for key in {full_name, nickname, abbreviation}:
            if key:
                mapping[key.lower()] = team_id
    return mapping


def _infer_postseason(event_date: pd.Timestamp) -> bool:
    event_date = pd.Timestamp(event_date)
    if event_date.month > 4:
        return True
    return event_date.month == 4 and event_date.day >= 15


def _apply_recommendation_rule(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["selection_side"] = pd.NA
    out["selection_distance"] = (out["line"] - out["q50"]).abs()
    out["tail_distance"] = (out["q90"] - out["line"]).abs()
    out["selection_ratio"] = pd.NA
    out["model_recommendation"] = np.where(
        out["q50"] > out["line"],
        "over",
        np.where(out["q50"] < out["line"], "under", "push"),
    )

    recommended_candidate = (
        out["model_recommendation"].isin(["over", "under"])
        & out["selection_distance"].ge(MIN_ABS_Q50_LINE_DISTANCE)
    )
    out.loc[recommended_candidate, "selection_side"] = out.loc[
        recommended_candidate, "model_recommendation"
    ]
    out["pick_odds"] = np.where(
        out["selection_side"].eq("over"),
        out["over_price"],
        np.where(out["selection_side"].eq("under"), out["under_price"], np.nan),
    )
    out["is_recommended"] = out["selection_side"].isin(["over", "under"])
    return out


def _load_points_market_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for event_file in sorted(CURRENT_POINTS_DIR.glob("*.json")):
        if event_file.name == "events.json":
            continue
        with event_file.open(encoding="utf-8") as handle:
            event = json.load(handle)

        event_id = _clean_text(event.get("id"))
        if not event_id:
            continue

        grouped: dict[tuple[str, str, float], dict] = {}
        for bookmaker in event.get("bookmakers", []) or []:
            bookmaker_key = _clean_text(bookmaker.get("key"))
            bookmaker_title = _clean_text(bookmaker.get("title"))
            for market in bookmaker.get("markets", []) or []:
                if _clean_text(market.get("key")) != "player_points":
                    continue
                for outcome in market.get("outcomes", []) or []:
                    side_name = _clean_text(outcome.get("name")).lower()
                    if side_name not in {"over", "under"}:
                        continue
                    player_name = _clean_text(outcome.get("description"))
                    if not player_name:
                        continue
                    try:
                        line = float(outcome.get("point"))
                        price = float(outcome.get("price"))
                    except (TypeError, ValueError):
                        continue

                    player_key = normalize_player_name(player_name)
                    group_key = (event_id, player_key, line)
                    if group_key not in grouped:
                        grouped[group_key] = {
                            "event_id": event_id,
                            "player_name": player_name,
                            "player_key": player_key,
                            "line": line,
                            "commence_time": _clean_text(event.get("commence_time")),
                            "home_team": _clean_text(event.get("home_team")),
                            "away_team": _clean_text(event.get("away_team")),
                            "over_price": np.nan,
                            "under_price": np.nan,
                            "books": {},
                        }
                    grouped[group_key][f"{side_name}_price"] = price
                    grouped[group_key]["books"][bookmaker_key] = bookmaker_title or bookmaker_key

        for group in grouped.values():
            rows.append(
                {
                    "event_id": group["event_id"],
                    "player_name_odds": group["player_name"],
                    "player_key": group["player_key"],
                    "line_points": group["line"],
                    "line": group["line"],
                    "commence_time": group["commence_time"],
                    "event_home_team": group["home_team"],
                    "event_away_team": group["away_team"],
                    "over_price": group["over_price"],
                    "under_price": group["under_price"],
                    "all_bookmaker_titles": ", ".join(group["books"].values()),
                    "bookmaker_count": float(len(group["books"])),
                }
            )

    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def _load_historical_points_market_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for event_file in sorted(HISTORICAL_POINTS_DIR.glob("*.json")):
        event = load_odds_event(event_file)
        event_id = _clean_text(event.get("id")) or event_file.stem
        if not event_id:
            continue
        event_rows = dedup_player_points_alternate_rows(event, market_key="player_points")
        for row in event_rows:
            rows.append(
                {
                    "event_id": event_id,
                    "player_name_odds": row.get("player_name_odds"),
                    "player_key": normalize_player_name(_clean_text(row.get("player_name_odds"))),
                    "line_points": pd.to_numeric(row.get("line_points"), errors="coerce"),
                    "line": pd.to_numeric(row.get("line_points"), errors="coerce"),
                    "over_price": pd.to_numeric(row.get("over_price"), errors="coerce"),
                    "under_price": pd.to_numeric(row.get("under_price"), errors="coerce"),
                    "all_bookmaker_titles": _clean_text(row.get("all_bookmaker_titles")),
                    "bookmaker_title": _clean_text(row.get("bookmaker_title")),
                    "bookmaker_count": pd.to_numeric(row.get("bookmaker_count"), errors="coerce"),
                    "bookmaker_last_update": _clean_text(row.get("bookmaker_last_update")),
                    "market_last_update": _clean_text(row.get("market_last_update")),
                    "commence_time": _clean_text(event.get("commence_time")),
                    "event_home_team": _clean_text(event.get("home_team")),
                    "event_away_team": _clean_text(event.get("away_team")),
                }
            )
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def _load_game_event_bijection() -> pd.DataFrame:
    bijection = pd.read_csv(GAME_EVENT_BIJECTION_CSV, dtype={"game_id": str, "event_id": str})
    bijection["game_id"] = bijection["game_id"].astype(str)
    bijection["event_id"] = bijection["event_id"].astype(str)
    return bijection


def score_current_points_ou() -> pd.DataFrame:
    """Score current points O/U lines with the local quantile-model artifact."""
    if not ARTIFACT_PATH.exists():
        return pd.DataFrame()

    history_df, current_players, current_teams = _load_history_state()
    rosters = _load_rosters()
    artifact = load_artifact(ARTIFACT_PATH)
    rows = _load_points_market_rows()
    if rows.empty:
        return rows

    team_id_map = _team_name_resolution(current_teams)
    unique_matchups = (
        rows[["event_id", "event_home_team", "event_away_team", "commence_time"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    prediction_frames: list[pd.DataFrame] = []
    for matchup in unique_matchups.itertuples(index=False):
        home_team_id = team_id_map.get(_clean_text(matchup.event_home_team).lower())
        away_team_id = team_id_map.get(_clean_text(matchup.event_away_team).lower())
        if not home_team_id or not away_team_id:
            continue
        event_date = pd.to_datetime(matchup.commence_time, utc=True, errors="coerce")
        if pd.isna(event_date):
            continue
        event_date = event_date.tz_convert(None)

        try:
            preds = predict_matchup(
                artifacts=artifact,
                current_players=current_players,
                current_teams=current_teams,
                history_df=history_df,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                event_date=event_date,
                is_playoff=_infer_postseason(event_date),
                roster_df=rosters,
            )
        except Exception:
            continue

        preds = preds.copy()
        preds["event_id"] = matchup.event_id
        preds["player_key"] = preds["PLAYER_NAME"].map(normalize_player_name)
        prediction_frames.append(preds)

    if not prediction_frames:
        return pd.DataFrame()

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.drop_duplicates(subset=["event_id", "player_key"], keep="first")
    merged = rows.merge(
        predictions[["event_id", "player_key", "PLAYER_ID", "TEAM_ID", "q10", "q50", "q90"]],
        on=["event_id", "player_key"],
        how="left",
    )
    merged = merged.rename(columns={"PLAYER_ID": "person_id", "TEAM_ID": "team_id"})
    merged["q10"] = pd.to_numeric(merged["q10"], errors="coerce")
    merged["q50"] = pd.to_numeric(merged["q50"], errors="coerce")
    merged["q90"] = pd.to_numeric(merged["q90"], errors="coerce")
    merged["over_price"] = pd.to_numeric(merged["over_price"], errors="coerce")
    merged["under_price"] = pd.to_numeric(merged["under_price"], errors="coerce")
    merged["commence_time"] = merged["commence_time"].astype(str)
    merged = _apply_recommendation_rule(merged)
    return merged


def score_historical_points_ou() -> pd.DataFrame:
    """Score historical points O/U lines using the split artifact's held-out predictions."""
    if not ARTIFACT_PATH.exists():
        return pd.DataFrame()

    history_df, _, _ = _load_history_state()
    artifact = load_artifact(ARTIFACT_PATH)
    evaluation = evaluate_test_set(history_df, artifact)
    predictions = evaluation.predictions.copy()
    predictions["GAME_ID"] = predictions["GAME_ID"].astype(str)
    predictions["PLAYER_KEY"] = predictions["PLAYER_NAME"].map(normalize_player_name)

    odds_rows = _load_historical_points_market_rows()
    if odds_rows.empty:
        return pd.DataFrame()

    merged = odds_rows.merge(_load_game_event_bijection(), on="event_id", how="left")
    merged = merged.rename(columns={"game_id": "GAME_ID"})
    merged["GAME_ID"] = merged["GAME_ID"].fillna("").astype(str)
    merged = merged.merge(
        predictions[
            [
                "GAME_ID",
                "PLAYER_ID",
                "PLAYER_NAME",
                "PLAYER_KEY",
                "TEAM_ID",
                "GAME_DATE",
                "actual",
                "q10",
                "q50",
                "q90",
            ]
        ],
        left_on=["GAME_ID", "player_key"],
        right_on=["GAME_ID", "PLAYER_KEY"],
        how="inner",
    )
    if merged.empty:
        return merged

    merged["line"] = pd.to_numeric(merged["line"], errors="coerce")
    merged["line_points"] = pd.to_numeric(merged["line_points"], errors="coerce")
    merged["actual"] = pd.to_numeric(merged["actual"], errors="coerce")
    merged["over_price"] = pd.to_numeric(merged["over_price"], errors="coerce")
    merged["under_price"] = pd.to_numeric(merged["under_price"], errors="coerce")
    merged["GAME_DATE"] = pd.to_datetime(merged["GAME_DATE"], errors="coerce")
    merged = _apply_recommendation_rule(merged)

    merged["actual_side_calc"] = pd.NA
    merged.loc[merged["actual"].gt(merged["line"]), "actual_side_calc"] = "over"
    merged.loc[merged["actual"].lt(merged["line"]), "actual_side_calc"] = "under"
    merged["is_correct"] = (
        merged["selection_side"].fillna("").astype(str)
        == merged["actual_side_calc"].fillna("").astype(str)
    )
    merged["pick_odds"] = np.where(
        merged["selection_side"].eq("under"),
        merged["under_price"],
        np.where(merged["selection_side"].eq("over"), merged["over_price"], np.nan),
    )
    merged["profit"] = np.where(
        merged["is_recommended"] & merged["is_correct"] & merged["pick_odds"].gt(1.0),
        merged["pick_odds"] - 1.0,
        np.where(merged["is_recommended"], -1.0, np.nan),
    )
    merged["game_date"] = merged["GAME_DATE"]
    return merged


__all__ = ["ARTIFACT_PATH", "score_current_points_ou", "score_historical_points_ou"]
