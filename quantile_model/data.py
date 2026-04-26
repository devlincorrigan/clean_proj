"""Local feature engineering for the points O/U quantile model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PLAYER_SEQUENCE_COLS: tuple[str, ...] = (
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "AST",
    "REB",
    "OREB",
    "DREB",
    "TOV",
    "STL",
    "BLK",
    "PF",
    "PLUS_MINUS",
)

RAW_TEAM_PERFORMANCE_COLS: tuple[str, ...] = (
    "PTS",
    "AST",
    "REB",
    "FGA",
    "FG3A",
    "TOV",
    "STL",
    "BLK",
    "PLUS_MINUS",
)

TEAM_LAST_GAME_COLS: tuple[str, ...] = tuple(
    f"Team_LastGame_{col}" for col in RAW_TEAM_PERFORMANCE_COLS
)

OPP_LAST_GAME_COLS: tuple[str, ...] = tuple(
    f"Opp_LastGame_{col}" for col in RAW_TEAM_PERFORMANCE_COLS
)

TEAM_INFERENCE_COLS: tuple[str, ...] = OPP_LAST_GAME_COLS

PLAYER_COLUMN_MAP = {
    "minutesFloat": "MIN",
    "fieldGoalsMade": "FGM",
    "fieldGoalsAttempted": "FGA",
    "threePointersMade": "FG3M",
    "threePointersAttempted": "FG3A",
    "freeThrowsMade": "FTM",
    "freeThrowsAttempted": "FTA",
    "assists": "AST",
    "reboundsTotal": "REB",
    "reboundsOffensive": "OREB",
    "reboundsDefensive": "DREB",
    "turnovers": "TOV",
    "steals": "STL",
    "blocks": "BLK",
    "foulsPersonal": "PF",
    "plusMinusPoints": "PLUS_MINUS",
}

TEAM_COLUMN_MAP = {
    "points": "PTS",
    "assists": "AST",
    "reboundsTotal": "REB",
    "fieldGoalsAttempted": "FGA",
    "threePointersAttempted": "FG3A",
    "turnovers": "TOV",
    "steals": "STL",
    "blocks": "BLK",
    "plusMinusPoints": "PLUS_MINUS",
}


def _read_csv(path: str | Path, *, dtype: dict[str, str]) -> pd.DataFrame:
    return pd.read_csv(path, dtype=dtype)


def _standardize_players(players_raw: pd.DataFrame) -> pd.DataFrame:
    players = players_raw.copy()
    players["GAME_ID"] = players["gameId"].astype(str)
    players["GAME_DATE"] = pd.to_datetime(players["gameDate"], errors="coerce")
    players["PLAYER_ID"] = players["personId"].astype(str)
    players["TEAM_ID"] = players["teamId"].astype(str)
    players["TEAM_CITY"] = players["teamCity"].fillna("").astype(str).str.strip()
    players["TEAM_NAME"] = players["teamName"].fillna("").astype(str).str.strip()
    players["TEAM_ABBREVIATION"] = players["teamTricode"].fillna("").astype(str).str.strip()
    players["PLAYER_NAME"] = (
        players["firstName"].fillna("").astype(str).str.strip()
        + " "
        + players["familyName"].fillna("").astype(str).str.strip()
    ).str.strip()
    players["PTS"] = pd.to_numeric(players["points"], errors="coerce")
    players["is_playoff"] = pd.to_numeric(players["isPlayoffs"], errors="coerce").fillna(0).astype(int)
    players["home"] = pd.to_numeric(players["isHome"], errors="coerce").fillna(0).astype(int)

    for source_col, target_col in PLAYER_COLUMN_MAP.items():
        players[target_col] = pd.to_numeric(players[source_col], errors="coerce")

    keep_cols = [
        "GAME_ID",
        "GAME_DATE",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_CITY",
        "TEAM_NAME",
        "TEAM_ABBREVIATION",
        "PTS",
        "is_playoff",
        "home",
        *RAW_PLAYER_SEQUENCE_COLS,
    ]
    return players[keep_cols].copy()


def _standardize_teams(teams_raw: pd.DataFrame) -> pd.DataFrame:
    teams = teams_raw.copy()
    teams["GAME_ID"] = teams["gameId"].astype(str)
    teams["GAME_DATE"] = pd.to_datetime(teams["gameDate"], errors="coerce")
    teams["TEAM_ID"] = teams["teamId"].astype(str)
    teams["TEAM_CITY"] = teams["teamCity"].fillna("").astype(str).str.strip()
    teams["TEAM_NAME"] = teams["teamName"].fillna("").astype(str).str.strip()
    teams["TEAM_ABBREVIATION"] = teams["teamTricode"].fillna("").astype(str).str.strip()

    for source_col, target_col in TEAM_COLUMN_MAP.items():
        teams[target_col] = pd.to_numeric(teams[source_col], errors="coerce")

    keep_cols = [
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "TEAM_CITY",
        "TEAM_NAME",
        "TEAM_ABBREVIATION",
        *RAW_TEAM_PERFORMANCE_COLS,
    ]
    return teams[keep_cols].copy()


def build_local_datasets(
    *,
    players_csv: str | Path,
    teams_csv: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train/inference tables from local downloaded box-score data."""
    players_raw = _read_csv(
        players_csv,
        dtype={"gameId": str, "personId": str, "teamId": str},
    )
    teams_raw = _read_csv(
        teams_csv,
        dtype={"gameId": str, "teamId": str},
    )

    players = _standardize_players(players_raw)
    teams = _standardize_teams(teams_raw)

    players = players.dropna(subset=["GAME_DATE", "PTS", "MIN"]).copy()
    players = players[players["MIN"] > 0].copy()
    players = players.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    players["days_of_rest"] = players.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days
    players["days_of_rest"] = (
        players["days_of_rest"].fillna(10).clip(lower=0, upper=10).astype(float)
    )

    teams = teams.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    for raw_col, team_col in zip(RAW_TEAM_PERFORMANCE_COLS, TEAM_LAST_GAME_COLS):
        teams[team_col] = teams.groupby("TEAM_ID")[raw_col].shift(1)

    team_profiles = teams[
        ["GAME_ID", "TEAM_ID", "TEAM_CITY", "TEAM_NAME", "TEAM_ABBREVIATION", *TEAM_LAST_GAME_COLS]
    ].copy()
    matchup_profiles = pd.merge(
        team_profiles,
        team_profiles,
        on="GAME_ID",
        suffixes=("", "_OPP"),
    )
    matchup_profiles = matchup_profiles[
        matchup_profiles["TEAM_ID"] != matchup_profiles["TEAM_ID_OPP"]
    ].copy()
    for team_col, opp_col in zip(TEAM_LAST_GAME_COLS, OPP_LAST_GAME_COLS):
        matchup_profiles[opp_col] = matchup_profiles[f"{team_col}_OPP"]

    matchup_profiles = matchup_profiles[["GAME_ID", "TEAM_ID", *TEAM_INFERENCE_COLS]].copy()
    final_df = pd.merge(players, matchup_profiles, on=["GAME_ID", "TEAM_ID"], how="left")

    training_cols = [
        "GAME_ID",
        "GAME_DATE",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "PTS",
        "is_playoff",
        "home",
        "days_of_rest",
        *RAW_PLAYER_SEQUENCE_COLS,
        *TEAM_INFERENCE_COLS,
    ]
    final_df = final_df[training_cols].copy().reset_index(drop=True)

    current_players = (
        players.groupby("PLAYER_ID", sort=False).tail(1).copy().reset_index(drop=True)
    )
    current_players = current_players[
        [
            "GAME_DATE",
            "PLAYER_ID",
            "PLAYER_NAME",
            "TEAM_ID",
            "TEAM_CITY",
            "TEAM_NAME",
            "TEAM_ABBREVIATION",
            "days_of_rest",
            *RAW_PLAYER_SEQUENCE_COLS,
        ]
    ].copy()

    current_teams = teams.groupby("TEAM_ID", sort=False).tail(1).copy().reset_index(drop=True)
    current_teams = current_teams[
        [
            "GAME_DATE",
            "TEAM_ID",
            "TEAM_CITY",
            "TEAM_NAME",
            "TEAM_ABBREVIATION",
            *TEAM_LAST_GAME_COLS,
        ]
    ].copy()

    return final_df, current_players, current_teams


__all__ = [
    "OPP_LAST_GAME_COLS",
    "RAW_PLAYER_SEQUENCE_COLS",
    "RAW_TEAM_PERFORMANCE_COLS",
    "TEAM_INFERENCE_COLS",
    "TEAM_LAST_GAME_COLS",
    "build_local_datasets",
]
