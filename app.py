#!/usr/bin/env python3

import csv
import json
import math
import os
import statistics
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Player Context", page_icon="🏀")
startup_message = st.empty()
startup_message.info(
    "Loading charts and model artifacts. If Matplotlib needs to build its font cache, "
    "the first startup can take a moment."
)

APP_DIR = Path(__file__).resolve().parent
MPL_DIR = APP_DIR / ".matplotlib"
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR.resolve()))

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import altair as alt
import pandas as pd

from multi_output_threshold_mlp.odds_processing import (
    BOOKMAKER_PRIORITY_RANK,
    dedup_player_points_alternate_rows,
    load_odds_event,
    normalize_player_name,
)
from streamlit_app.points_ou_model import score_current_points_ou, score_historical_points_ou
from streamlit_app.score_live_odds import score_current_points_ladders

PLAYERS_CSV = APP_DIR / "data" / "box_scores" / "players.csv"
NBA_ROSTERS_CSV = APP_DIR / "data" / "rosters.csv"
CURRENT_PLAYER_POINTS_EVENTS_JSON = (
    APP_DIR / "data" / "current_odds_api" / "player_points" / "events.json"
)
CURRENT_PLAYER_POINTS_DIR = APP_DIR / "data" / "current_odds_api" / "player_points"
CURRENT_PLAYER_POINTS_ALTERNATE_EVENTS_JSON = (
    APP_DIR / "data" / "current_odds_api" / "player_points_alternate" / "events.json"
)
NBA_TEAMS_CSV = APP_DIR / "data" / "teams.csv"
WALK_FORWARD_PERIOD_SUMMARY_CSV = (
    APP_DIR / "multi_output_threshold_mlp" / "walk_forward_stacked_period_summary.csv"
)
WALK_FORWARD_SELECTED_BETS_CSV = (
    APP_DIR / "multi_output_threshold_mlp" / "walk_forward_stacked_selected_bets.csv"
)
RECENT_GAMES_WINDOW = 80
FIGURE_SIZE = (14, 5.5)
FIGURE_SUBPLOT_WSPACE = 0.2
HISTOGRAM_LABEL_SPACING = 3
HISTOGRAM_LEFT_PADDING = 0.65
MINUTES_BIN_WIDTH = 4
CHART_FIGURE_BG = "#111827"
CHART_AXES_BG = "#1f2937"
CHART_TEXT = "#e5e7eb"
CHART_GRID = "#4b5563"
PAGE_OPTIONS = ("Player Analytics", "Points O/U", "Points Alternate")
ROLLING_MEDIAN_WINDOW = 15
STAT_OPTIONS = {
    "Points": "points",
    "Minutes": "minutes",
    "Assists": "assists",
    "Rebounds": "reboundsTotal",
    "Steals": "steals",
    "Blocks": "blocks",
    "Turnovers": "turnovers",
    "3-Pointers Made": "threePointersMade",
    "Field Goals Made": "fieldGoalsMade",
    "Free Throws Made": "freeThrowsMade",
}
ROLLING_SUMMARY_STATS = (
    ("MIN", "minutes"),
    ("PTS", "points"),
    ("REB", "reboundsTotal"),
    ("AST", "assists"),
    ("STL", "steals"),
    ("BLK", "blocks"),
    ("TOV", "turnovers"),
    ("3PM", "threePointersMade"),
)
plt.style.use("ggplot")
startup_message.empty()


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def parse_minutes_float(value):
    text = clean_text(value)
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def parse_stat_value(value):
    text = clean_text(value)
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


@st.cache_data
def load_players():
    players_by_id = {}

    with NBA_ROSTERS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {NBA_ROSTERS_CSV}")

        for row in reader:
            person_id = clean_text(row.get("PLAYER_ID"))
            if not person_id or person_id in players_by_id:
                continue

            full_name = clean_text(row.get("PLAYER"))

            players_by_id[person_id] = {
                "personId": person_id,
                "fullName": full_name,
                "teamName": clean_text(row.get("TEAM_NAME")),
                "position": clean_text(row.get("POSITION")),
                "height": clean_text(row.get("HEIGHT")),
                "weight": clean_text(row.get("WEIGHT")),
                "age": clean_text(row.get("AGE")),
                "experience": clean_text(row.get("EXP")),
                "school": clean_text(row.get("SCHOOL")),
            }

    return sorted(
        players_by_id.values(),
        key=lambda player: (player["fullName"].casefold(), player["personId"]),
    )


@st.cache_data
def load_player_games(person_id):
    games = []

    with PLAYERS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {PLAYERS_CSV}")

        for row in reader:
            if clean_text(row.get("personId")) != person_id:
                continue

            minutes_float = parse_minutes_float(row.get("minutesFloat"))
            if minutes_float is None or minutes_float <= 0:
                continue

            games.append(
                {
                    "gameDate": clean_text(row.get("gameDate")),
                    "gameId": clean_text(row.get("gameId")),
                    "minutes": minutes_float,
                    "points": parse_stat_value(row.get("points")),
                    "assists": parse_stat_value(row.get("assists")),
                    "reboundsTotal": parse_stat_value(row.get("reboundsTotal")),
                    "steals": parse_stat_value(row.get("steals")),
                    "blocks": parse_stat_value(row.get("blocks")),
                    "turnovers": parse_stat_value(row.get("turnovers")),
                    "threePointersMade": parse_stat_value(row.get("threePointersMade")),
                    "fieldGoalsMade": parse_stat_value(row.get("fieldGoalsMade")),
                    "freeThrowsMade": parse_stat_value(row.get("freeThrowsMade")),
                }
            )

    return sorted(games, key=lambda game: (game["gameDate"], game["gameId"]))


@st.cache_data
def load_matchup_rows(events_json_path):
    with Path(events_json_path).open(encoding="utf-8") as handle:
        events = json.load(handle)

    matchup_rows = [
        {
            "home_team": event.get("home_team", ""),
            "away_team": event.get("away_team", ""),
            "commence_time": event.get("commence_time", ""),
        }
        for event in events
    ]
    return sorted(matchup_rows, key=lambda row: row["commence_time"])


@st.cache_data(show_spinner=False)
def load_live_points_ladder_scores():
    return score_current_points_ladders()


@st.cache_data
def load_live_points_ou_scores():
    return score_current_points_ou()


@st.cache_data(show_spinner=False)
def load_historical_points_ou_scores():
    return score_historical_points_ou()


@st.cache_data
def load_current_points_ou_rows():
    rows = []
    for event_file in sorted(CURRENT_PLAYER_POINTS_DIR.glob("*.json")):
        if event_file.name == "events.json":
            continue

        with event_file.open(encoding="utf-8") as handle:
            event = json.load(handle)

        event_id = clean_text(event.get("id"))
        if not event_id:
            continue

        grouped = {}
        for bookmaker in event.get("bookmakers", []) or []:
            bookmaker_key = clean_text(bookmaker.get("key"))
            bookmaker_title = clean_text(bookmaker.get("title"))
            bookmaker_rank = bookmaker_priority_rank(bookmaker_key)
            for market in bookmaker.get("markets", []) or []:
                if clean_text(market.get("key")) != "player_points":
                    continue
                for outcome in market.get("outcomes", []) or []:
                    side_name = clean_text(outcome.get("name")).lower()
                    if side_name not in {"over", "under"}:
                        continue
                    player_name = clean_text(outcome.get("description"))
                    if not player_name:
                        continue
                    try:
                        line_points = float(outcome.get("point"))
                        price = float(outcome.get("price"))
                    except (TypeError, ValueError):
                        continue

                    player_key = normalize_player_name(player_name)
                    group_key = (event_id, player_key, line_points)
                    if group_key not in grouped:
                        grouped[group_key] = {
                            "event_id": event_id,
                            "player_name_odds": player_name,
                            "line_points": line_points,
                            "event_home_team": clean_text(event.get("home_team")),
                            "event_away_team": clean_text(event.get("away_team")),
                            "commence_time": clean_text(event.get("commence_time")),
                            "bookmakers_by_key": {},
                            "over_price": pd.NA,
                            "under_price": pd.NA,
                            "bookmaker_key": "",
                            "bookmaker_title": "",
                        }

                    group = grouped[group_key]
                    group["bookmakers_by_key"][bookmaker_key] = bookmaker_title or bookmaker_key

                    current_side_rank = group.get(f"{side_name}_bookmaker_rank")
                    current_side_price = pd.to_numeric(
                        pd.Series([group.get(f"{side_name}_price")]), errors="coerce"
                    ).iloc[0]
                    should_replace = current_side_rank is None or bookmaker_rank < current_side_rank
                    if (
                        current_side_rank is not None
                        and bookmaker_rank == current_side_rank
                        and (pd.isna(current_side_price) or price > current_side_price)
                    ):
                        should_replace = True

                    if should_replace:
                        group[f"{side_name}_price"] = price
                        group[f"{side_name}_bookmaker_key"] = bookmaker_key
                        group[f"{side_name}_bookmaker_title"] = bookmaker_title
                        group[f"{side_name}_bookmaker_rank"] = bookmaker_rank

                    canonical_rank = group.get("bookmaker_rank")
                    if canonical_rank is None or bookmaker_rank < canonical_rank:
                        group["bookmaker_rank"] = bookmaker_rank
                        group["bookmaker_key"] = bookmaker_key
                        group["bookmaker_title"] = bookmaker_title

        for group in grouped.values():
            ranked_books = sorted(
                group["bookmakers_by_key"].items(),
                key=lambda item: (bookmaker_priority_rank(item[0]), item[1].casefold()),
            )
            book_titles = [title for _, title in ranked_books if clean_text(title)]
            rows.append(
                {
                    "event_id": group["event_id"],
                    "person_id": "",
                    "team_id": "",
                    "player_name_odds": group["player_name_odds"],
                    "line_points": group["line_points"],
                    "bet_side": "",
                    "bet_price": pd.NA,
                    "bet_price_numeric": pd.NA,
                    "trust_blend_score": pd.NA,
                    "trust_win_prob": pd.NA,
                    "trust_bet_ev": pd.NA,
                    "breakeven_probability": pd.NA,
                    "probability_edge": pd.NA,
                    "model_bet_probability": pd.NA,
                    "bookmaker_key": group.get("bookmaker_key", ""),
                    "bookmaker_title": group.get("bookmaker_title", ""),
                    "all_bookmaker_titles": ", ".join(book_titles),
                    "bookmaker_count": float(len(book_titles)),
                    "over_price": group.get("over_price", group.get("over_price", pd.NA)),
                    "under_price": group.get("under_price", group.get("under_price", pd.NA)),
                    "event_home_team": group["event_home_team"],
                    "event_away_team": group["event_away_team"],
                    "commence_time": group["commence_time"],
                }
            )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["line_points"] = pd.to_numeric(frame["line_points"], errors="coerce")
    frame["commence_sort"] = pd.to_datetime(frame["commence_time"], utc=True, errors="coerce")
    return frame.sort_values(
        ["commence_sort", "event_away_team", "event_home_team", "player_name_odds", "line_points"],
        ascending=[True, True, True, True, True],
    ).drop(columns=["commence_sort"], errors="ignore").reset_index(drop=True)


@st.cache_data
def load_walk_forward_period_summary():
    if not WALK_FORWARD_PERIOD_SUMMARY_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(WALK_FORWARD_PERIOD_SUMMARY_CSV)


@st.cache_data
def load_walk_forward_selected_bets():
    if not WALK_FORWARD_SELECTED_BETS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(WALK_FORWARD_SELECTED_BETS_CSV)


@st.cache_data
def load_team_names_by_id():
    mapping = {}
    with NBA_ROSTERS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {NBA_ROSTERS_CSV}")
        for row in reader:
            team_id = clean_text(row.get("TEAM_ID") or row.get("TeamID"))
            team_name = clean_text(row.get("TEAM_NAME"))
            if team_id and team_name and team_id not in mapping:
                mapping[team_id] = team_name
    return mapping


@st.cache_data
def load_team_abbreviations():
    by_full_name = {}
    by_nickname = {}
    with NBA_TEAMS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {NBA_TEAMS_CSV}")
        for row in reader:
            full_name = clean_text(row.get("full_name"))
            nickname = clean_text(row.get("nickname"))
            abbreviation = clean_text(row.get("abbreviation"))
            if full_name and abbreviation:
                by_full_name[full_name] = abbreviation
            if nickname and abbreviation:
                by_nickname[nickname] = abbreviation
    return by_full_name, by_nickname


@st.cache_data
def load_current_ladder_book_metadata():
    metadata = {}
    odds_dir = APP_DIR / "data" / "current_odds_api" / "player_points_alternate"
    for path in sorted(odds_dir.glob("*.json")):
        if path.name == "events.json":
            continue
        event = load_odds_event(path)
        event_id = clean_text(event.get("id"))
        if not event_id:
            continue
        for row in dedup_player_points_alternate_rows(event):
            player_key = normalize_player_name(clean_text(row.get("player_name_odds")))
            if not player_key:
                continue
            try:
                line_points = float(row.get("line_points"))
            except (TypeError, ValueError):
                continue
            metadata[(event_id, player_key, line_points)] = {
                "all_bookmaker_titles": clean_text(row.get("all_bookmaker_titles")),
                "bookmaker_count": row.get("bookmaker_count"),
            }
    return metadata


def bookmaker_priority_rank(bookmaker_key):
    normalized = clean_text(bookmaker_key)
    return BOOKMAKER_PRIORITY_RANK.get(normalized, len(BOOKMAKER_PRIORITY_RANK))


def format_commence_time_local(value):
    if not clean_text(value):
        return "TBD"
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return clean_text(value)
    local = timestamp.tz_convert("America/New_York")
    return local.strftime("%a %b %-d, %-I:%M %p ET")


def dedupe_ladder_bets(rows):
    if rows.empty:
        return rows.copy()

    working = rows.copy()
    working["bookmaker_priority_rank"] = working["bookmaker_key"].map(bookmaker_priority_rank)
    working["line_points_numeric"] = pd.to_numeric(working["line_points"], errors="coerce")
    working["bet_price_numeric"] = pd.to_numeric(working["bet_price"], errors="coerce")
    working["trust_blend_score_numeric"] = pd.to_numeric(
        working["trust_blend_score"], errors="coerce"
    )
    working = working.sort_values(
        [
            "event_id",
            "person_id",
            "line_points_numeric",
            "bet_side",
            "bookmaker_priority_rank",
            "trust_blend_score_numeric",
            "bet_price_numeric",
        ],
        ascending=[True, True, True, True, True, False, False],
    )
    deduped = working.drop_duplicates(
        subset=["event_id", "person_id", "line_points_numeric", "bet_side"],
        keep="first",
    ).copy()
    return deduped.drop(
        columns=[
            "bookmaker_priority_rank",
            "line_points_numeric",
            "bet_price_numeric",
            "trust_blend_score_numeric",
        ],
        errors="ignore",
    )


def add_ladder_probability_columns(rows):
    if rows.empty:
        enriched = rows.copy()
        enriched["model_bet_probability"] = pd.Series(dtype="float64")
        enriched["breakeven_probability"] = pd.Series(dtype="float64")
        enriched["probability_edge"] = pd.Series(dtype="float64")
        return enriched

    enriched = rows.copy()
    enriched["bet_price_numeric"] = pd.to_numeric(enriched["bet_price"], errors="coerce")
    enriched["model_over_prob_numeric"] = pd.to_numeric(
        enriched["model_over_prob"], errors="coerce"
    )
    enriched["model_under_prob_numeric"] = pd.to_numeric(
        enriched["model_under_prob"], errors="coerce"
    )
    enriched["bet_side_normalized"] = enriched["bet_side"].fillna("").astype(str).str.lower()
    enriched["model_bet_probability"] = enriched["model_over_prob_numeric"].where(
        enriched["bet_side_normalized"] == "over",
        enriched["model_under_prob_numeric"],
    )
    enriched["breakeven_probability"] = 1.0 / enriched["bet_price_numeric"]
    enriched.loc[enriched["bet_price_numeric"] <= 0, "breakeven_probability"] = pd.NA
    enriched["probability_edge"] = (
        enriched["model_bet_probability"] - enriched["breakeven_probability"]
    )
    return enriched.drop(
        columns=[
            "bet_price_numeric",
            "model_over_prob_numeric",
            "model_under_prob_numeric",
            "bet_side_normalized",
        ],
        errors="ignore",
    )


def format_ev_multiple(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "N/A"
    return f"{1.0 + numeric:.2f}"


def format_decimal_price(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "N/A"
    return f"{numeric:.2f}"


def format_score_value(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "N/A"
    return f"{numeric:.3f}"


def format_test_window_label(date_series) -> str:
    dates = pd.to_datetime(pd.Series(date_series), errors="coerce").dropna()
    if dates.empty:
        return "N/A"
    start = dates.min().strftime("%b %-d, %Y")
    end = dates.max().strftime("%b %-d, %Y")
    return f"{start} to {end}"


def compact_book_name(book_name):
    text = clean_text(book_name)
    compact_map = {
        "BetOnline.ag": "BetOnline",
        "BetOnline": "BetOnline",
        "William Hill US": "Caesars",
        "Unibet US": "Unibet",
        "PointsBet US": "PointsBet",
    }
    return compact_map.get(text, text)


def compact_books_label(books_label, *, keep=2):
    books = [clean_text(part) for part in str(books_label or "").split(",")]
    books = [book for book in books if book]
    if not books:
        return ""
    books = [compact_book_name(book) for book in books]
    if len(books) <= keep:
        return ", ".join(books)
    shown = ", ".join(books[:keep])
    return f"{shown}\u00A0+{len(books) - keep}"


def filter_rows_by_bookmaker(rows: pd.DataFrame, bookmaker_title: str) -> pd.DataFrame:
    if rows.empty or "all_bookmaker_titles" not in rows.columns:
        return rows.copy()

    selected = clean_text(bookmaker_title).casefold()
    if not selected:
        return rows.copy()

    def has_bookmaker(value) -> bool:
        books = [clean_text(part).casefold() for part in str(value or "").split(",")]
        books = [book for book in books if book]
        return selected in books

    mask = rows["all_bookmaker_titles"].map(has_bookmaker)
    return rows[mask.fillna(False)].copy()


def render_card_stat(label, value):
    st.markdown(f"**{label}**  \n{value}")


def score_tier_style(score_value):
    numeric = pd.to_numeric(pd.Series([score_value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ("#9ca3af", "#f3f4f6")
    if numeric > 0.20:
        return ("#7dd3fc", "#e0f2fe")
    if numeric > 0.15:
        return ("#86efac", "#ecfdf5")
    if numeric >= 0.10:
        return ("#fde68a", "#fffbeb")
    return ("#ef4444", "#fef2f2")


def attach_current_book_metadata(rows):
    if rows.empty:
        return rows.copy()

    metadata = load_current_ladder_book_metadata()
    enriched = rows.copy()
    event_ids = enriched["event_id"].fillna("").astype(str)
    player_keys = enriched["player_name_odds"].fillna("").astype(str).map(normalize_player_name)
    line_values = pd.to_numeric(enriched["line_points"], errors="coerce")
    matched = [
        metadata.get((event_id, player_key, line_value), {})
        if pd.notna(line_value)
        else {}
        for event_id, player_key, line_value in zip(event_ids, player_keys, line_values)
    ]

    existing_titles = (
        enriched["all_bookmaker_titles"].fillna("").astype(str)
        if "all_bookmaker_titles" in enriched.columns
        else pd.Series([""] * len(enriched), index=enriched.index)
    )
    fallback_titles = pd.Series(
        [clean_text(item.get("all_bookmaker_titles")) for item in matched],
        index=enriched.index,
        dtype="string",
    ).fillna("")
    enriched["all_bookmaker_titles"] = existing_titles.mask(existing_titles.eq(""), fallback_titles)

    existing_counts = (
        pd.to_numeric(enriched["bookmaker_count"], errors="coerce")
        if "bookmaker_count" in enriched.columns
        else pd.Series([float("nan")] * len(enriched), index=enriched.index, dtype="float64")
    )
    fallback_counts = pd.Series(
        [item.get("bookmaker_count") for item in matched],
        index=enriched.index,
        dtype="float64",
    )
    enriched["bookmaker_count"] = existing_counts.fillna(fallback_counts)
    return enriched


def team_abbreviation(name, full_name_to_abbrev, nickname_to_abbrev):
    cleaned = clean_text(name)
    if not cleaned:
        return "TBD"
    return (
        full_name_to_abbrev.get(cleaned)
        or nickname_to_abbrev.get(cleaned)
        or cleaned
    )


def card_game_label(row, full_name_to_abbrev, nickname_to_abbrev, team_names_by_id):
    away_team = team_abbreviation(
        row.get("event_away_team"),
        full_name_to_abbrev,
        nickname_to_abbrev,
    )
    home_team = team_abbreviation(
        row.get("event_home_team"),
        full_name_to_abbrev,
        nickname_to_abbrev,
    )
    team_id = clean_text(row.get("team_id"))
    team_name = clean_text(team_names_by_id.get(team_id, ""))
    team_abbrev = team_abbreviation(team_name, full_name_to_abbrev, nickname_to_abbrev)
    if team_abbrev == home_team:
        return f"{team_abbrev} vs {away_team}"
    if team_abbrev == away_team:
        return f"{team_abbrev} at {home_team}"
    return f"{away_team} at {home_team}"


def neutral_game_label(row, full_name_to_abbrev, nickname_to_abbrev):
    away_team = team_abbreviation(
        row.get("event_away_team"),
        full_name_to_abbrev,
        nickname_to_abbrev,
    )
    home_team = team_abbreviation(
        row.get("event_home_team"),
        full_name_to_abbrev,
        nickname_to_abbrev,
    )
    return f"{away_team} at {home_team}"


def build_game_browser_frame(rows):
    if rows.empty:
        return pd.DataFrame(
            columns=["event_id", "game_label", "commence_time", "time_et", "line_count"]
        )

    full_name_to_abbrev, nickname_to_abbrev = load_team_abbreviations()
    working = rows.copy()
    working["commence_sort"] = pd.to_datetime(
        working["commence_time"], utc=True, errors="coerce"
    )
    working["game_label"] = working.apply(
        lambda row: neutral_game_label(row, full_name_to_abbrev, nickname_to_abbrev),
        axis=1,
    )
    summary = (
        working.groupby(
            ["event_id", "game_label", "commence_time", "commence_sort"],
            dropna=False,
            sort=False,
        )
        .size()
        .reset_index(name="line_count")
        .sort_values(["commence_sort", "game_label"], ascending=[True, True])
        .reset_index(drop=True)
    )
    summary["time_et"] = summary["commence_time"].map(format_commence_time_local)
    return summary


def build_game_line_table(game_rows):
    if game_rows.empty:
        return pd.DataFrame()

    table = game_rows.copy()

    def format_table_price(value):
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return ""
        return f"{numeric:.2f}"

    table["over_price_display"] = table["over_price"].map(format_table_price)
    table["under_price_display"] = table["under_price"].map(format_table_price)
    table["trust_ev_display"] = table["trust_bet_ev"].map(format_ev_multiple)
    table["trust_win_prob_display"] = pd.to_numeric(
        table["trust_win_prob"], errors="coerce"
    ).map(lambda value: f"{value:.1%}" if pd.notna(value) else "")
    table["recommendation"] = table["bet_side"].fillna("").astype(str).str.upper()
    if "all_bookmaker_titles" in table.columns:
        table["books_display"] = table["all_bookmaker_titles"].fillna("").astype(str)
    else:
        table["books_display"] = ""
    if "bookmaker_title" in table.columns:
        fallback_books = table["bookmaker_title"].fillna("").astype(str)
        table["books_display"] = table["books_display"].mask(
            table["books_display"].eq(""),
            fallback_books,
        )
    table = table.sort_values(
        ["player_name_odds", "line_points"],
        ascending=[True, True],
    )
    return table[
        [
            "player_name_odds",
            "line_points",
            "over_price_display",
            "under_price_display",
            "recommendation",
            "trust_win_prob_display",
            "trust_ev_display",
            "books_display",
        ]
    ].rename(
        columns={
            "player_name_odds": "player",
            "line_points": "line",
            "over_price_display": "over",
            "under_price_display": "under",
            "trust_win_prob_display": "trust_win_pct",
            "trust_ev_display": "trust_ev",
            "books_display": "books",
        }
    )


def expand_rows_for_card_display(rows):
    if rows.empty:
        return rows.copy()

    expanded = []
    for row in rows.to_dict("records"):
        existing_side = clean_text(row.get("bet_side")).lower()
        if existing_side in {"over", "under"}:
            expanded.append(row)
            continue

        emitted = False
        for side_name, price_key in [("over", "over_price"), ("under", "under_price")]:
            side_price = pd.to_numeric(pd.Series([row.get(price_key)]), errors="coerce").iloc[0]
            if pd.isna(side_price):
                continue
            card_row = dict(row)
            card_row["bet_side"] = side_name
            card_row["bet_price"] = side_price
            card_row["breakeven_probability"] = 1.0 / side_price if side_price > 0 else pd.NA
            expanded.append(card_row)
            emitted = True

        if not emitted:
            expanded.append(row)

    return pd.DataFrame(expanded)


def render_ladder_cards(rows, *, empty_message, sort_mode="best", cards_per_row=3):
    if rows.empty:
        st.info(empty_message)
        return

    ordered = expand_rows_for_card_display(rows)
    ordered["commence_sort"] = pd.to_datetime(
        ordered["commence_time"], utc=True, errors="coerce"
    )
    ordered["trust_blend_score_numeric"] = pd.to_numeric(
        ordered["trust_blend_score"], errors="coerce"
    )
    ordered["bet_price_numeric"] = pd.to_numeric(ordered["bet_price"], errors="coerce")
    ordered["trust_bet_ev_numeric"] = pd.to_numeric(ordered["trust_bet_ev"], errors="coerce")
    if sort_mode == "best":
        ordered = ordered.sort_values(
            ["trust_blend_score_numeric", "trust_bet_ev_numeric", "commence_sort"],
            ascending=[False, False, True],
        )
    elif sort_mode == "player":
        ordered = ordered.sort_values(
            ["player_name_odds", "line_points", "trust_blend_score_numeric"],
            ascending=[True, True, False],
        )
    else:
        ordered = ordered.sort_values(
            ["commence_sort", "event_away_team", "event_home_team", "trust_blend_score_numeric"],
            ascending=[True, True, True, False],
        )

    full_name_to_abbrev, nickname_to_abbrev = load_team_abbreviations()
    team_names_by_id = load_team_names_by_id()
    records = ordered.to_dict("records")
    for start in range(0, len(records), cards_per_row):
        columns = st.columns(cards_per_row)
        chunk = records[start : start + cards_per_row]
        for column, row in zip(columns, chunk):
            side = clean_text(row.get("bet_side")).upper()
            line_points = row.get("line_points")
            player_name = clean_text(row.get("player_name_odds")) or "Unknown player"
            breakeven_probability = pd.to_numeric(
                pd.Series([row.get("breakeven_probability")]), errors="coerce"
            ).iloc[0]
            trust_win_prob = pd.to_numeric(
                pd.Series([row.get("trust_win_prob")]), errors="coerce"
            ).iloc[0]
            blend_score = pd.to_numeric(
                pd.Series([row.get("trust_blend_score")]), errors="coerce"
            ).iloc[0]
            bet_price = pd.to_numeric(pd.Series([row.get("bet_price")]), errors="coerce").iloc[0]
            over_price = pd.to_numeric(pd.Series([row.get("over_price")]), errors="coerce").iloc[0]
            under_price = pd.to_numeric(pd.Series([row.get("under_price")]), errors="coerce").iloc[0]
            all_bookmaker_titles = clean_text(row.get("all_bookmaker_titles"))
            if all_bookmaker_titles.lower() == "nan":
                all_bookmaker_titles = ""
            bookmaker_title = clean_text(row.get("bookmaker_title")) or "Unknown bookmaker"
            if not all_bookmaker_titles:
                all_bookmaker_titles = bookmaker_title
            compact_books = compact_books_label(all_bookmaker_titles)
            side_label = f"{side[:1]} {line_points}" if side else f"Line {line_points}"
            game_label = card_game_label(
                row,
                full_name_to_abbrev,
                nickname_to_abbrev,
                team_names_by_id,
            )
            commence_label = format_commence_time_local(row.get("commence_time"))
            accent_color, accent_background = score_tier_style(row.get("trust_blend_score"))
            display_price = bet_price
            if pd.isna(display_price):
                if side == "OVER" and pd.notna(over_price):
                    display_price = over_price
                elif side == "UNDER" and pd.notna(under_price):
                    display_price = under_price
            display_breakeven = (
                1.0 / display_price if pd.notna(display_price) and display_price > 0 else pd.NA
            )

            with column:
                with st.container(border=True):
                    st.markdown(
                        (
                            "<div style='height:0.45rem;"
                            f"background:{accent_color};"
                            f"border-radius:999px;"
                            f"margin-bottom:0.45rem;"
                            f"box-shadow: inset 0 0 0 1px {accent_background};"
                            "'></div>"
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{player_name}**  \n**{side_label}**")
                    st.caption(
                        f"{game_label}  \n{commence_label}  \nBooks: {compact_books or all_bookmaker_titles}"
                    )

                    top_left, top_right = st.columns(2)
                    with top_left:
                        render_card_stat("Price", format_decimal_price(display_price))
                    with top_right:
                        render_card_stat(
                            "BE %",
                            (
                                f"{(breakeven_probability if side else display_breakeven):.1%}"
                                if pd.notna(breakeven_probability if side else display_breakeven)
                                else "N/A"
                            ),
                        )

                    if pd.notna(trust_win_prob) or pd.notna(blend_score):
                        trust_left, trust_right = st.columns(2)
                        with trust_left:
                            render_card_stat(
                                "Trust %",
                                f"{trust_win_prob:.1%}" if pd.notna(trust_win_prob) else "N/A",
                            )
                        with trust_right:
                            render_card_stat("Blend Score", format_score_value(blend_score))
                    else:
                        render_card_stat("Recommendation", "No pick")


def render_points_ou_cards(rows, *, empty_message, sort_mode="best", cards_per_row=3):
    if rows.empty:
        st.info(empty_message)
        return

    ordered = rows.copy()
    ordered["commence_sort"] = pd.to_datetime(
        ordered["commence_time"], utc=True, errors="coerce"
    )
    ordered["line_points_numeric"] = pd.to_numeric(ordered["line_points"], errors="coerce")
    ordered["player_name_sort"] = ordered["player_name_odds"].fillna("").astype(str)
    ordered["recommendation_rank"] = (
        ordered["is_recommended"].fillna(False).astype(int)
        if "is_recommended" in ordered.columns
        else 0
    )
    ordered["selection_ratio_numeric"] = pd.to_numeric(
        ordered.get("selection_ratio"),
        errors="coerce",
    )
    if sort_mode == "player":
        ordered = ordered.sort_values(
            ["player_name_sort", "line_points_numeric", "commence_sort"],
            ascending=[True, True, True],
        )
    else:
        ordered = ordered.sort_values(
            [
                "recommendation_rank",
                "selection_ratio_numeric",
                "commence_sort",
                "event_away_team",
                "event_home_team",
                "player_name_sort",
                "line_points_numeric",
            ],
            ascending=[False, False, True, True, True, True, True],
        )

    full_name_to_abbrev, nickname_to_abbrev = load_team_abbreviations()
    team_names_by_id = load_team_names_by_id()
    records = ordered.to_dict("records")
    for start in range(0, len(records), cards_per_row):
        columns = st.columns(cards_per_row)
        chunk = records[start : start + cards_per_row]
        for column, row in zip(columns, chunk):
            line_points = row.get("line_points")
            player_name = clean_text(row.get("player_name_odds")) or "Unknown player"
            over_price = pd.to_numeric(pd.Series([row.get("over_price")]), errors="coerce").iloc[0]
            under_price = pd.to_numeric(pd.Series([row.get("under_price")]), errors="coerce").iloc[0]
            q10 = pd.to_numeric(pd.Series([row.get("q10")]), errors="coerce").iloc[0]
            q50 = pd.to_numeric(pd.Series([row.get("q50")]), errors="coerce").iloc[0]
            q90 = pd.to_numeric(pd.Series([row.get("q90")]), errors="coerce").iloc[0]
            selection_side = clean_text(row.get("selection_side")).lower()
            is_recommended = bool(row.get("is_recommended"))
            all_bookmaker_titles = clean_text(row.get("all_bookmaker_titles"))
            if all_bookmaker_titles.lower() == "nan":
                all_bookmaker_titles = ""
            bookmaker_title = clean_text(row.get("bookmaker_title")) or "Unknown bookmaker"
            if not all_bookmaker_titles:
                all_bookmaker_titles = bookmaker_title
            compact_books = compact_books_label(all_bookmaker_titles)
            accent_color = "#22c55e" if is_recommended else "#ef4444"
            game_label = card_game_label(
                row,
                full_name_to_abbrev,
                nickname_to_abbrev,
                team_names_by_id,
            )
            commence_label = format_commence_time_local(row.get("commence_time"))

            with column:
                with st.container(border=True):
                    st.markdown(
                        (
                            "<div style='height:0.45rem;"
                            f"background:{accent_color};"
                            "border-radius:999px;"
                            "margin-bottom:0.45rem;"
                            "box-shadow: inset 0 0 0 1px #f3f4f6;"
                            "'></div>"
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{player_name}**  \n**O/U {line_points}**")
                    st.caption(
                        f"{game_label}  \n{commence_label}  \nBooks: {compact_books or all_bookmaker_titles}"
                    )

                    price_left, price_right = st.columns(2)
                    with price_left:
                        render_card_stat("Over", format_decimal_price(over_price))
                    with price_right:
                        render_card_stat("Under", format_decimal_price(under_price))

                    if pd.notna(q50):
                        q_left, q_mid, q_right = st.columns(3)
                        with q_left:
                            render_card_stat("Q10", f"{q10:.1f}" if pd.notna(q10) else "N/A")
                        with q_mid:
                            render_card_stat("Q50", f"{q50:.1f}" if pd.notna(q50) else "N/A")
                        with q_right:
                            render_card_stat("Q90", f"{q90:.1f}" if pd.notna(q90) else "N/A")

                    recommendation_label = "N/A"
                    if selection_side:
                        recommendation_label = (
                            "Under" if selection_side == "under" else selection_side.title()
                        )
                    render_card_stat("Recommendation", recommendation_label)


def format_player_option(player):
    return player["fullName"] or player["personId"]


def render_player_selector(players):
    selected_player = st.selectbox(
        "Player",
        players,
        format_func=format_player_option,
        index=None,
        placeholder="Select a player",
    )
    if selected_player is None:
        return None

    selected_person_id = selected_player.get("personId")
    for player in players:
        if player["personId"] == selected_person_id:
            return player

    return selected_player


def format_player_detail(value):
    return value if clean_text(value) else "N/A"


def render_player_profile(player):
    info_columns = st.columns(3)
    player_details = [
        ("Team", player.get("teamName")),
        ("Position", player.get("position")),
        ("Height", player.get("height")),
        ("Weight", player.get("weight")),
        ("Age", player.get("age")),
        ("Experience", player.get("experience")),
    ]

    for index, (label, value) in enumerate(player_details):
        with info_columns[index % len(info_columns)]:
            st.markdown(f"**{label}:** {format_player_detail(value)}")


def build_rolling_median_row(games):
    row = {}
    for label, stat_key in ROLLING_SUMMARY_STATS:
        stat_values = [game[stat_key] for game in games]
        row[label] = round(statistics.median(stat_values), 1)
    return row


def render_rolling_median_table(selected_games):
    rolling_games = selected_games[-ROLLING_MEDIAN_WINDOW:]

    st.divider()
    st.subheader("Median stats (last 15 games)")
    if len(rolling_games) < ROLLING_MEDIAN_WINDOW:
        st.caption(f"Showing {len(rolling_games)} available games.")
    rolling_median_df = pd.DataFrame([build_rolling_median_row(rolling_games)])
    st.dataframe(rolling_median_df, hide_index=True, use_container_width=True)


def render_stats_controls(selected_games):
    stat_col, lookback_col = st.columns([2, 1])

    with stat_col:
        selected_stat_label = st.selectbox("Stat Category", list(STAT_OPTIONS.keys()))

    max_games = len(selected_games) if selected_games else 1
    with lookback_col:
        recent_games_limit = st.number_input(
            "Lookback Window",
            min_value=1,
            max_value=max_games,
            value=min(RECENT_GAMES_WINDOW, max_games),
            step=1,
            disabled=not selected_games,
        )

    return selected_stat_label, int(recent_games_limit)


def build_histogram_bins(min_points, max_points):
    return [point - 0.5 for point in range(min_points, max_points + 2)]


def build_histogram_ticks(min_points, max_points):
    tick_start = (min_points // HISTOGRAM_LABEL_SPACING) * HISTOGRAM_LABEL_SPACING
    tick_end = (
        (max_points + HISTOGRAM_LABEL_SPACING - 1)
        // HISTOGRAM_LABEL_SPACING
        * HISTOGRAM_LABEL_SPACING
    )
    return list(range(tick_start, tick_end + 1, HISTOGRAM_LABEL_SPACING))


def stat_values_are_integer_like(stat_values):
    return all(float(value).is_integer() for value in stat_values)


def build_continuous_histogram_bins(stat_values):
    min_value = min(stat_values)
    max_value = max(stat_values)
    if min_value == max_value:
        return [min_value - 0.5, max_value + 0.5]

    bin_count = min(12, max(6, math.ceil(math.sqrt(len(stat_values)))))
    bin_width = (max_value - min_value) / bin_count
    return [min_value + (index * bin_width) for index in range(bin_count + 1)]


def build_fixed_width_bins(stat_values, bin_width):
    min_value = min(stat_values)
    max_value = max(stat_values)
    bin_start = math.floor(min_value / bin_width) * bin_width
    bin_end = math.ceil(max_value / bin_width) * bin_width

    if bin_start == bin_end:
        bin_end += bin_width

    return list(range(int(bin_start), int(bin_end) + bin_width, bin_width))


def summarize_stat(games, stat_key):
    stat_values = [game[stat_key] for game in games]
    median_value = statistics.median(stat_values)
    return {
        "games_available": len(stat_values),
        "mean_value": sum(stat_values) / len(stat_values),
        "median_value": median_value,
        "median_bin": (
            statistics.median_low(stat_values)
            if stat_values_are_integer_like(stat_values)
            else median_value
        ),
    }


def build_stat_figure(games, stat_key, stat_label, summary):
    game_numbers = list(range(1, len(games) + 1))
    stat_values = [game[stat_key] for game in games]
    min_value = min(stat_values)
    max_value = max(stat_values)
    use_fixed_minute_bins = stat_key == "minutes"
    use_integer_bins = stat_values_are_integer_like(stat_values)

    if use_fixed_minute_bins:
        histogram_bins = build_fixed_width_bins(stat_values, MINUTES_BIN_WIDTH)
        histogram_ticks = histogram_bins
        histogram_rwidth = None
    elif use_integer_bins:
        histogram_bins = build_histogram_bins(int(min_value), int(max_value))
        histogram_ticks = build_histogram_ticks(int(min_value), int(max_value))
        histogram_rwidth = 0.9
    else:
        histogram_bins = build_continuous_histogram_bins(stat_values)
        histogram_ticks = None
        histogram_rwidth = None

    fig, (ax_line, ax_hist) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    fig.patch.set_facecolor(CHART_FIGURE_BG)

    ax_line.plot(
        game_numbers,
        stat_values,
        marker="o",
        linewidth=2,
        markersize=5,
        color="#1f77b4",
    )
    ax_line.axhline(
        summary["mean_value"],
        linestyle="--",
        linewidth=1.5,
        color="#d62728",
        label="Mean",
    )
    ax_line.set_title(f"{stat_label} Per Game")
    ax_line.set_xlabel("Game Number")
    ax_line.set_ylabel(stat_label)
    ax_line.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_line.legend(loc="best")

    ax_hist.hist(
        stat_values,
        bins=histogram_bins,
        alpha=0.45,
        color="#4c78a8",
        edgecolor="white",
        rwidth=histogram_rwidth,
    )
    ax_hist.set_title(f"{stat_label} Distribution")
    ax_hist.set_xlabel(stat_label)
    ax_hist.set_ylabel("Number of Games")
    ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
    if use_fixed_minute_bins:
        ax_hist.set_xlim(histogram_bins[0] - HISTOGRAM_LEFT_PADDING, histogram_bins[-1])
        ax_hist.set_xticks(histogram_ticks)
    elif use_integer_bins:
        ax_hist.set_xlim(min_value - 0.5 - HISTOGRAM_LEFT_PADDING, max_value + 0.5)
        ax_hist.set_xticks(histogram_ticks)
    else:
        padding = max(0.5, (max_value - min_value) * 0.05)
        ax_hist.set_xlim(min_value - padding, max_value + padding)
    ax_hist.axvline(
        summary["median_bin"],
        linestyle="--",
        linewidth=2,
        color="#ff7f0e",
        label=f"Median = {summary['median_value']:.1f}",
    )
    ax_hist.legend(loc="best")

    for ax in (ax_line, ax_hist):
        ax.set_facecolor(CHART_AXES_BG)
        ax.tick_params(colors=CHART_TEXT)
        ax.xaxis.label.set_color(CHART_TEXT)
        ax.yaxis.label.set_color(CHART_TEXT)
        ax.title.set_color(CHART_TEXT)
        ax.grid(True, color=CHART_GRID, alpha=0.35)
        for spine in ax.spines.values():
            spine.set_color(CHART_GRID)

    for legend in (ax_line.get_legend(), ax_hist.get_legend()):
        if legend is not None:
            legend.get_frame().set_facecolor(CHART_AXES_BG)
            legend.get_frame().set_edgecolor(CHART_GRID)
            legend.get_frame().set_alpha(0.95)
            for text in legend.get_texts():
                text.set_color(CHART_TEXT)

    fig.tight_layout()
    fig.subplots_adjust(wspace=FIGURE_SUBPLOT_WSPACE)
    return fig


def render_stat_charts(player_name, stat_label, games):
    stat_key = STAT_OPTIONS[stat_label]
    summary = summarize_stat(games, stat_key)
    fig = build_stat_figure(games, stat_key, stat_label, summary)

    st.caption(
        f"Games shown: {summary['games_available']} | "
        f"Mean {stat_label.lower()}: {summary['mean_value']:.1f} | "
        f"Median {stat_label.lower()}: {summary['median_value']:.1f}"
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_placeholder_page(title, caption, message):
    st.title(title)
    st.caption(caption)
    st.info(message)


def render_player_analytics_page():
    st.title("Player Context")

    players = load_players()
    if not players:
        st.warning("No players were found in the source CSV.")
        return

    selected_player = render_player_selector(players)

    if selected_player is None:
        st.info("Select a player to view their scoring charts.")
        return

    selected_games = load_player_games(selected_player["personId"])
    if not selected_games:
        st.warning("No games with positive minutes were found for this player.")
        return

    render_player_profile(selected_player)
    render_rolling_median_table(selected_games)
    st.divider()
    st.subheader("Explore")
    selected_stat_label, recent_games_limit = render_stats_controls(selected_games)
    recent_games = selected_games[-recent_games_limit:]
    render_stat_charts(selected_player["fullName"], selected_stat_label, recent_games)


def render_points_ou_page():
    st.title("Points O/U")
    selected_section = st.radio(
        "Points O/U Section",
        options=("Recommended Bets", "Browse by Game", "Historical Performance", "Methodology"),
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="points_ou_section",
    )

    if selected_section in {"Recommended Bets", "Browse by Game"}:
        with st.spinner("Loading current points O/U lines..."):
            scored_rows = load_live_points_ou_scores()
            ou_rows = scored_rows.copy() if not scored_rows.empty else load_current_points_ou_rows()
            game_browser = build_game_browser_frame(ou_rows)

        if ou_rows.empty:
            st.warning("No current player points lines were found.")
            return

    if selected_section == "Recommended Bets":
        if scored_rows.empty:
            st.info(
                "Points O/U model predictions are not wired into this tab yet. "
                "Once the quantile regression + selection logic is connected, recommendations will appear here."
            )
        else:
            selected_bookmaker = st.selectbox(
                "Bookmaker",
                options=("DraftKings", "FanDuel"),
                index=0,
                key="points_ou_recommended_bookmaker",
            )
            recommended_rows = ou_rows[ou_rows["is_recommended"].fillna(False)].copy()
            recommended_rows = filter_rows_by_bookmaker(recommended_rows, selected_bookmaker)
            render_points_ou_cards(
                recommended_rows,
                empty_message=(
                    f"No current points O/U rows passed the recommendation rule for "
                    f"{selected_bookmaker}."
                ),
                sort_mode="best",
            )

    elif selected_section == "Browse by Game":
        if game_browser.empty:
            st.info("No current points O/U games were available to browse.")
        else:
            game_options = game_browser.to_dict("records")
            selected_game = st.selectbox(
                "Choose a game",
                game_options,
                format_func=lambda row: f"{row['game_label']} • {row['time_et']}",
                index=0,
                key="points_ou_game_select",
            )
            selected_event_id = clean_text(selected_game.get("event_id"))
            selected_game_rows = ou_rows[
                ou_rows["event_id"].fillna("").astype(str) == selected_event_id
            ].copy()
            game_sort_label = st.radio(
                "Sort lines in this game",
                options=("Best", "Player"),
                index=1,
                horizontal=True,
                label_visibility="collapsed",
                key="points_ou_game_sort",
            )
            if game_sort_label == "Player":
                game_sort_mode = "player"
                player_options = sorted(
                    {
                        clean_text(name)
                        for name in selected_game_rows["player_name_odds"].dropna().tolist()
                        if clean_text(name)
                    },
                    key=str.casefold,
                )
                selected_player_name = st.selectbox(
                    "Filter to player",
                    options=["All players", *player_options],
                    index=0,
                    key=f"points_ou_game_player_{selected_event_id}",
                )
                if selected_player_name != "All players":
                    selected_game_rows = selected_game_rows[
                        selected_game_rows["player_name_odds"].fillna("").astype(str)
                        == selected_player_name
                    ].copy()
            else:
                game_sort_mode = "best"
            render_points_ou_cards(
                selected_game_rows,
                empty_message="No lines were available for this game.",
                sort_mode=game_sort_mode,
            )

    elif selected_section == "Historical Performance":
        with st.spinner("Loading historical points O/U performance..."):
            historical_rows = load_historical_points_ou_scores()
        if historical_rows.empty:
            st.info("Historical performance for the Points O/U strategy is not available.")
        else:
            st.caption(
                f"Test Window: {format_test_window_label(historical_rows.get('game_date'))}"
            )
            recommended_hist = historical_rows[
                historical_rows["is_recommended"].fillna(False)
                & historical_rows["actual_side_calc"].isin(["under", "over"])
            ].copy()
            if recommended_hist.empty:
                st.info("No historical points O/U bets passed the recommendation rule.")
            else:
                recommended_hist["is_correct"] = recommended_hist["is_correct"].fillna(False)
                recommended_hist["profit_numeric"] = pd.to_numeric(
                    recommended_hist["profit"], errors="coerce"
                )
                recommended_hist["pick_odds_numeric"] = pd.to_numeric(
                    recommended_hist["pick_odds"], errors="coerce"
                )
                recommended_hist["game_date_sort"] = pd.to_datetime(
                    recommended_hist["game_date"], errors="coerce"
                )
                recommended_hist["bookmaker_last_update_sort"] = pd.to_datetime(
                    recommended_hist["bookmaker_last_update"], utc=True, errors="coerce"
                )

                total_recs = int(len(recommended_hist))
                correct_recs = int(recommended_hist["is_correct"].sum())
                accuracy = correct_recs / total_recs if total_recs else float("nan")

                st.subheader("Historical Recommendation Summary")
                top_1, top_2, top_3 = st.columns(3)
                with top_1:
                    st.metric("Historical Recs", f"{total_recs:,}")
                with top_2:
                    st.metric("Correct Recs", f"{correct_recs:,}")
                with top_3:
                    st.metric(
                        "Rec Accuracy",
                        f"{accuracy:.1%}" if pd.notna(accuracy) else "N/A",
                    )

                accuracy_rows = recommended_hist.copy()
                accuracy_rows = accuracy_rows.sort_values(
                    [
                        "game_date_sort",
                        "bookmaker_last_update_sort",
                        "player_name_odds",
                        "line_points",
                    ],
                    ascending=[True, True, True, True],
                ).reset_index(drop=True)
                accuracy_rows["bet_number"] = range(1, len(accuracy_rows) + 1)
                accuracy_rows["correct_numeric"] = accuracy_rows["is_correct"].astype(int)
                accuracy_rows["cumulative_accuracy"] = (
                    accuracy_rows["correct_numeric"].cumsum() / accuracy_rows["bet_number"]
                )

                if not accuracy_rows.empty:
                    st.subheader("Cumulative Accuracy")
                    accuracy_chart = (
                        alt.Chart(accuracy_rows)
                        .mark_line(color="#10b981", strokeWidth=2)
                        .encode(
                            x=alt.X("bet_number:Q", title="Bet #"),
                            y=alt.Y(
                                "cumulative_accuracy:Q",
                                title="Accuracy",
                                axis=alt.Axis(format=".0%"),
                            ),
                            tooltip=[
                                alt.Tooltip("bet_number:Q", title="Bet #"),
                                alt.Tooltip(
                                    "cumulative_accuracy:Q",
                                    title="Cumulative Accuracy",
                                    format=".1%",
                                ),
                            ],
                        )
                        .properties(height=260, title="Accuracy Over Time")
                    )
                    st.altair_chart(accuracy_chart, use_container_width=True)

                bankroll_rows = recommended_hist.dropna(subset=["profit_numeric"]).copy()
                bankroll_rows = bankroll_rows.sort_values(
                    [
                        "game_date_sort",
                        "bookmaker_last_update_sort",
                        "player_name_odds",
                        "line_points",
                    ],
                    ascending=[True, True, True, True],
                ).reset_index(drop=True)
                bankroll_rows["bet_number"] = range(1, len(bankroll_rows) + 1)
                bankroll_rows["bankroll"] = 100.0 + bankroll_rows["profit_numeric"].cumsum()

                if not bankroll_rows.empty:
                    starting_bankroll = 100.0
                    ending_bankroll = bankroll_rows["bankroll"].iloc[-1]
                    total_profit = ending_bankroll - starting_bankroll
                    st.subheader("Cumulative Bankroll")
                    st.caption(
                        "Historical O/U recommendations, starting with \\$100 and flat \\$1 stakes."
                    )
                    bank_1, bank_2, bank_3 = st.columns(3)
                    with bank_1:
                        st.metric("Start", f"${starting_bankroll:,.2f}")
                    with bank_2:
                        st.metric("Current", f"${ending_bankroll:,.2f}")
                    with bank_3:
                        st.metric("Profit", f"${total_profit:+,.2f}")

                    bankroll_chart = (
                        alt.Chart(bankroll_rows)
                        .mark_line(color="#1f77b4", strokeWidth=2)
                        .encode(
                            x=alt.X("bet_number:Q", title="Bet #"),
                            y=alt.Y("bankroll:Q", title="Bankroll ($)"),
                            tooltip=[
                                alt.Tooltip("bet_number:Q", title="Bet #"),
                                alt.Tooltip("bankroll:Q", title="Bankroll", format=".2f"),
                            ],
                        )
                        .properties(height=260, title="Bankroll Over Time")
                    )
                    st.altair_chart(bankroll_chart, use_container_width=True)

                display = recommended_hist.copy()
                display["game_date"] = display["game_date_sort"].dt.strftime("%Y-%m-%d")
                display["bet_label"] = (
                    display["selection_side"]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .str[:1]
                    + " "
                    + display["line_points"].astype(str)
                ).str.strip()
                display["pick_odds_display"] = display["pick_odds_numeric"].map(
                    lambda value: f"{value:.2f}" if pd.notna(value) else ""
                )
                display["profit_display"] = display["profit_numeric"].map(
                    lambda value: f"{value:+.2f}" if pd.notna(value) else ""
                )
                display["result_display"] = display["is_correct"].map(
                    lambda value: "Win" if bool(value) else "Loss"
                )

                st.subheader("Recent Recommended Bets")
                st.dataframe(
                    display.sort_values(
                        ["game_date_sort", "player_name_odds", "line_points"],
                        ascending=[False, True, True],
                    )[
                        [
                            "game_date",
                            "player_name_odds",
                            "bet_label",
                            "pick_odds_display",
                            "q10",
                            "q50",
                            "q90",
                            "all_bookmaker_titles",
                            "result_display",
                            "profit_display",
                        ]
                    ]
                    .rename(
                        columns={
                            "game_date": "date",
                            "player_name_odds": "player",
                            "bet_label": "bet",
                            "pick_odds_display": "odds",
                            "all_bookmaker_titles": "books",
                            "result_display": "result",
                            "profit_display": "profit",
                        }
                    )
                    .head(40),
                    hide_index=True,
                    use_container_width=True,
                )

    else:
        st.subheader("How This Will Work")
        st.markdown(
            """
This tab now uses a local quantile-regression model built from your downloaded box-score data.

Planned flow:
- load current `player_points` lines
- dedupe repeated bookmaker copies of the same player-line
- generate `q10 / q50 / q90` forecasts with the local transformer
- use the median forecast direction as the side
- filter to large-separation spots only
- surface recommended bets on the first tab
- keep the full browse-by-game experience on the second tab

Current recommendation rule:
- pick `Over` when `q50 > line`
- pick `Under` when `q50 < line`
- only recommend bets where `|q50 - line| >= 7.0`
            """
        )


def render_points_ladder_history_tab():
    period_summary = load_walk_forward_period_summary()
    selected_bets = load_walk_forward_selected_bets()

    if period_summary.empty and selected_bets.empty:
        st.info("No walk-forward history files were found yet.")
        return

    if not period_summary.empty:
        summary = period_summary.copy()
        summary["roi_numeric"] = pd.to_numeric(summary["roi"], errors="coerce")
        summary["win_rate_numeric"] = pd.to_numeric(summary["win_rate"], errors="coerce")
        summary["selected_bets_numeric"] = pd.to_numeric(
            summary["selected_bets"], errors="coerce"
        )
        summary["candidate_rows_numeric"] = pd.to_numeric(
            summary["candidate_rows"], errors="coerce"
        )

        total_bets = int(summary["selected_bets_numeric"].fillna(0).sum())
        header_parts = []
        if not selected_bets.empty:
            header_parts.append(
                f"Test Window: {format_test_window_label(selected_bets.get('game_date'))}"
            )
        header_parts.append(f"Walk-Forward Bets: {total_bets:,}")
        st.caption("  \n".join(header_parts))

        st.subheader("Monthly Summary")
        monthly_display = summary.copy()
        monthly_display["roi"] = monthly_display["roi_numeric"].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else ""
        )
        monthly_display["win_rate"] = monthly_display["win_rate_numeric"].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else ""
        )
        monthly_display["selected_bets"] = monthly_display["selected_bets_numeric"].map(
            lambda value: f"{int(value):,}" if pd.notna(value) else ""
        )
        monthly_display["candidate_rows"] = monthly_display["candidate_rows_numeric"].map(
            lambda value: f"{int(value):,}" if pd.notna(value) else ""
        )
        st.dataframe(
            monthly_display[
                [
                    "eval_period",
                    "candidate_rows",
                    "selected_bets",
                    "win_rate",
                    "roi",
                ]
            ].rename(
                columns={
                    "eval_period": "period",
                    "candidate_rows": "candidates",
                    "selected_bets": "bets",
                    "win_rate": "win %",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    if not selected_bets.empty:
        bets = selected_bets.copy()
        bets["game_date_sort"] = pd.to_datetime(bets["game_date"], errors="coerce")
        bets["bookmaker_last_update_sort"] = pd.to_datetime(
            bets["bookmaker_last_update"], utc=True, errors="coerce"
        )
        bets["trust_win_prob_numeric"] = pd.to_numeric(bets["trust_win_prob"], errors="coerce")
        bets["trust_bet_ev_numeric"] = pd.to_numeric(bets["trust_bet_ev"], errors="coerce")
        bets["profit_numeric"] = pd.to_numeric(bets["profit"], errors="coerce")
        bets["bet_price_numeric"] = pd.to_numeric(bets["bet_price"], errors="coerce")
        bets["bet_side_display"] = bets["bet_side"].fillna("").astype(str).str.upper()
        bets["bet_label"] = (
            bets["bet_side_display"].str[:1] + " " + bets["line_points"].astype(str)
        ).str.strip()
        bets["trust_win_pct"] = bets["trust_win_prob_numeric"].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else ""
        )
        bets["trust_ev"] = bets["trust_bet_ev_numeric"].map(format_ev_multiple)
        bets["profit"] = bets["profit_numeric"].map(
            lambda value: f"{value:+.2f}" if pd.notna(value) else ""
        )
        bets["result"] = bets["bet_result"].fillna("").astype(str).str.title()

        bankroll_rows = bets.dropna(subset=["profit_numeric"]).copy()
        bankroll_rows = bankroll_rows.sort_values(
            ["game_date_sort", "bookmaker_last_update_sort", "player_name_odds", "line_points"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
        bankroll_rows["bet_number"] = range(1, len(bankroll_rows) + 1)
        bankroll_rows["bankroll"] = 100.0 + bankroll_rows["profit_numeric"].cumsum()

        if not bankroll_rows.empty:
            starting_bankroll = 100.0
            ending_bankroll = bankroll_rows["bankroll"].iloc[-1]
            total_profit = ending_bankroll - starting_bankroll
            st.subheader("Cumulative Bankroll")
            st.caption("Walk-forward selected bets, starting with \\$100 and flat \\$1 stakes.")
            bank_1, bank_2, bank_3 = st.columns(3)
            with bank_1:
                st.metric("Start", f"${starting_bankroll:,.2f}")
            with bank_2:
                st.metric("Current", f"${ending_bankroll:,.2f}")
            with bank_3:
                st.metric("Profit", f"${total_profit:+,.2f}")

            bankroll_chart = (
                alt.Chart(bankroll_rows)
                .mark_line(color="#1f77b4", strokeWidth=2)
                .encode(
                    x=alt.X("bet_number:Q", title="Bet #"),
                    y=alt.Y("bankroll:Q", title="Bankroll ($)"),
                    tooltip=[
                        alt.Tooltip("bet_number:Q", title="Bet #"),
                        alt.Tooltip("bankroll:Q", title="Bankroll", format=".2f"),
                    ],
                )
                .properties(height=260, title="Bankroll Over Time")
            )
            st.altair_chart(bankroll_chart, use_container_width=True)

        st.subheader("Recent Selected Bets")
        st.caption("A recent slice of the walk-forward bets that actually passed selection.")
        recent_display = bets.sort_values(
            ["game_date_sort", "player_name_odds", "line_points"],
            ascending=[False, True, True],
        ).head(40)
        st.dataframe(
            recent_display[
                [
                    "game_date",
                    "player_name_odds",
                    "bet_label",
                    "bet_price_numeric",
                    "trust_win_pct",
                    "trust_ev",
                    "result",
                    "profit",
                    "bookmaker_title",
                ]
            ].rename(
                columns={
                    "game_date": "date",
                    "player_name_odds": "player",
                    "bet_label": "bet",
                    "bet_price_numeric": "price",
                    "bookmaker_title": "book",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )


def render_points_ladder_methodology_tab():
    st.subheader("Methodology")
    st.markdown(
        """
This page uses a two-stage modeling pipeline for player points alternate lines.

**Model**

Stage 1 models the scoring distribution through threshold-clearing probabilities. Instead of predicting one number like expected points, it predicts quantities like:

- `P(X >= 1)`
- `P(X >= 2)`
- `P(X >= 3)`
- ...

where `X` is the player's point total in that game.

These are survival-style probabilities: for each possible points threshold, the model estimates the chance that the player finishes at or above that mark. For a player who is very likely to score at least a few points, the early probabilities stay near `1`, and then they gradually fall as the threshold gets more difficult:

- `P(X >= 1) ≈ 1`
- `P(X >= 2) ≈ 1`
- `P(X >= 3) ≈ 1`
- ...
- `P(X >= 20)` lower
- `P(X >= 30)` lower still

The Stage 1 model itself is a multi-output MLP. For each threshold, it produces a logit, applies a sigmoid to turn that logit into a probability, and trains against the historical `0/1` threshold outcomes using binary cross-entropy with logits. After prediction, the threshold curve is made monotone so the estimated chance of clearing a harder threshold cannot exceed the estimated chance of clearing an easier one.

Once those threshold probabilities are available, the model can score the sportsbook's offered alternate lines by reading off the relevant tail probability for that line and converting it into `Over` and `Under` expected values at the posted odds.

Stage 2 then works as a trust model on top of Stage 1. Instead of relearning basketball scoring directly, it learns when a positive-EV Stage 1 bet is actually worth keeping. It uses the Stage 1 probabilities and EVs, plus player, team, opponent, price, and market-context features, and outputs a corrected win probability (`Trust %`) along with trust-adjusted bet quality signals. Those Stage 2 outputs are what drive the recommended bets tab.
        """
    )

    st.subheader("Validation")
    st.markdown(
        """
Validation happens in two layers.

First, Stage 1 is trained and checked with train / validation / test splits on historical player-game data and sportsbook rows. During training, model selection is based on out-of-sample validation performance, including sportsbook-oriented metrics such as log loss on the actually offered lines.

Second, Stage 2 is validated in a walk-forward setup. The trust model is refit month by month using only earlier data, then evaluated on later periods. That gives a much more realistic picture of what the full betting pipeline would have looked like in forward time, rather than just on one frozen split.

So the main idea is:

- Stage 1 validates whether the probability model is learning the scoring distribution well.
- Stage 2 validates whether those raw signals actually translate into useful betting decisions over time.
        """
    )

    st.subheader("Card Numbers")
    st.markdown(
        """
- `Price`: decimal odds for the displayed side.
- `BE %`: break-even probability needed at that price.
- `Trust %`: Stage 2 estimated win probability for that bet.
- `Trust EV`: trust-adjusted expected return multiple. Values above `1.00` are better.
        """
    )


def render_points_ladder_page():
    st.title("Points Alternate")

    selected_section = st.radio(
        "Points Alternate Section",
        options=("Recommended Bets", "Browse by Game", "Historical Performance", "Methodology"),
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="points_alternate_section",
    )

    if selected_section in {"Recommended Bets", "Browse by Game"}:
        with st.spinner("Loading current points alternate lines..."):
            scored_rows = load_live_points_ladder_scores()
            matchup_rows = load_matchup_rows(str(CURRENT_PLAYER_POINTS_ALTERNATE_EVENTS_JSON))

            if scored_rows.empty:
                ladder_rows = pd.DataFrame()
                game_browser = pd.DataFrame()
                primary_bets = pd.DataFrame()
            elif not matchup_rows:
                ladder_rows = pd.DataFrame()
                game_browser = pd.DataFrame()
                primary_bets = pd.DataFrame()
            else:
                ladder_rows = dedupe_ladder_bets(scored_rows)
                ladder_rows = attach_current_book_metadata(ladder_rows)
                ladder_rows = add_ladder_probability_columns(ladder_rows)
                ladder_rows["bet_price_numeric"] = pd.to_numeric(
                    ladder_rows["bet_price"], errors="coerce"
                )
                ladder_rows["trust_blend_score_numeric"] = pd.to_numeric(
                    ladder_rows["trust_blend_score"], errors="coerce"
                )
                ladder_rows["is_primary_bet"] = (
                    ladder_rows["bet_price_numeric"].between(2.0, 5.0, inclusive="left")
                    & ladder_rows["trust_blend_score_numeric"].gt(0.15)
                )
                candidate_rows = ladder_rows[ladder_rows["bet_side"] != ""].copy()
                if candidate_rows.empty:
                    primary_bets = candidate_rows.copy()
                else:
                    primary_bets = candidate_rows[
                        candidate_rows["is_primary_bet"]
                    ].copy()
                primary_bets = primary_bets.sort_values(
                    ["commence_time", "trust_blend_score_numeric", "bet_price_numeric"],
                    ascending=[True, False, False],
                ).reset_index(drop=True)
                game_browser = build_game_browser_frame(ladder_rows)

        if scored_rows.empty:
            st.info("Live ladder odds were found, but no model-scored ladder rows are ready yet.")
            return
        if not matchup_rows:
            st.warning("No current player points ladder events were found.")
            return

    if selected_section == "Recommended Bets":
        sort_label = st.radio(
            "Sort bets",
            options=("Best", "Soonest"),
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="points_ladder_sort",
        )
        sort_mode = "best" if sort_label == "Best" else "chronological"
        render_ladder_cards(
            primary_bets,
            empty_message="No current ladder rows passed the primary betting rule.",
            sort_mode=sort_mode,
        )

    elif selected_section == "Browse by Game":
        if game_browser.empty:
            st.info("No current ladder games were available to browse.")
        else:
            game_options = game_browser.to_dict("records")
            selected_game = st.selectbox(
                "Choose a game",
                game_options,
                format_func=lambda row: f"{row['game_label']} • {row['time_et']}",
                index=0,
            )
            selected_event_id = clean_text(selected_game.get("event_id"))
            selected_game_rows = ladder_rows[
                ladder_rows["event_id"].fillna("").astype(str) == selected_event_id
            ].copy()
            game_sort_label = st.radio(
                "Sort lines in this game",
                options=("Best", "Player"),
                index=0,
                horizontal=True,
                label_visibility="collapsed",
                key="points_ladder_game_sort",
            )
            if game_sort_label == "Player":
                game_sort_mode = "player"
                player_options = sorted(
                    {
                        clean_text(name)
                        for name in selected_game_rows["player_name_odds"].dropna().tolist()
                        if clean_text(name)
                    },
                    key=str.casefold,
                )
                selected_player_name = st.selectbox(
                    "Filter to player",
                    options=["All players", *player_options],
                    index=0,
                    key=f"points_ladder_game_player_{selected_event_id}",
                )
                if selected_player_name != "All players":
                    selected_game_rows = selected_game_rows[
                        selected_game_rows["player_name_odds"].fillna("").astype(str)
                        == selected_player_name
                    ].copy()
            else:
                game_sort_mode = "best"
            render_ladder_cards(
                selected_game_rows,
                empty_message="No lines were available for this game.",
                sort_mode=game_sort_mode,
            )

    elif selected_section == "Historical Performance":
        render_points_ladder_history_tab()

    else:
        render_points_ladder_methodology_tab()


def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio(
        "Go to",
        PAGE_OPTIONS,
        label_visibility="collapsed",
    )
    page_renderers = {
        "Player Analytics": render_player_analytics_page,
        "Points O/U": render_points_ou_page,
        "Points Alternate": render_points_ladder_page,
    }
    page_renderers[selected_page]()


if __name__ == "__main__":
    main()
