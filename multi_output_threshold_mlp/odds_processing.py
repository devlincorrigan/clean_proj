"""Shared odds parsing and bookmaker-priority dedup helpers."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


BOOKMAKER_PRIORITY = [
    "draftkings",
    "fanduel",
    "williamhill_us",  # Caesars
    "betmgm",
    "betrivers",
    "fanatics",
    "pointsbetus",
    "unibet_us",
    "bovada",
    "betonlineag",
    "barstool",
    "mybookieag",
]
BOOKMAKER_PRIORITY_RANK = {
    bookmaker_key: index for index, bookmaker_key in enumerate(BOOKMAKER_PRIORITY)
}
PLAYER_POINTS_ALT_MARKET_KEY = "player_points_alternate"


def normalize_player_name(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", value or "")
    ascii_text = "".join(
        character for character in ascii_text if not unicodedata.combining(character)
    )
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return " ".join(ascii_text.split())


def bookmaker_title(bookmaker: dict) -> str:
    if bookmaker.get("key") == "williamhill_us":
        return "Caesars"
    return bookmaker.get("title", bookmaker.get("key", "unknown"))


def bookmaker_priority_tuple(bookmaker_key: str) -> tuple[int, str]:
    normalized = str(bookmaker_key or "").strip()
    return (BOOKMAKER_PRIORITY_RANK.get(normalized, len(BOOKMAKER_PRIORITY)), normalized)


def should_replace_bookmaker(current_key: str, candidate_key: str) -> bool:
    if not candidate_key:
        return False
    if not current_key:
        return True
    return bookmaker_priority_tuple(candidate_key) < bookmaker_priority_tuple(current_key)


def line_key(value) -> str:
    return f"{float(value):.1f}"


def required_threshold_from_line(line_points: float) -> int:
    return int(math.floor(float(line_points)) + 1)


def load_odds_event(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload


def dedup_player_points_alternate_rows(
    event: dict,
    *,
    market_key: str = PLAYER_POINTS_ALT_MARKET_KEY,
) -> list[dict]:
    by_player_and_line = defaultdict(dict)

    for bookmaker in event.get("bookmakers", []):
        bookmaker_key = str(bookmaker.get("key", "") or "")
        bookmaker_name = bookmaker_title(bookmaker)
        bookmaker_last_update = bookmaker.get("last_update", "")

        for market in bookmaker.get("markets", []):
            if market.get("key") != market_key:
                continue

            market_last_update = market.get("last_update", "")
            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description")
                point = outcome.get("point")
                side = (outcome.get("name") or "").strip().lower()
                if player_name is None or point is None or side not in {"over", "under"}:
                    continue

                normalized_player_name = normalize_player_name(player_name)
                if not normalized_player_name:
                    continue

                line_points = float(point)
                entry = by_player_and_line[(normalized_player_name, line_points)]
                entry["line_points"] = line_points
                entry.setdefault("player_name_odds", str(player_name))
                entry.setdefault("all_bookmaker_keys", set()).add(bookmaker_key)
                entry.setdefault("all_bookmaker_titles", {})[bookmaker_key] = bookmaker_name
                entry.setdefault("over_bookmaker_keys", set())
                entry.setdefault("under_bookmaker_keys", set())
                entry[f"{side}_bookmaker_keys"].add(bookmaker_key)

                if should_replace_bookmaker(entry.get("bookmaker_key", ""), bookmaker_key):
                    entry["player_name_odds"] = str(player_name)
                    entry["bookmaker_key"] = bookmaker_key
                    entry["bookmaker_title"] = bookmaker_name
                    entry["bookmaker_last_update"] = bookmaker_last_update
                    entry["market_last_update"] = market_last_update

                side_key = f"{side}_bookmaker_key"
                if should_replace_bookmaker(entry.get(side_key, ""), bookmaker_key):
                    entry[f"{side}_price"] = outcome.get("price")
                    entry[f"{side}_bookmaker_key"] = bookmaker_key
                    entry[f"{side}_bookmaker_title"] = bookmaker_name
                    entry[f"{side}_bookmaker_last_update"] = bookmaker_last_update
                    entry[f"{side}_market_last_update"] = market_last_update

    rows = []
    for (_, _), line_info in sorted(
        by_player_and_line.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        rows.append(
            {
                "bookmaker_key": line_info.get("bookmaker_key", ""),
                "bookmaker_title": line_info.get("bookmaker_title", ""),
                "bookmaker_last_update": line_info.get("bookmaker_last_update", ""),
                "market_last_update": line_info.get("market_last_update", ""),
                "player_name_odds": line_info["player_name_odds"],
                "line_points": line_info["line_points"],
                "over_price": line_info.get("over_price"),
                "under_price": line_info.get("under_price"),
                "over_bookmaker_key": line_info.get("over_bookmaker_key", ""),
                "over_bookmaker_title": line_info.get("over_bookmaker_title", ""),
                "over_bookmaker_last_update": line_info.get("over_bookmaker_last_update", ""),
                "over_market_last_update": line_info.get("over_market_last_update", ""),
                "under_bookmaker_key": line_info.get("under_bookmaker_key", ""),
                "under_bookmaker_title": line_info.get("under_bookmaker_title", ""),
                "under_bookmaker_last_update": line_info.get("under_bookmaker_last_update", ""),
                "under_market_last_update": line_info.get("under_market_last_update", ""),
                "bookmaker_count": len(line_info.get("all_bookmaker_keys", set())),
                "over_bookmaker_count": len(line_info.get("over_bookmaker_keys", set())),
                "under_bookmaker_count": len(line_info.get("under_bookmaker_keys", set())),
                "all_bookmaker_titles": ", ".join(
                    line_info["all_bookmaker_titles"][bookmaker_key]
                    for bookmaker_key in sorted(
                        line_info.get("all_bookmaker_keys", set()),
                        key=bookmaker_priority_tuple,
                    )
                    if bookmaker_key in line_info.get("all_bookmaker_titles", {})
                ),
            }
        )

    return rows
