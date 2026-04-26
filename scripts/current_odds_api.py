import json
import os
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SPORT_KEY = "basketball_nba"
BASE_URL = "https://api.the-odds-api.com"
MARKET = "player_points_alternate"
REGIONS = "us"
CURRENT_ODDS_DIR = PROJECT_ROOT / "data" / "current_odds_api"
MARKET_OUTPUT_DIR = CURRENT_ODDS_DIR / MARKET
EVENTS_OUTPUT_PATH = MARKET_OUTPUT_DIR / "events.json"


def load_api_key():
    api_key = (
        os.environ.get("THE_ODDS_API_KEY")
        or os.environ.get("ODDS_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "Missing Odds API key. Set THE_ODDS_API_KEY or ODDS_API_KEY in your environment."
        )
    return api_key


def fetch_events():
    url = f"{BASE_URL}/v4/sports/{SPORT_KEY}/events"
    params = {
        "apiKey": load_api_key(),
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_event_odds(event_id):
    url = f"{BASE_URL}/v4/sports/{SPORT_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": load_api_key(),
        "regions": REGIONS,
        "markets": MARKET,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_unique_event_ids(events):
    unique_event_ids = []
    seen_event_ids = set()

    for event in events:
        event_id = event.get("id")
        if not event_id or event_id in seen_event_ids:
            continue

        seen_event_ids.add(event_id)
        unique_event_ids.append(event_id)

    return unique_event_ids


def save_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    CURRENT_ODDS_DIR.mkdir(parents=True, exist_ok=True)
    MARKET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    events = fetch_events()
    save_json(EVENTS_OUTPUT_PATH, events)

    event_ids = extract_unique_event_ids(events)

    for event_id in event_ids:
        event_odds = fetch_event_odds(event_id)
        save_json(MARKET_OUTPUT_DIR / f"{event_id}.json", event_odds)

    print(f"Saved {len(events)} events to {EVENTS_OUTPUT_PATH}")
    print(f"Saved {len(event_ids)} event odds files to {MARKET_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
