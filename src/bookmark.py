import json

BOOKMARK_PATH = "data/bookmark.json"


def load_bookmark():
    with open(BOOKMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def update_bookmark(pov, occurrence, date_str):
    with open(BOOKMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["pov"] = pov
    data["occurrence"] = occurrence
    data["last_read"] = date_str

    with open(BOOKMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
