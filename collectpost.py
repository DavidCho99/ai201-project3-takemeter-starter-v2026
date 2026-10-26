import argparse
import csv
import re
import shutil
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

API_URL = "https://api.stackexchange.com/2.3/search/advanced"

DATASET = Path("labels.csv")
CANDIDATES = Path("troubleshooting_candidates.csv")
BACKUP = Path("labels_before_improvement.csv")

FIELDS = ["text", "label", "note"]

# Search terms are for candidate discovery only.
# They do NOT determine the correct label.
SEARCH_TERMS = [
    "can't access account",
    "payment failed",
    "transaction declined",
    "transfer missing",
    "bank account locked",
    "withdrawal problem",
    "credit card payment",
    "refund not received",
    "401k account access",
    "deposit missing",
]


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not set(FIELDS).issubset(reader.fieldnames or []):
        raise ValueError(
            f"{path} must contain: {FIELDS}"
        )

    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def clean_html(html):
    return BeautifulSoup(
        html, "html.parser"
    ).get_text(" ", strip=True)


def collect_candidates():
    original = read_csv(DATASET)

    existing = {
        normalize(row["text"])
        for row in original
    }

    candidates = []
    seen = set(existing)

    session = requests.Session()

    for term in SEARCH_TERMS:
        print(f"\nSearching: {term}")

        params = {
            "site": "money",
            "q": term,
            "page": 1,
            "pagesize": 100,
            "sort": "relevance",
            "order": "desc",
            "filter": "withbody",
        }

        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=30,
            )

            response.raise_for_status()
            data = response.json()

        except requests.RequestException as e:
            print(f"Request failed: {e}")
            continue

        for item in data.get("items", []):
            title = clean_html(item.get("title", ""))
            body = clean_html(item.get("body", ""))

            text = f"{title}\n\n{body}".strip()

            if len(text) < 50:
                continue

            key = normalize(text)

            if key in seen:
                continue

            seen.add(key)

            source = item.get("link", "")
            author = item.get("owner", {}).get(
                "display_name", "Unknown"
            )

            candidates.append({
                "text": text,

                # Intentionally empty!
                # You must review each post yourself.
                "label": "",

                "note": (
                    f"candidate; Source: {source}; "
                    f"Author: {author}; CC BY-SA"
                ),
            })

        print(f"Candidates collected: {len(candidates)}")

        if data.get("quota_remaining") is not None:
            print("API quota remaining:", data["quota_remaining"])

        # Respect API backoff.
        time.sleep(max(data.get("backoff", 0), 1))

    write_csv(CANDIDATES, candidates)

    print(f"\nSaved {len(candidates)} candidates")
    print(f"File: {CANDIDATES}")
    print("Review the candidates and assign labels manually.")


def apply_improvement():
    original = read_csv(DATASET)
    candidates = read_csv(CANDIDATES)

    # Select only questions explicitly approved
    # as troubleshooting.
    approved = [
        row for row in candidates
        if row["label"].strip() == "troubleshooting"
    ]

    if len(approved) != 30:
        raise ValueError(
            f"Expected exactly 30 approved troubleshooting "
            f"posts, found {len(approved)}."
        )

    existing = {
        normalize(row["text"])
        for row in original
    }

    approved_keys = [
        normalize(row["text"])
        for row in approved
    ]

    if len(set(approved_keys)) != 30:
        raise ValueError(
            "Duplicate questions found among approved posts."
        )

    overlap = existing.intersection(approved_keys)

    if overlap:
        raise ValueError(
            f"{len(overlap)} approved questions "
            "already exist in labels.csv."
        )

    if BACKUP.exists():
        raise FileExistsError(
            f"{BACKUP} already exists. "
            "Refusing to overwrite the original backup."
        )

    # Backup BEFORE changing anything.
    shutil.copy2(DATASET, BACKUP)

    improved = original + approved

    write_csv(DATASET, improved)

    print("\nImprovement successfully applied!")
    print(f"Original examples: {len(original)}")
    print(f"New examples: {len(approved)}")
    print(f"Total examples: {len(improved)}")
    print(f"Backup: {BACKUP}")

    print("\nNew class distribution:")

    for label in [
        "concept",
        "advice",
        "regulation",
        "troubleshooting",
    ]:
        count = sum(
            row["label"] == label
            for row in improved
        )

        print(f"{label}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["collect", "apply"],
        required=True,
    )

    args = parser.parse_args()

    if args.mode == "collect":
        collect_candidates()

    elif args.mode == "apply":
        apply_improvement()
