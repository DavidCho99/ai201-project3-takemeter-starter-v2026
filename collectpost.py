import requests
import csv
import time
from bs4 import BeautifulSoup

API_URL = "https://api.stackexchange.com/2.3/questions"

posts = []

for page in range(1, 4):

    params = {
        "site": "money",
        "page": page,
        "pagesize": 100,
        "order": "desc",
        "sort": "creation",
        "filter": "withbody"
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    for item in data["items"]:

        title = BeautifulSoup(
            item["title"], "html.parser"
        ).get_text(" ", strip=True)

        body = BeautifulSoup(
            item.get("body", ""), "html.parser"
        ).get_text(" ", strip=True)

        text = f"{title}\n\n{body}".strip()

        if len(text) < 50:
            continue

        source = item["link"]
        author = item.get("owner", {}).get(
            "display_name", "Unknown"
        )

        posts.append({
            "text": text,
            "label": "",
            "note": f"Source: {source}; Author: {author}; CC BY-SA"
        })

        if len(posts) >= 200:
            break

    if len(posts) >= 200:
        break

    if not data.get("has_more", False):
        break

    time.sleep(max(data.get("backoff", 0), 1))

with open(
    "labels.csv",
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["text", "label", "note"]
    )

    writer.writeheader()
    writer.writerows(posts)

print(f"Collected {len(posts)} posts.")
