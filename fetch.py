#!/usr/bin/env python3
"""Pull recent AI papers from the public arXiv API into data/papers.json.

arXiv asks for one request every three seconds, so that is the pace. Pages that come back empty
(the API does this under load) are retried a few times before the run gives up on that page.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = "http://export.arxiv.org/api/query"
CATEGORIES = ("cs.AI", "cs.LG", "cs.CL", "cs.CV")
DELAY = 3.0  # seconds between requests, per arXiv's API terms
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


def parse_feed(xml_text: str) -> list[dict]:
    """One Atom page -> list of papers. Whitespace in titles and abstracts is collapsed."""
    papers = []
    for e in ET.fromstring(xml_text).iter(ATOM + "entry"):
        url = (e.findtext(ATOM + "id") or "").strip()
        if "/abs/" not in url:
            continue
        versioned = url.rsplit("/abs/", 1)[1]
        primary = e.find(ARXIV + "primary_category")
        papers.append({
            "id": re.sub(r"v\d+$", "", versioned),
            "url": "https://arxiv.org/abs/" + versioned,
            "title": " ".join((e.findtext(ATOM + "title") or "").split()),
            "abstract": " ".join((e.findtext(ATOM + "summary") or "").split()),
            "authors": [a.findtext(ATOM + "name") for a in e.iter(ATOM + "author")],
            "date": (e.findtext(ATOM + "published") or "")[:10],
            "category": primary.get("term") if primary is not None else "",
        })
    return papers


def fetch_page(start: int, size: int) -> list[dict]:
    query = urllib.parse.urlencode({
        "search_query": " OR ".join("cat:" + c for c in CATEGORIES),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": size,
    })
    req = urllib.request.Request(API + "?" + query, headers={"User-Agent": "jev-papers/1.0 (open source paper classifier)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return parse_feed(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--page", type=int, default=200)
    ap.add_argument("--refresh", action="store_true", help="ignore the cached data/papers.json")
    a = ap.parse_args()

    out = ROOT / "data" / "papers.json"
    if out.is_file() and not a.refresh:
        print(f"cached: {len(json.loads(out.read_text()))} papers in {out} (use --refresh to pull again)")
        return 0

    seen: dict[str, dict] = {}
    start = 0
    while len(seen) < a.count:
        page = []
        for attempt in range(4):
            try:
                page = fetch_page(start, a.page)
            except OSError as err:
                print(f"  start={start}: {err}", file=sys.stderr)
            if page:
                break
            time.sleep(DELAY * (attempt + 2))
        if not page:
            print(f"arXiv returned nothing at start={start} after 4 tries, stopping with {len(seen)}", file=sys.stderr)
            break
        for p in page:
            seen.setdefault(p["id"], p)
        print(f"start={start}: {len(page)} entries, {len(seen)} unique")
        start += a.page
        time.sleep(DELAY)

    papers = list(seen.values())[: a.count]
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(papers, indent=1, ensure_ascii=False))
    print(f"wrote {len(papers)} papers to {out} ({papers[-1]['date']} to {papers[0]['date']})")
    return 0 if papers else 1


if __name__ == "__main__":
    sys.exit(main())
