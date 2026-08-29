#!/usr/bin/env python3
"""
Phase 4 - academic preprint crawler.

Reads arXiv through its documented public API. Not a scraper: arXiv publishes an
API precisely so nobody has to scrape them, and the terms ask for one request
per three seconds. That number is the default here and it is not a suggestion.

WHAT THIS HONESTLY DOES

It fetches metadata - title, abstract, authors, categories, versions - and ranks
new papers by how unusual their vocabulary is against a rolling corpus of what
this category normally talks about. That ranking is TRIAGE, not comprehension.
It tells you "this one does not sound like the others", which is a cheap way to
put fifteen papers in front of a human instead of four hundred.

It emphatically does not read papers, evaluate methods, or judge whether a
result is true. Nothing downstream may treat a high novelty score as evidence of
anything. It is a sort order.

The version tracking matters more than the novelty score: v1 -> v2 of a preprint
can silently reverse a headline result, and a pipeline that only ever saw v1 will
keep believing something the authors have already withdrawn.
"""
from __future__ import annotations

import math
import re
import time
import urllib.parse
import xml.etree.ElementTree as ElementTree
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from base import Cache, HttpClient, Provenance, RateLimiter, Record

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# arXiv's terms ask for no more than one request every three seconds.
ARXIV_MIN_INTERVAL = 3.0

STOPWORDS = frozenset("""
a an and are as at be by for from has have in is it its of on or that the to
we our this these those with which can may using used show shows results paper
""".split())


@dataclass(frozen=True)
class Preprint:
    arxiv_id: str            # without version suffix
    version: int
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    published: float
    updated: float
    url: str

    @property
    def is_revision(self) -> bool:
        return self.version > 1

    def terms(self) -> list[str]:
        text = f"{self.title} {self.abstract}".lower()
        return [w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", text) if w not in STOPWORDS]

    def __str__(self) -> str:
        mark = f" (v{self.version})" if self.is_revision else ""
        return f"{self.arxiv_id}{mark}  {self.title[:72]}"


class NoveltyIndex:
    """
    Document-frequency model of what a field normally says.

    A term's surprise is -log(P(term)) under the corpus; a paper's score is the
    mean surprise of its rarest terms. New vocabulary scores high. So does
    nonsense, and so does a paper from an adjacent field that wandered in - which
    is exactly why this is a sort order and not a verdict.
    """

    def __init__(self, smoothing: float = 1.0) -> None:
        self.document_frequency: Counter[str] = Counter()
        self.documents = 0
        self.smoothing = float(smoothing)

    def fit(self, preprints: Iterable[Preprint]) -> "NoveltyIndex":
        for preprint in preprints:
            self.document_frequency.update(set(preprint.terms()))
            self.documents += 1
        return self

    def surprise(self, term: str) -> float:
        total = self.documents + self.smoothing
        seen = self.document_frequency.get(term, 0) + self.smoothing
        return float(-math.log(seen / total))

    def score(self, preprint: Preprint) -> float:
        """
        Mean surprise across ALL of a paper's terms.

        Averaging only the rarest terms saturates: against a small corpus every
        paper has a dozen words nobody else used, so every paper scores
        identically and the ranking carries no information at all. Averaging
        over everything discriminates, because what separates papers is the
        PROPORTION of unusual vocabulary, not whether any exists.
        """
        terms = preprint.terms()
        if not terms or self.documents == 0:
            return 0.0
        return float(sum(self.surprise(t) for t in terms) / len(terms))

    def unusual_terms(self, preprint: Preprint, k: int = 6) -> list[str]:
        terms = set(preprint.terms())
        return sorted(terms, key=self.surprise, reverse=True)[:k]


class ArxivCrawler:
    """
    Polite arXiv client. Default rate limit is arXiv's own published one.

    Constructed with `offline=True` plus a cache directory, it re-derives
    yesterday's answer without touching the network - which is what you want when
    reproducing a result, and what the test suite uses.
    """

    name = "arxiv"

    def __init__(
        self,
        categories: Sequence[str] = ("q-fin.TR", "q-fin.ST"),
        max_results: int = 50,
        client: HttpClient | None = None,
        cache_dir: str | None = None,
        offline: bool = False,
    ) -> None:
        self.categories = list(categories)
        self.max_results = int(max_results)
        self.client = client or HttpClient(
            rate_limiter=RateLimiter(min_interval=ARXIV_MIN_INTERVAL),
            cache=Cache(cache_dir) if cache_dir else None,
            offline=offline,
        )

    def query_url(self, start: int = 0) -> str:
        query = " OR ".join(f"cat:{c}" for c in self.categories)
        return ARXIV_API + "?" + urllib.parse.urlencode({
            "search_query": query,
            "start": start,
            "max_results": self.max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })

    def fetch(self, start: int = 0) -> list[Record]:
        body, provenance = self.client.get(self.query_url(start))
        return [
            Record(self.name, p.arxiv_id, p.updated,
                   {"title": p.title, "abstract": p.abstract, "authors": list(p.authors),
                    "categories": list(p.categories), "version": p.version, "url": p.url},
                   provenance)
            for p in parse_arxiv_atom(body)
        ]

    def fetch_preprints(self, start: int = 0) -> list[Preprint]:
        body, _ = self.client.get(self.query_url(start))
        return parse_arxiv_atom(body)


def parse_arxiv_atom(body: bytes) -> list[Preprint]:
    """Parse an arXiv Atom feed. Malformed entries are skipped, never guessed at."""
    root = ElementTree.fromstring(body)
    out: list[Preprint] = []
    for entry in root.findall("a:entry", ATOM):
        raw_id = _text(entry, "a:id")
        if not raw_id:
            continue
        arxiv_id, version = _split_version(raw_id)
        out.append(Preprint(
            arxiv_id=arxiv_id,
            version=version,
            title=_clean(_text(entry, "a:title")),
            abstract=_clean(_text(entry, "a:summary")),
            authors=tuple(_clean(_text(a, "a:name")) for a in entry.findall("a:author", ATOM)),
            categories=tuple(c.attrib.get("term", "") for c in entry.findall("a:category", ATOM)),
            published=_timestamp(_text(entry, "a:published")),
            updated=_timestamp(_text(entry, "a:updated")),
            url=raw_id,
        ))
    return out


def triage(preprints: Sequence[Preprint], corpus: Sequence[Preprint] | None = None,
           k: int = 10) -> list[tuple[Preprint, float, list[str]]]:
    """
    Rank papers by unusual vocabulary against a corpus. A sort order, nothing more.

    GIVE IT A BACKGROUND CORPUS. Scored against itself a batch tells you almost
    nothing - "unusual" only means anything relative to what a field normally
    says, so the corpus should be the last few hundred papers in the category,
    not today's twenty.
    """
    index = NoveltyIndex().fit(corpus if corpus is not None else preprints)
    scored = [(p, index.score(p), index.unusual_terms(p)) for p in preprints]
    scored.sort(key=lambda row: row[1], reverse=True)
    return scored[:k]


def revisions(previous: Sequence[Preprint], current: Sequence[Preprint]) -> list[Preprint]:
    """
    Papers whose version number moved since last run.

    Worth more attention than new submissions: a v1 you already acted on can
    become a v2 that says the opposite, and nothing announces that.
    """
    seen = {p.arxiv_id: p.version for p in previous}
    return [p for p in current if p.arxiv_id in seen and p.version > seen[p.arxiv_id]]


def _text(node, path: str) -> str:
    found = node.find(path, ATOM)
    return (found.text or "") if found is not None else ""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_version(raw_id: str) -> tuple[str, int]:
    tail = raw_id.rstrip("/").split("/abs/")[-1]
    match = re.match(r"^(.*?)v(\d+)$", tail)
    return (match.group(1), int(match.group(2))) if match else (tail, 1)


def _timestamp(value: str) -> float:
    try:
        return time.mktime(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, TypeError):
        return 0.0


if __name__ == "__main__":
    import sys

    print("STARK MK18 - preprint crawler (live arXiv API)\n")
    crawler = ArxivCrawler(categories=["q-fin.TR", "q-fin.ST", "q-fin.PM"], max_results=200)
    try:
        papers = crawler.fetch_preprints()
    except Exception as e:
        print(f"could not reach arXiv ({type(e).__name__}: {e}) - nothing to show.")
        sys.exit(0)
    if len(papers) < 40:
        print(f"only {len(papers)} papers returned; need more for a background corpus.")
        sys.exit(0)

    papers.sort(key=lambda p: p.updated, reverse=True)
    todays, background = papers[:25], papers[25:]
    print(f"fetched {len(papers)} preprints from {', '.join(crawler.categories)}")
    print(f"  the {len(background)} older ones become the background corpus - what this")
    print(f"  field normally says - and the newest {len(todays)} are scored against it.")

    revised = [p for p in todays if p.is_revision]
    print(f"  {len(revised)} of the newest are REVISIONS. A v2 can reverse a v1 you already")
    print("  acted on, and nothing announces that.\n")

    ranked = triage(todays, corpus=background, k=len(todays))
    print("triage by unusual vocabulary (a SORT ORDER, not a judgement):")
    for preprint, score, unusual in ranked[:5]:
        print(f"  [{score:5.2f}] {preprint}")
        print(f"           unusual: {', '.join(unusual[:5])}")

    scores = [s for _, s, _ in ranked]
    print(f"\n  score spread {min(scores):.2f} to {max(scores):.2f} across "
          f"{len(scores)} papers - the ranking discriminates.")
    print(f"\nWhat this did: put 5 papers in front of a human instead of {len(todays)}.")
    print("What it did NOT do: read them, check them, or judge whether any of it is true.")
