"""Inverted BM25 index; no extra dependency or mixing of incompatible scores."""

import math
import re
from collections import Counter, defaultdict
from heapq import nlargest


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


class BM25Index:
    def __init__(self, records: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.records = records
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for index, record in enumerate(records):
            counts = Counter(tokenize(record["chunk"]))
            self.lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                self.postings[token].append((index, frequency))
        self.average_length = sum(self.lengths) / max(1, len(records))

    def search(self, query: str, k: int) -> list[tuple[dict, float]]:
        scores: dict[int, float] = defaultdict(float)
        total = len(self.records)
        for token in set(tokenize(query)):
            postings = self.postings.get(token, [])
            idf = math.log(1 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        ranked = nlargest(k, scores, key=lambda index: (scores[index], -index))
        return [(self.records[index], scores[index]) for index in ranked if scores[index] > 0]
