from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .documents import Chunk


_ATTRIBUTE_QUESTION_RE = re.compile(
    r"^\s*(?:what\s+(?:colou?r|size|shape|date)\s+(?:is|are)|what\s+(?:is|are)\s+the\s+(?:colou?r|size|shape|date)\s+of)\b",
    flags=re.IGNORECASE,
)
_ATTRIBUTE_TERMS = {"color", "colour", "size", "shape", "date"}


# Deliberately conservative: domain terms and acronyms such as "AI" must survive.
STOPWORDS: set[str] = {
    # English
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that",
    "the", "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with",
    # Romanian
    "ale", "al", "care", "ce", "cine", "cu", "cum", "de", "din", "este",
    "la", "pe", "pentru", "și", "sunt", "unde",
    # Turkish
    "bir", "bu", "da", "için", "ile", "kim", "mı", "mi", "mu", "mü",
    "nasıl", "ne", "nerede", "ve",
}



_CANONICAL_SYNONYMS = {
    "capability": "function",
    "feature": "function",
    "objective": "goal",
    "progression": "progress",
}


def _canonical_term(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        term = term[:-3] + "y"
    elif len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
        term = term[:-1]
    return _CANONICAL_SYNONYMS.get(term, term)


def tokens(text: str) -> list[str]:
    """Return Unicode-aware alphabetic and numeric tokens."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W\d_]+|\d+", normalized, flags=re.UNICODE)


def content_tokens(text: str) -> list[str]:
    return [_canonical_term(token) for token in tokens(text) if token not in STOPWORDS]


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    confidence: float = 0.0
    position: int = 0


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"Embedding dimension mismatch: query={len(left)}, document={len(right)}"
        )
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return dot / norm if norm else 0.0


class HybridRetriever:
    """BM25 plus optional dense embeddings, fused with filtered RRF."""

    def __init__(self, chunks: list[Chunk], embedder: EmbeddingProvider | None = None):
        if not chunks:
            raise ValueError("At least one document chunk is required")
        self.chunks = chunks
        self.embedder = embedder
        self.term_counts = [Counter(content_tokens(chunk.text)) for chunk in chunks]
        self.metadata_terms = [
            set(content_tokens(f"{chunk.source} {chunk.heading or ''}")) for chunk in chunks
        ]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / len(self.lengths) or 1.0
        self.document_frequency = Counter(
            term for counts in self.term_counts for term in counts
        )
        self.embeddings = (
            embedder.embed_documents([chunk.text for chunk in chunks]) if embedder else None
        )
        if self.embeddings is not None and len(self.embeddings) != len(self.chunks):
            raise RuntimeError("Embedding provider returned the wrong number of vectors")

    def _bm25(self, query: str) -> list[float]:
        query_terms = Counter(content_tokens(query))
        if not query_terms:
            return [0.0] * len(self.chunks)

        scores: list[float] = []
        for counts, length in zip(self.term_counts, self.lengths, strict=True):
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency[term]
                inverse_document_frequency = math.log(
                    1
                    + (len(self.chunks) - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + 1.2 * (
                    1 - 0.75 + 0.75 * length / self.average_length
                )
                score += (
                    inverse_document_frequency
                    * frequency
                    * 2.2
                    / denominator
                    * query_frequency
                )
            scores.append(score)
        return scores

    @staticmethod
    def _positive_rank(scores: list[float]) -> list[int]:
        """Rank only genuinely positive signals; zeroes must not earn RRF points."""
        return sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda index: scores[index],
            reverse=True,
        )

    def _rerank_bonus(self, query: str, chunk: Chunk, index: int) -> tuple[float, float]:
        query_terms = set(content_tokens(query))
        if _ATTRIBUTE_QUESTION_RE.search(query.strip()):
            query_terms -= _ATTRIBUTE_TERMS
        if not query_terms:
            return 0.0, 0.0

        chunk_terms = set(content_tokens(chunk.text))
        metadata_terms = self.metadata_terms[index]

        # Terms such as a course code or document title identify the source. Do not
        # let those scope terms compete equally with the actual requested fact.
        scope_terms = query_terms & metadata_terms
        answer_terms = query_terms - scope_terms
        terms_to_match = answer_terms or query_terms

        coverage = len(terms_to_match & chunk_terms) / len(terms_to_match)
        query_normalized = " ".join(tokens(query))
        chunk_normalized = " ".join(tokens(chunk.text))
        phrase_bonus = (
            0.15 if query_normalized and query_normalized in chunk_normalized else 0.0
        )
        metadata_coverage = len(scope_terms) / len(query_terms)
        confidence = min(1.0, coverage + metadata_coverage * 0.35 + phrase_bonus)
        rerank = coverage + metadata_coverage * 0.15 + phrase_bonus
        return rerank, confidence

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not query.strip() or top_k < 1:
            return []

        top_k = min(top_k, len(self.chunks))
        lexical = self._bm25(query)
        lexical_rank = self._positive_rank(lexical)
        semantic = [0.0] * len(self.chunks)
        semantic_rank: list[int] = []

        if self.embedder is not None and self.embeddings:
            query_embedding = self.embedder.embed_query(query)
            semantic = [_cosine(query_embedding, vector) for vector in self.embeddings]
            semantic_rank = self._positive_rank(semantic)

        reciprocal_rank_fusion = [0.0] * len(self.chunks)
        for rank, index in enumerate(lexical_rank, start=1):
            reciprocal_rank_fusion[index] += 1 / (60 + rank)
        for rank, index in enumerate(semantic_rank, start=1):
            reciprocal_rank_fusion[index] += 1 / (60 + rank)

        results: list[SearchResult] = []
        for index, chunk in enumerate(self.chunks):
            rerank, lexical_confidence = self._rerank_bonus(query, chunk, index)
            positive_semantic = max(0.0, semantic[index])
            if lexical[index] <= 0 and positive_semantic <= 0 and rerank <= 0:
                continue

            score = (
                reciprocal_rank_fusion[index]
                + rerank * 0.1
                + positive_semantic * 0.05
            )
            semantic_confidence = positive_semantic * 0.75 if semantic_rank else 0.0
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    lexical_score=lexical[index],
                    semantic_score=semantic[index],
                    confidence=max(lexical_confidence, semantic_confidence),
                    position=index,
                )
            )

        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def expand_neighbors(
        self, results: list[SearchResult], window: int = 0
    ) -> list[SearchResult]:
        """Add nearby chunks from the same source while preserving ranked hits first."""
        if window < 1 or not results:
            return results

        positions = {chunk.id: index for index, chunk in enumerate(self.chunks)}
        expanded = list(results)
        seen = {result.chunk.id for result in results}

        for result in results:
            center = positions[result.chunk.id]
            for offset in range(1, window + 1):
                for index in (center - offset, center + offset):
                    if not 0 <= index < len(self.chunks):
                        continue
                    chunk = self.chunks[index]
                    if chunk.source != result.chunk.source or chunk.id in seen:
                        continue
                    seen.add(chunk.id)
                    expanded.append(
                        SearchResult(
                            chunk=chunk,
                            score=result.score / (offset + 1),
                            lexical_score=0.0,
                            semantic_score=0.0,
                            confidence=result.confidence / (offset + 1),
                            position=index,
                        )
                    )
        return expanded

    def close(self) -> None:
        close = getattr(self.embedder, "close", None)
        if callable(close):
            close()


