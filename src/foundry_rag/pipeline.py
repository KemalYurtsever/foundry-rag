from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .documents import load_documents
from .generators import (
    NO_ANSWER,
    Evidence,
    EvidenceSelection,
    ExtractiveGenerator,
    Generator,
    is_attribute_question,
    question_evidence_terms,
)
from .retriever import EmbeddingProvider, HybridRetriever, SearchResult, content_tokens


LOGGER = logging.getLogger(__name__)


_NUMBERED_DISPLAY_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+\S")


def _is_display_heading(line: str) -> bool:
    if len(line) > 90 or line.endswith((".", "!", "?", ":", ";", ",")):
        return False
    if _NUMBERED_DISPLAY_HEADING_RE.match(line):
        return True
    if not line[:1].isupper():
        return False
    words = line.split()
    if not 1 <= len(words) <= 8:
        return False
    return sum(word[:1].isupper() for word in words) >= max(1, len(words) - 1)


def _format_verified_quote(quote: str) -> str:
    """Format verified evidence for display without adding facts."""
    lines = [line.strip() for line in quote.strip().splitlines() if line.strip()]
    if not lines:
        return ""

    if any(line.startswith("Prompt: ") for line in lines):
        answer_lines: list[str] = []
        in_answer = False
        for line in lines:
            if line.startswith("Answer: "):
                in_answer = True
                answer_lines.append(line.removeprefix("Answer: ").strip())
                continue
            if in_answer:
                answer_lines.append(line)
        answer = " ".join(part for part in answer_lines if part).strip()
        if answer:
            return re.sub(r"\s+", " ", answer).strip()
        return ""

    bullet_re = re.compile(r"^(?:[-*\u2022\u2013\u2014\uf0b7]|o\s+)\s*(.+)$", re.IGNORECASE)
    has_colon_list = any(
        line.endswith(":") and index + 1 < len(lines)
        for index, line in enumerate(lines)
    )
    has_structure = len(lines) >= 3 and (
        has_colon_list
        or any(bullet_re.match(line) or _is_display_heading(line) for line in lines)
    )
    if not has_structure:
        return re.sub(r"\s+", " ", quote).strip()

    # PDF line wraps sometimes leave a preceding sentence before a wrapped
    # "This section..." lead-in. Keep the relevant lead-in for readability.
    if len(lines) > 1 and lines[1][:1].islower():
        match = re.search(r"\.\s+(This)\s*$", lines[0])
        if match:
            lines[0] = match.group(1)
    if len(lines) > 1 and lines[0] == "This" and lines[1][:1].islower():
        lines[1] = f"This {lines[1]}"
        lines = lines[1:]

    output: list[str] = []
    paragraph: list[str] = []
    in_colon_list = False

    def flush_paragraph() -> None:
        if paragraph:
            output.append(re.sub(r"\s+", " ", " ".join(paragraph)).strip())
            paragraph.clear()

    def looks_like_plain_list_item(line: str) -> bool:
        return bool(line) and len(line) <= 100 and not line.endswith((".", "!", "?", ":"))

    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        line = line.replace("T est", "Test")
        line = line.replace("functi on", "function")
        line = line.replace("analy zing", "analyzing")
        bullet = bullet_re.match(line)
        if bullet:
            flush_paragraph()
            output.append(f"- {bullet.group(1).strip()}")
            in_colon_list = True
            continue
        if _is_display_heading(line):
            flush_paragraph()
            in_colon_list = True
            if output and output[-1]:
                output.append("")
            output.append(line)
            continue
        if line.endswith(":"):
            flush_paragraph()
            output.append(line)
            in_colon_list = True
            continue
        if in_colon_list and looks_like_plain_list_item(line):
            flush_paragraph()
            output.append(f"- {line}")
            continue
        paragraph.append(line)
        in_colon_list = False
    flush_paragraph()

    return "\n".join(output).strip()


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


def _canonical_terms(text: str) -> set[str]:
    return {_canonical_term(term) for term in content_tokens(text)}


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[str, ...]
    results: tuple[SearchResult, ...]
    diagnostics: tuple[str, ...] = ()


class RAGPipeline:
    """Fail-closed RAG pipeline that emits only verified document quotations."""

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Generator | None = None,
        neighbor_window: int = 0,
    ):
        if neighbor_window < 0:
            raise ValueError("neighbor_window cannot be negative")
        self.retriever = retriever
        self.generator = generator or ExtractiveGenerator()
        self.neighbor_window = neighbor_window
        self._closed = False

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        generator: Generator | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        embedder: EmbeddingProvider | None = None,
        cache_dir: str | Path | None = ".rag_cache",
        neighbor_window: int = 0,
        ocr: bool = False,
        ocr_language: str = "eng",
    ) -> "RAGPipeline":
        chunks = load_documents(
            directory,
            chunk_size,
            overlap,
            cache_dir,
            ocr,
            ocr_language,
        )
        return cls(HybridRetriever(chunks, embedder), generator, neighbor_window)

    def _dynamic_top_k(self) -> int:
        corpus_size = len(self.retriever.chunks)
        context_length = getattr(self.generator, "context_length", None)
        max_output = getattr(self.generator, "max_output_tokens", None) or 0

        corpus_target = max(3, math.ceil(math.log2(corpus_size + 1)))
        if context_length:
            reserved_output = min(max_output or 512, max(128, context_length // 4))
            available_characters = max(256, (context_length - reserved_output - 256) * 3)
            average_chunk = max(
                1,
                round(
                    sum(len(chunk.text) for chunk in self.retriever.chunks)
                    / corpus_size
                ),
            )
            context_target = max(1, available_characters // average_chunk)
            return max(1, min(corpus_size, corpus_target, context_target))
        return max(1, min(corpus_size, corpus_target))

    def _fit_results_to_context(
        self, question: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        context_length = getattr(self.generator, "context_length", None)
        if not context_length:
            return results

        max_output = getattr(self.generator, "max_output_tokens", None) or 512
        reserved_output = min(max_output, max(128, context_length // 4))
        prompt_overhead_tokens = 384 + len(question.split()) * 2
        available_tokens = max(
            128,
            context_length - reserved_output - prompt_overhead_tokens,
        )
        available_characters = available_tokens * 3

        selected: list[SearchResult] = []
        used = 0
        for result in results:
            estimated = len(result.chunk.text) + len(result.chunk.source) + 100
            if selected and used + estimated > available_characters:
                continue
            if not selected and estimated > available_characters:
                # Retain one result. The generator receives a bounded slice rather than no context.
                truncated_chunk = type(result.chunk)(
                    id=result.chunk.id,
                    source=result.chunk.source,
                    text=result.chunk.text[: max(1, available_characters - 100)],
                    page=result.chunk.page,
                    heading=result.chunk.heading,
                )
                selected.append(
                    SearchResult(
                        chunk=truncated_chunk,
                        score=result.score,
                        lexical_score=result.lexical_score,
                        semantic_score=result.semantic_score,
                        confidence=result.confidence,
                        position=result.position,
                    )
                )
                break
            selected.append(result)
            used += estimated
        return selected

    @staticmethod
    def _minimum_confidence(question: str) -> float:
        term_count = max(1, len(set(content_tokens(question))))
        return min(0.45, 0.25 + 0.15 / math.sqrt(term_count))

    @staticmethod
    def _validate_selection(
        question: str,
        selection: EvidenceSelection,
        results: list[SearchResult],
    ) -> tuple[Evidence, ...]:
        if not selection.answerable or not selection.evidence:
            return ()

        chunks_by_id = {result.chunk.id: result.chunk for result in results}
        valid: list[Evidence] = []
        seen: set[tuple[str, str]] = set()

        for evidence in selection.evidence:
            chunk = chunks_by_id.get(evidence.chunk_id)
            quote = evidence.quote.strip()
            key = (evidence.chunk_id, quote)
            if chunk is None or not quote or key in seen:
                continue
            # Exact substring validation is the hard grounding boundary.
            if quote not in chunk.text:
                LOGGER.warning(
                    "Rejected non-verbatim evidence for chunk %s", evidence.chunk_id
                )
                continue
            seen.add(key)
            valid.append(Evidence(evidence.chunk_id, quote))

        if not valid:
            return ()

        query_terms = question_evidence_terms(question)
        if not query_terms:
            return ()
        if is_attribute_question(question):
            color_supported = False
            for evidence in valid:
                quote = f" {' '.join(evidence.quote.split()).casefold()} "
                if any(phrase in quote for phrase in (" is ", " are ", " was ", " were ", " color is ", " colour is ")):
                    color_supported = True
                    break
            if not color_supported:
                return ()

        support_text = " ".join(
            f"{evidence.quote} {chunks_by_id[evidence.chunk_id].source} "
            f"{chunks_by_id[evidence.chunk_id].heading or ''}"
            for evidence in valid
        )
        query_terms = {_canonical_term(term) for term in query_terms}
        evidence_terms = _canonical_terms(support_text)
        coverage = len(query_terms & evidence_terms) / len(query_terms)
        if coverage < 0.75:
            LOGGER.info(
                "Rejected evidence because query-term coverage %.3f is below 0.75",
                coverage,
            )
            return ()
        return tuple(valid)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        if self._closed:
            raise RuntimeError("RAG pipeline is closed")
        if not question.strip():
            return Answer(NO_ANSWER, (), (), ("empty question",))
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")

        retrieval_count = self._dynamic_top_k() if top_k is None else top_k
        results = self.retriever.search(question, retrieval_count)
        LOGGER.info("Retrieved %d candidate chunks", len(results))

        minimum_confidence = self._minimum_confidence(question)
        if not results:
            LOGGER.info("Abstaining because retrieval returned no candidates")
            return Answer(NO_ANSWER, (), tuple(results), ("no retrieved chunks",))
        if results[0].confidence < minimum_confidence:
            LOGGER.info(
                "Abstaining because retrieval confidence is below %.3f",
                minimum_confidence,
            )
            diagnostics = (
                f"low retrieval confidence: {results[0].confidence:.3f} < {minimum_confidence:.3f}",
            )
            return Answer(NO_ANSWER, (), tuple(results), diagnostics)

        expanded_results = self.retriever.expand_neighbors(
            results, self.neighbor_window
        )
        generation_results = self._fit_results_to_context(question, expanded_results)
        selection = self.generator.generate(question, generation_results)
        evidence = self._validate_selection(question, selection, generation_results)
        if not evidence:
            return Answer(NO_ANSWER, (), tuple(expanded_results), ("no validated evidence",))

        chunks_by_id = {
            result.chunk.id: result.chunk for result in generation_results
        }
        # Chunk IDs are internal validation metadata. The user receives a clean,
        # document-grounded answer; source filenames/pages are shown separately.
        formatted_quotes = [
            _format_verified_quote(item.quote) for item in evidence
        ]
        text = "\n".join(quote for quote in formatted_quotes if quote)
        if not text:
            return Answer(NO_ANSWER, (), tuple(expanded_results), ("validated evidence formatted to empty text",))
        sources = tuple(
            dict.fromkeys(
                (
                    f"{chunks_by_id[item.chunk_id].source} "
                    f"(page {chunks_by_id[item.chunk_id].page})"
                    if chunks_by_id[item.chunk_id].page
                    else chunks_by_id[item.chunk_id].source
                )
                for item in evidence
            )
        )
        return Answer(text, sources, tuple(expanded_results))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        generator_close = getattr(self.generator, "close", None)
        if callable(generator_close):
            generator_close()
        retriever_close = getattr(self.retriever, "close", None)
        if callable(retriever_close):
            retriever_close()

    def __enter__(self) -> "RAGPipeline":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
