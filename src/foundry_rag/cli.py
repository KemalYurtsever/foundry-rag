from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .embeddings import optional_foundry_embedder
from .documents import Chunk
from .generators import FoundryLocalGenerator, NO_ANSWER, list_foundry_models
from .pipeline import RAGPipeline
from .retriever import HybridRetriever


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict document-only Foundry Local RAG over local files"
    )
    parser.add_argument("--config", help="JSON file containing CLI default values")
    parser.add_argument("--verbose", action="store_true", help="enable diagnostic logging")
    parser.add_argument("question", nargs="*", help="omit for interactive mode")
    parser.add_argument("--docs", default="data/documents")
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        help="retrieval count; defaults to a context-aware value",
    )
    parser.add_argument(
        "--neighbor-window",
        type=_nonnegative_int,
        default=0,
        help="include this many adjacent chunks around each hit",
    )
    parser.add_argument(
        "--chunk-size",
        type=_positive_int,
        help="override structure-derived chunk size",
    )
    parser.add_argument(
        "--overlap",
        type=_nonnegative_int,
        help="override dynamically calculated chunk overlap",
    )
    parser.add_argument(
        "--model",
        help="Foundry Local chat model alias; omit for conservative extractive mode",
    )
    parser.add_argument(
        "--embedding-model",
        default="qwen3-embedding-0.6b",
        help="Foundry Local embedding model alias",
    )
    parser.add_argument(
        "--cache-dir",
        default=".rag_cache",
        help="document and embedding cache directory",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="OCR PDF pages that contain no extractable text",
    )
    parser.add_argument(
        "--ocr-language",
        default="eng",
        help="Tesseract language code used with --ocr",
    )
    parser.add_argument(
        "--no-embeddings", action="store_true", help="use BM25 retrieval only"
    )
    parser.add_argument(
        "--download-embedding",
        action="store_true",
        help="download the embedding model if needed",
    )
    parser.add_argument(
        "--download", action="store_true", help="download --model if not cached"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="show Foundry Local model aliases and exit",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a small document-grounding smoke test and exit",
    )
    return parser


def _apply_config_defaults(parser: argparse.ArgumentParser) -> None:
    preliminary, _ = parser.parse_known_args()
    if not preliminary.config:
        return

    try:
        defaults = json.loads(Path(preliminary.config).read_text(encoding="utf-8"))
        if not isinstance(defaults, dict):
            raise ValueError("configuration must be a JSON object")
        valid_options = {action.dest for action in parser._actions}
        unknown = set(defaults) - valid_options
        if unknown:
            raise ValueError(f"unknown options: {', '.join(sorted(unknown))}")
        parser.set_defaults(**defaults)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"cannot load configuration: {exc}")


def _validate_resolved_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("top_k", "chunk_size"):
        value = getattr(args, name)
        if value is not None and (not isinstance(value, int) or value < 1):
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    for name in ("neighbor_window", "overlap"):
        value = getattr(args, name)
        if value is not None and (not isinstance(value, int) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be zero or greater")
    if (
        args.chunk_size is not None
        and args.overlap is not None
        and args.overlap >= args.chunk_size
    ):
        parser.error("--overlap must be smaller than --chunk-size")


def _run_self_test() -> bool:
    cases = [
        (
            Chunk("selftest-flag", "self-test.txt", "The sample flag is orange."),
            "What color is the sample flag?",
            "The sample flag is orange.",
            ("self-test.txt",),
        ),
        (
            Chunk("selftest-car", "self-test.txt", "The car has four wheels."),
            "What color is the car?",
            NO_ANSWER,
            (),
        ),
    ]

    passed = True
    for chunk, question, expected_text, expected_sources in cases:
        answer = RAGPipeline(HybridRetriever([chunk])).ask(question, top_k=1)
        ok = answer.text == expected_text and answer.sources == expected_sources
        status = "ok" if ok else "FAILED"
        print(f"{status}: {question}")
        if not ok:
            passed = False
            print(f"  expected: {expected_text!r} sources={expected_sources!r}")
            print(f"  got:      {answer.text!r} sources={answer.sources!r}")
    return passed


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="replace")

    parser = _build_parser()
    _apply_config_defaults(parser)
    args = parser.parse_args()
    _validate_resolved_args(parser, args)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.self_test:
        return 0 if _run_self_test() else 1

    if args.list_models:
        try:
            for alias, cached in list_foundry_models():
                print(f"{alias}\t{'cached' if cached else 'not downloaded'}")
            return 0
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    generator = None
    embedder = None
    pipeline = None
    try:
        generator = (
            FoundryLocalGenerator(args.model, args.download) if args.model else None
        )
        embedder = (
            None
            if args.no_embeddings
            else optional_foundry_embedder(
                args.embedding_model,
                args.download_embedding,
                args.cache_dir,
            )
        )
        pipeline = RAGPipeline.from_directory(
            directory=args.docs,
            generator=generator,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            embedder=embedder,
            cache_dir=args.cache_dir,
            neighbor_window=args.neighbor_window,
            ocr=args.ocr,
            ocr_language=args.ocr_language,
        )
    except Exception as exc:
        close = getattr(generator, "close", None)
        if callable(close):
            close()
        close = getattr(embedder, "close", None)
        if callable(close):
            close()
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    def answer(question: str) -> bool:
        try:
            result = pipeline.ask(question, args.top_k)
        except Exception as exc:
            print(f"\nRequest failed: {exc}", file=sys.stderr)
            return False
        print(f"\n{result.text}")
        if result.sources:
            print(f"\nSources: {', '.join(result.sources)}")
        if args.verbose and result.diagnostics:
            print(f"\nDiagnostics: {'; '.join(result.diagnostics)}")
        return True

    try:
        if args.question:
            return 0 if answer(" ".join(args.question)) else 1

        print("RAG ready. Type a question, or 'quit' to exit.")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if question.lower() in {"quit", "exit"}:
                return 0
            if question:
                answer(question)
    finally:
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
