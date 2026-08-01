from __future__ import annotations

import hashlib
import importlib
import math
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from .database import SQLiteStore


_EXTRACTION_CACHE_VERSION = 6
_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".pptx"}
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
MAX_OFFICE_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_OFFICE_COMPRESSION_RATIO = 100
_OFFICE_ACTIVE_CONTENT = (
    "vbaproject.bin",
    "activex/",
    "embeddings/",
    "oleobject",
)


def validate_document(path: Path) -> None:
    """Reject disguised files and Office packages containing active content."""
    suffix = path.suffix.casefold()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {path}")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Document must be a regular file: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"Document is empty: {path}")
    if size > MAX_DOCUMENT_BYTES:
        raise ValueError(
            f"Document exceeds the {MAX_DOCUMENT_BYTES // (1024 * 1024)} MB upload limit: {path}"
        )

    if suffix in {".txt", ".md"}:
        try:
            path.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Text document is not valid UTF-8: {path}") from exc
        return

    if suffix == ".pdf":
        with path.open("rb") as source:
            header = source.read(1024)
        if b"%PDF-" not in header:
            raise ValueError(f"File has a .pdf extension but is not a PDF: {path}")
        return

    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = {name.replace("\\", "/").casefold() for name in archive.namelist()}
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Office document is not a valid ZIP package: {path}") from exc

    required = "word/document.xml" if suffix == ".docx" else "ppt/presentation.xml"
    if required not in names:
        raise ValueError(f"File contents do not match its {suffix} extension: {path}")
    uncompressed_size = sum(entry.file_size for entry in entries)
    compressed_size = max(1, sum(entry.compress_size for entry in entries))
    if (
        uncompressed_size > MAX_OFFICE_UNCOMPRESSED_BYTES
        or uncompressed_size / compressed_size > MAX_OFFICE_COMPRESSION_RATIO
    ):
        raise ValueError(f"Office document exceeds safe decompression limits: {path}")
    active = sorted(
        name for name in names if any(marker in name for marker in _OFFICE_ACTIVE_CONTENT)
    )
    if active:
        raise ValueError(
            f"Office document contains blocked active content ({active[0]}): {path}"
        )


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str
    page: int | None = None
    heading: str | None = None


def _stable_chunk_id(source: str, position: int, page: int | None, text: str) -> str:
    payload = f"{source}\0{position}\0{page or 0}\0{text}".encode("utf-8")
    return f"chunk-{hashlib.sha256(payload).hexdigest()[:20]}"


def _split_keep_separator(text: str, separator: str) -> list[str]:
    """Split text while retaining separators at the end of each preceding unit."""
    if not separator:
        return [text]
    units: list[str] = []
    start = 0
    for match in re.finditer(re.escape(separator), text):
        units.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        units.append(text[start:])
    return units


def _merge_units(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Merge units into bounded chunks and retain a bounded trailing overlap."""
    merged: list[str] = []
    current: list[str] = []

    def current_length(items: list[str]) -> int:
        return sum(len(item) for item in items)

    for unit in units:
        if not unit:
            continue
        if current and current_length(current) + len(unit) > chunk_size:
            chunk = "".join(current).strip()
            if chunk:
                merged.append(chunk)

            trailing: list[str] = []
            trailing_length = 0
            if overlap > 0:
                for item in reversed(current):
                    if trailing_length + len(item) > overlap:
                        break
                    trailing.insert(0, item)
                    trailing_length += len(item)
            current = trailing

            while current and current_length(current) + len(unit) > chunk_size:
                current.pop(0)

        current.append(unit)

    final = "".join(current).strip()
    if final:
        merged.append(final)
    return merged


def _recursive_split(
    text: str,
    separators: tuple[str, ...],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= chunk_size:
        return [stripped]

    for separator_index, separator in enumerate(separators):
        if separator and separator not in text:
            continue
        if not separator:
            step = chunk_size - overlap
            if step <= 0:
                raise ValueError("chunk_size must be greater than overlap")
            chunks: list[str] = []
            start = 0
            while start < len(text):
                piece = text[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                if start + chunk_size >= len(text):
                    break
                start += step
            return chunks

        remaining = separators[separator_index + 1 :] or ("",)
        split_units: list[str] = []
        for unit in _split_keep_separator(text, separator):
            if len(unit.strip()) <= chunk_size:
                split_units.append(unit)
            else:
                split_units.extend(_recursive_split(unit, remaining, chunk_size, overlap))
        return _merge_units(split_units, chunk_size, overlap)

    return [stripped]


def _dynamic_chunk_limits(text: str) -> tuple[int, int]:
    """Estimate a conservative character window from document structure."""
    paragraph_lengths = sorted(
        len(part.strip()) for part in text.split("\n\n") if part.strip()
    )
    line_lengths = sorted(len(part.strip()) for part in text.splitlines() if part.strip())
    if not paragraph_lengths:
        return max(1, len(text)), 0

    paragraph_index = min(
        len(paragraph_lengths) - 1,
        math.floor((len(paragraph_lengths) - 1) * 0.75),
    )
    line_index = min(
        len(line_lengths) - 1,
        math.floor((len(line_lengths) - 1) * 0.75),
    ) if line_lengths else 0
    paragraph_p75 = paragraph_lengths[paragraph_index]
    line_p75 = line_lengths[line_index] if line_lengths else paragraph_p75

    typical_unit = max(80, min(paragraph_p75, line_p75 * 4))
    document_scale = max(1, round(math.sqrt(max(1, len(text)))))
    chunk_size = min(max(typical_unit * 3, document_scale * 12), max(1, len(text)))
    if "\nAnswer: " in text:
        chunk_size = min(max(1, len(text)), max(chunk_size, max(paragraph_lengths)))
    overlap = min(chunk_size - 1, round(chunk_size * 0.15)) if chunk_size > 1 else 0
    return chunk_size, overlap


def chunk_text(
    text: str,
    source: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    page: int | None = None,
    heading: str | None = None,
    start_position: int = 1,
) -> list[Chunk]:
    """Split text on semantic boundaries while preserving punctuation."""
    dynamic_size, dynamic_overlap = _dynamic_chunk_limits(text)
    resolved_size = dynamic_size if chunk_size is None else chunk_size
    resolved_overlap = dynamic_overlap if overlap is None else overlap

    if resolved_size <= 0:
        raise ValueError("chunk_size must be positive")
    if resolved_overlap < 0 or resolved_size <= resolved_overlap:
        raise ValueError("chunk_size must be greater than overlap, and overlap cannot be negative")

    pieces = _recursive_split(
        text,
        ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""),
        resolved_size,
        resolved_overlap,
    )

    chunks: list[Chunk] = []
    for offset, piece in enumerate(pieces):
        position = start_position + offset
        chunks.append(
            Chunk(
                id=_stable_chunk_id(source, position, page, piece),
                source=source,
                text=piece,
                page=page,
                heading=heading,
            )
        )
    return chunks


def _docx_node_text(node: ElementTree.Element, namespace: str) -> str:
    parts: list[str] = []
    for child in node.iter():
        if child.tag == f"{namespace}t" and child.text:
            parts.append(child.text)
        elif child.tag == f"{namespace}tab":
            parts.append("\t")
        elif child.tag in {f"{namespace}br", f"{namespace}cr"}:
            parts.append("\n")
    return "".join(parts).strip()


def _format_docx_table(table: ElementTree.Element, namespace: str) -> str:
    rows: list[list[str]] = []
    for row in table.findall(f"{namespace}tr"):
        cells = [
            re.sub(r"\s+", " ", _docx_node_text(cell, namespace)).strip()
            for cell in row.findall(f"{namespace}tc")
        ]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    output = ["Table:"]
    if len(rows) > 1 and all(len(row) == 1 for row in rows):
        prompt = rows[0][0]
        for index, row in enumerate(rows[1:], 1):
            output.append(f"Entry {index}:")
            output.append(f"Prompt: {prompt}")
            output.append(f"Answer: {row[0]}")
        return "\n".join(output)

    header = rows[0]
    has_header = len(rows) > 1 and len(header) > 1
    if has_header:
        output.append(f"Columns: {' | '.join(header)}")

    data_rows = rows[1:] if has_header else rows
    for index, row in enumerate(data_rows, 1):
        if has_header:
            pairs = [
                f"{header[column]}: {value}"
                for column, value in enumerate(row)
                if column < len(header)
            ]
            extras = row[len(header) :]
            cells = pairs + extras
        else:
            cells = [f"Cell {column}: {value}" for column, value in enumerate(row, 1)]
        output.append(f"Row {index}: {' | '.join(cells)}")

    return "\n".join(output)


def _docx_block_children(root: ElementTree.Element, namespace: str) -> list[ElementTree.Element]:
    body = root.find(f".//{namespace}body")
    return list(body) if body is not None else list(root)


def _extract_docx(path: Path) -> str:
    """Extract paragraph, table, header, footer, footnote, and endnote text."""
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    try:
        with zipfile.ZipFile(path) as archive:
            part_names = [
                name
                for name in archive.namelist()
                if name == "word/document.xml"
                or re.match(r"word/(?:header|footer)\d+\.xml$", name)
                or name in {"word/footnotes.xml", "word/endnotes.xml"}
            ]
            roots = [ElementTree.fromstring(archive.read(name)) for name in part_names]
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise ValueError(f"Cannot read DOCX file '{path}': {exc}") from exc

    blocks: list[str] = []
    for root in roots:
        root_blocks: list[str] = []
        for child in _docx_block_children(root, namespace):
            if child.tag == f"{namespace}p":
                paragraph_text = _docx_node_text(child, namespace)
                if paragraph_text:
                    root_blocks.append(paragraph_text)
            elif child.tag == f"{namespace}tbl":
                table_text = _format_docx_table(child, namespace)
                if table_text:
                    root_blocks.append(table_text)

        if not root_blocks:
            for paragraph in root.iter(f"{namespace}p"):
                paragraph_text = _docx_node_text(paragraph, namespace)
                if paragraph_text:
                    root_blocks.append(paragraph_text)
        blocks.extend(root_blocks)
    return "\n\n".join(blocks)


def _extract_pptx_slides(path: Path) -> list[tuple[str, int | None]]:
    """Extract visible text from each PPTX slide in presentation order."""
    drawing_namespace = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    try:
        with zipfile.ZipFile(path) as archive:
            slide_names = [
                name
                for name in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ]
            slide_names.sort(
                key=lambda name: int(re.search(r"(\d+)\.xml$", name).group(1))
            )
            roots = [ElementTree.fromstring(archive.read(name)) for name in slide_names]
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, AttributeError) as exc:
        raise ValueError(f"Cannot read PPTX file '{path}': {exc}") from exc

    slides: list[tuple[str, int | None]] = []
    for slide_number, root in enumerate(roots, 1):
        paragraphs: list[str] = []
        for paragraph in root.iter(f"{drawing_namespace}p"):
            fragments = [
                node.text or ""
                for node in paragraph.iter(f"{drawing_namespace}t")
            ]
            text = "".join(fragments).strip()
            if text:
                paragraphs.append(text)
        slides.append(("\n".join(paragraphs), slide_number))
    return slides


def _pdf_reader(path: Path):
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires pypdf. Install it with: python -m pip install pypdf"
        ) from exc
    try:
        return PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot read PDF file '{path}': {exc}") from exc


def _extract_pdf(path: Path) -> str:
    try:
        return "\n".join(page.extract_text() or "" for page in _pdf_reader(path).pages)
    except Exception as exc:
        if isinstance(exc, (RuntimeError, ValueError)):
            raise
        raise ValueError(f"Cannot read PDF file '{path}': {exc}") from exc


def _ocr_pdf_page(path: Path, page_index: int, language: str) -> str:
    try:
        fitz = importlib.import_module("fitz")
        pytesseract = importlib.import_module("pytesseract")
        image_module = importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise RuntimeError(
            "OCR requires PyMuPDF, pytesseract, Pillow, and a Tesseract installation"
        ) from exc

    try:
        with fitz.open(path) as document:
            pixmap = document[page_index].get_pixmap(
                matrix=fitz.Matrix(2, 2), alpha=False
            )
            image = image_module.frombytes(
                "RGB", (pixmap.width, pixmap.height), pixmap.samples
            )
            return pytesseract.image_to_string(image, lang=language).strip()
    except Exception as exc:
        raise RuntimeError(
            f"OCR failed for page {page_index + 1} of '{path}': {exc}"
        ) from exc


def _extract_pdf_pages(
    path: Path,
    ocr: bool = False,
    ocr_language: str = "eng",
) -> list[tuple[str, int | None]]:
    pages: list[tuple[str, int | None]] = []
    try:
        for index, page in enumerate(_pdf_reader(path).pages, 1):
            text = (page.extract_text() or "").strip()
            if ocr and len(re.sub(r"\W", "", text, flags=re.UNICODE)) < 10:
                text = _ocr_pdf_page(path, index - 1, ocr_language)
            pages.append((text, index))
    except Exception as exc:
        if isinstance(exc, (RuntimeError, ValueError)):
            raise
        raise ValueError(f"Cannot read PDF file '{path}': {exc}") from exc
    return pages


def _infer_heading(text: str) -> str | None:
    for line in text.splitlines():
        candidate = line.strip().lstrip("#").strip()
        if candidate and len(candidate) <= 120:
            return candidate
    return None


def extract_text(path: Path) -> str:
    validate_document(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="strict")
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".pptx":
        return "\n\n".join(text for text, _slide in _extract_pptx_slides(path))
    raise ValueError(f"Unsupported document type: {path}")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_collection_key(root: Path, ocr: bool, ocr_language: str) -> str:
    identity = (
        f"version={_EXTRACTION_CACHE_VERSION}\0{root.resolve()}\0"
        f"ocr={ocr}\0language={ocr_language}"
    ).casefold()
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def load_documents(
    directory: str | Path,
    chunk_size: int | None = None,
    overlap: int | None = None,
    cache_dir: str | Path | None = None,
    ocr: bool = False,
    ocr_language: str = "eng",
    progress_callback: Callable[[str, int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[Chunk]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {root}")

    store = SQLiteStore(Path(cache_dir) / "rag.sqlite3") if cache_dir is not None else None
    collection = _document_collection_key(root, ocr, ocr_language)
    active_sources: set[str] = set()
    chunks: list[Chunk] = []

    document_paths = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
    ]
    for file_index, path in enumerate(document_paths, 1):
        if cancel_check and cancel_check():
            raise RuntimeError("Indexing cancelled")
        if progress_callback:
            progress_callback(path.relative_to(root).as_posix(), file_index, len(document_paths))

        source = path.relative_to(root).as_posix()
        validate_document(path)
        active_sources.add(source)
        digest = _file_digest(path)
        sections = (
            store.get_document_sections(collection, source, digest) if store else None
        )
        if sections is None:
            if path.suffix.lower() == ".pdf":
                sections = _extract_pdf_pages(path, ocr, ocr_language)
            elif path.suffix.lower() == ".pptx":
                sections = _extract_pptx_slides(path)
            else:
                sections = [(extract_text(path), None)]
            if store:
                store.put_document_sections(collection, source, digest, sections)

        source_position = 1
        for text, page in sections:
            if not text.strip():
                continue
            page_chunks = chunk_text(
                text=text,
                source=source,
                chunk_size=chunk_size,
                overlap=overlap,
                page=page,
                heading=_infer_heading(text),
                start_position=source_position,
            )
            chunks.extend(page_chunks)
            source_position += len(page_chunks)

    if store:
        store.prune_documents(collection, active_sources)
    return chunks

