# -*- coding: utf-8 -*-
"""Paper-QA Document and PDF Reader Integration for DaltonX.

Ports the core document parsing, page-by-page extraction, and chunking mechanics
from paper-qa (types, pypdf parser, readers) into pure Python, zero-bloat components
that convert Text chunks into DaltonX Passage objects with exact page spans,
docnames, dockey MD5 hashes, and citations attached.
"""

from __future__ import absolute_import, print_function, unicode_literals

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pypdf
import pypdf.errors

from daltonx.reader.extractor import Passage

# Regex for invalid unicode handling (orphaned surrogates, null bytes)
INVALID_UNICODE_CHARS = re.compile(r"[\x00\uD800-\uDFFF]")
REPLACEMENT_CHAR = "\ufffd"


def clean_invalid_unicode(text: str, repl: str = REPLACEMENT_CHAR) -> str:
    """Clean invalid Unicode characters by replacing them with the replacement character."""
    if not text:
        return ""
    return INVALID_UNICODE_CHARS.sub(repl, text)


def md5sum(file_path: Union[str, os.PathLike]) -> str:
    """Compute hex MD5 hash of a file's contents."""
    data = Path(file_path).read_bytes()
    return hashlib.md5(data).hexdigest()


def resolve_page_range(
    page_range: Union[int, Tuple[int, int], Sequence[int], None],
    page_count: int,
) -> range:
    """Convert a 1-indexed page_range into a 0-indexed range object."""
    if page_range is None:
        return range(page_count)
    if isinstance(page_range, int):
        return range(page_range - 1, min(page_range, page_count))
    return range(page_range[0] - 1, min(page_range[1], page_count))


@dataclass
class ParsedMetadata:
    """Metadata describing document parsing provenance and metrics."""

    parsing_libraries: List[str] = field(default_factory=list)
    total_parsed_text_length: int = 0
    count_parsed_media: int = 0
    name: str = ""
    chunk_metadata: Optional[Any] = None


@dataclass
class ParsedText:
    """Represents raw parsed text content with associated parser metadata.

    content is typically a dict mapping 1-based page numbers ('1', '2') to text,
    or a single string for unpaginated text.
    """

    content: Union[str, Dict[str, Any]]
    metadata: ParsedMetadata = field(default_factory=ParsedMetadata)


class Doc:
    """Represents a source document with identifier, content hash, and citation.

    Parameters
    ----------
    docname : str
        Human-readable document name (e.g. 'Hubinger2024').
    dockey : str
        Unique MD5 hash of document contents.
    citation : str
        Formal bibliographic citation.
    content_hash : Optional[str]
        Optional content hash if distinct from dockey.
    """

    def __init__(
        self,
        docname: str,
        dockey: str,
        citation: str,
        content_hash: Optional[str] = None,
    ):
        self.docname = docname
        self.dockey = dockey
        self.citation = citation
        self.content_hash = content_hash or dockey

    def __hash__(self) -> int:
        return hash((self.docname, self.dockey))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Doc):
            return False
        return self.docname == other.docname and self.dockey == other.dockey

    def __repr__(self) -> str:
        return f"Doc(docname={self.docname!r}, dockey={self.dockey[:8]}..., citation={self.citation!r})"


class Text:
    """A sliced text chunk linked to its parent document and page span.

    Parameters
    ----------
    text : str
        Processed text content of the chunk.
    name : str
        Identifier with page span (e.g. 'Hubinger2024 pages 2-3' or 'Hubinger2024 pages 1').
    doc : Doc
        Parent document instance.
    media : Optional[List[Any]]
        Associated media items (if any).
    """

    def __init__(
        self,
        text: str,
        name: str,
        doc: Doc,
        media: Optional[List[Any]] = None,
    ):
        self.text = text
        self.name = name
        self.doc = doc
        self.media = media or []

    def to_passage(self, chunk_index: int = 0) -> Passage:
        """Convert this Text chunk into a DaltonX Passage object with full metadata.

        Preserves docname, dockey, citation, chunk name, and page-span.
        """
        # Extract page span from name if present (e.g. 'Hubinger2024 pages 1-2' -> 'pages 1-2')
        page_span = ""
        if " pages " in self.name:
            page_span = self.name.split(" pages ", 1)[1]
            if not page_span.startswith("pages"):
                page_span = f"pages {page_span}"
        elif " chunk " in self.name:
            page_span = self.name.split(" ", 1)[1]

        metadata: Dict[str, Any] = {
            'docname': self.doc.docname,
            'dockey': self.doc.dockey,
            'citation': self.doc.citation,
            'name': self.name,
            'chunk_name': self.name,
            'paper_id': self.name,
            'page_span': page_span or self.name,
        }
        return Passage(
            text=self.text,
            metadata=metadata,
            chunk_index=chunk_index,
        )

    def __repr__(self) -> str:
        return f"Text(name={self.name!r}, len={len(self.text)}, doc={self.doc.docname!r})"


def parse_pdf_to_pages(
    path: Union[str, os.PathLike],
    page_size_limit: Optional[int] = None,
    page_range: Union[int, Tuple[int, int], None] = None,
    **_: Any,
) -> ParsedText:
    """Parse a PDF file page-by-page using pypdf into a ParsedText object.

    Parameters
    ----------
    path : Union[str, os.PathLike]
        Path to the PDF file.
    page_size_limit : Optional[int]
        Maximum characters allowed per page (raises error if exceeded).
    page_range : Union[int, Tuple[int, int], None]
        1-based single page or inclusive tuple (start_page, end_page).

    Returns
    -------
    ParsedText
        ParsedText object mapping 1-based page numbers ('1', '2') to text.
    """
    path_obj = Path(path)
    with path_obj.open("rb") as f:
        try:
            reader = pypdf.PdfReader(f)
        except pypdf.errors.PdfReadError as exc:
            raise ValueError(f"PDF reading via pypdf failed on {path!r}: {exc}") from exc

        pages: Dict[str, str] = {}
        total_length = 0
        total_pages = len(reader.pages)
        target_range = resolve_page_range(page_range, total_pages)

        for i in target_range:
            page = reader.pages[i]
            extracted = page.extract_text() or ""
            text = clean_invalid_unicode(extracted)

            if page_size_limit and len(text) > page_size_limit:
                raise ValueError(
                    f"Page {i + 1} of {total_pages} has {len(text)} chars, exceeding limit of {page_size_limit}."
                )

            pages[str(i + 1)] = text
            total_length += len(text)

    lib_ver = getattr(pypdf, "__version__", "unknown")
    metadata = ParsedMetadata(
        parsing_libraries=[f"pypdf ({lib_ver})"],
        total_parsed_text_length=total_length,
        count_parsed_media=0,
        name=f"pdf|page_range={page_range}|pages={len(pages)}",
    )
    return ParsedText(content=pages, metadata=metadata)


def parse_text(
    path: Union[str, os.PathLike],
    html: bool = False,
    page_size_limit: Optional[int] = None,
    **_: Any,
) -> ParsedText:
    """Parse plain text, markdown, or HTML file into a ParsedText object."""
    path_obj = Path(path)
    try:
        raw_text = path_obj.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = path_obj.read_text(encoding="utf-8", errors="ignore")

    parsing_libs = []
    if html:
        try:
            import html2text  # type: ignore
            converter = html2text.HTML2Text()
            converter.ignore_links = False
            converter.ignore_images = False
            converter.bodywidth = 0
            text = converter.handle(raw_text)
            parsing_libs.append("html2text")
        except ImportError:
            # Fallback regex strip for HTML tags
            text = re.sub(r"<[^>]+>", " ", raw_text)
            text = re.sub(r"\s+", " ", text).strip()
    else:
        text = raw_text

    clean_text = clean_invalid_unicode(text)
    if page_size_limit and len(clean_text) > page_size_limit:
        raise ValueError(f"Text in {path} exceeds page_size_limit of {page_size_limit}.")

    content: Dict[str, str] = {"1": clean_text}
    metadata = ParsedMetadata(
        parsing_libraries=parsing_libs or ["built-in"],
        total_parsed_text_length=len(clean_text),
        name=f"text|html={html}",
    )
    return ParsedText(content=content, metadata=metadata)


def _make_chunk(
    parsed_text: ParsedText,
    doc: Doc,
    text: str,
    lower_page: str,
    upper_page: str,
) -> Text:
    """Construct a Text chunk with pretty page formatting."""
    if lower_page == upper_page:
        page_span = f"pages {lower_page}"
    else:
        page_span = f"pages {lower_page}-{upper_page}"

    chunk_name = f"{doc.docname} {page_span}"
    return Text(text=text, name=chunk_name, doc=doc)


def chunk_pdf(
    parsed_text: ParsedText,
    doc: Doc,
    chunk_chars: int = 3000,
    overlap: int = 200,
) -> List[Text]:
    """Slice parsed PDF pages into overlapping chunks while tracking exact page numbers."""
    if not isinstance(parsed_text.content, dict):
        raise TypeError(f"ParsedText.content must be dict, got {type(parsed_text.content)}.")

    if not parsed_text.content:
        return []

    pages: List[str] = []
    texts: List[Text] = []
    split: str = ""

    for page_num, page_contents in parsed_text.content.items():
        page_text = page_contents if isinstance(page_contents, str) else page_contents[0]
        split += page_text
        pages.append(page_num)

        while len(split) > chunk_chars:
            texts.append(_make_chunk(parsed_text, doc, split[:chunk_chars], pages[0], pages[-1]))
            split = split[chunk_chars - overlap:]
            pages = [page_num]

    if len(split) > overlap or not texts:
        texts.append(_make_chunk(parsed_text, doc, split[:chunk_chars], pages[0], pages[-1]))

    return texts


def chunk_text(
    parsed_text: ParsedText,
    doc: Doc,
    chunk_chars: int = 3000,
    overlap: int = 200,
) -> List[Text]:
    """Slice plain text document into overlapping chunks."""
    if isinstance(parsed_text.content, dict):
        full_text = "\n\n".join(
            v if isinstance(v, str) else v[0]
            for v in parsed_text.content.values()
        )
    else:
        full_text = str(parsed_text.content)

    if not full_text:
        return []

    texts: List[Text] = []
    total_len = len(full_text)
    step = max(chunk_chars - overlap, 1)

    start = 0
    chunk_idx = 1
    while start < total_len:
        end = min(start + chunk_chars, total_len)
        chunk_str = full_text[start:end]
        texts.append(
            Text(
                text=chunk_str,
                name=f"{doc.docname} chunk {chunk_idx}",
                doc=doc,
            )
        )
        if end >= total_len:
            break
        start += step
        chunk_idx += 1

    return texts


def read_doc(
    path: Union[str, os.PathLike],
    doc: Optional[Doc] = None,
    chunk_chars: int = 3000,
    overlap: int = 200,
    page_range: Union[int, Tuple[int, int], None] = None,
    **kwargs: Any,
) -> List[Text]:
    """Parse a document (.pdf, .txt, .md, .html) and slice into chunks.

    Parameters
    ----------
    path : Union[str, os.PathLike]
        Local file path.
    doc : Optional[Doc]
        Doc metadata object. If None, automatically derived from file content and name.
    chunk_chars : int
        Target character count per chunk.
    overlap : int
        Overlap character count between consecutive chunks.
    page_range : Union[int, Tuple[int, int], None]
        Optional page range for PDF files.

    Returns
    -------
    List[Text]
        List of Text chunk objects.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Document not found at {path}")

    # Build default Doc if not provided
    if doc is None:
        docname = path_obj.stem
        dockey = md5sum(path_obj)
        citation = f"{docname} ({path_obj.name})"
        doc = Doc(docname=docname, dockey=dockey, citation=citation)

    suffix = path_obj.suffix.lower()
    if suffix == ".pdf":
        parsed = parse_pdf_to_pages(path_obj, page_range=page_range, **kwargs)
        return chunk_pdf(parsed, doc, chunk_chars=chunk_chars, overlap=overlap)
    elif suffix in (".txt", ".md"):
        parsed = parse_text(path_obj, html=False, **kwargs)
        return chunk_pdf(parsed, doc, chunk_chars=chunk_chars, overlap=overlap)
    elif suffix == ".html":
        parsed = parse_text(path_obj, html=True, **kwargs)
        return chunk_pdf(parsed, doc, chunk_chars=chunk_chars, overlap=overlap)
    else:
        parsed = parse_text(path_obj, **kwargs)
        return chunk_text(parsed, doc, chunk_chars=chunk_chars, overlap=overlap)
