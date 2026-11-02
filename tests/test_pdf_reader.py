# -*- coding: utf-8 -*-
"""Unit tests for DaltonX Paper-QA Document and PDF Reader Integration."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pypdf
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from daltonx.reader.extractor import Passage
from daltonx.reader.pdf_reader import (
    Doc,
    ParsedMetadata,
    ParsedText,
    Text,
    _make_chunk,
    chunk_pdf,
    chunk_text,
    clean_invalid_unicode,
    md5sum,
    parse_pdf_to_pages,
    parse_text,
    read_doc,
    resolve_page_range,
)
from daltonx.reader import process_pdf_to_graph


def create_test_pdf_file(
    path: str,
    pages_text: list[str],
) -> None:
    """Helper to programmatically generate a multi-page PDF file with valid font and text stream."""
    writer = pypdf.PdfWriter()

    font = DictionaryObject({
        NameObject('/Type'): NameObject('/Font'),
        NameObject('/Subtype'): NameObject('/Type1'),
        NameObject('/BaseFont'): NameObject('/Helvetica'),
        NameObject('/Encoding'): NameObject('/WinAnsiEncoding'),
    })
    font_ref = writer._add_object(font)
    res = writer._add_object(DictionaryObject({
        NameObject('/Font'): DictionaryObject({
            NameObject('/F1'): font_ref
        })
    }))

    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject('/Resources')] = res
        stream = DecodedStreamObject()
        # Clean text of parentheses to keep PDF string simple
        safe_text = text.replace('(', '[').replace(')', ']')
        stream_bytes = f"BT /F1 12 Tf 72 700 Td ({safe_text}) Tj ET".encode('latin-1', errors='replace')
        stream.set_data(stream_bytes)
        page[NameObject('/Contents')] = writer._add_object(stream)

    with open(path, 'wb') as f:
        writer.write(f)


class TestPDFReader(unittest.TestCase):
    """Test suite for Paper-QA PDF reader and document processing mechanics."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_unicode_and_hash_utils(self):
        """Test Unicode cleaning and MD5 computation."""
        cleaned = clean_invalid_unicode("Hello\x00World\ud800Test")
        self.assertIn("Hello", cleaned)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\ud800", cleaned)

        txt_file = os.path.join(self.test_dir, "sample.txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("AI Safety Research Claim")
        self.assertTrue(len(md5sum(txt_file)) == 32)

    def test_resolve_page_range(self):
        """Test 1-indexed to 0-indexed page range conversion."""
        self.assertEqual(list(resolve_page_range(None, 5)), [0, 1, 2, 3, 4])
        self.assertEqual(list(resolve_page_range(1, 5)), [0])
        self.assertEqual(list(resolve_page_range(3, 5)), [2])
        self.assertEqual(list(resolve_page_range((2, 4), 5)), [1, 2, 3])

    def test_make_chunk_formatting(self):
        """Test _make_chunk page span formatting ('pages 1' vs 'pages 1-2')."""
        doc = Doc("Anthropic2024", "abc123md5", "Anthropic (2024)")
        parsed = ParsedText(content={"1": "Text on page 1", "2": "Text on page 2"})

        single_chunk = _make_chunk(parsed, doc, "Text on page 1", "1", "1")
        self.assertEqual(single_chunk.name, "Anthropic2024 pages 1")
        self.assertEqual(single_chunk.text, "Text on page 1")

        multi_chunk = _make_chunk(parsed, doc, "Combined text", "1", "2")
        self.assertEqual(multi_chunk.name, "Anthropic2024 pages 1-2")

        passage = multi_chunk.to_passage(chunk_index=0)
        self.assertIsInstance(passage, Passage)
        self.assertEqual(passage.metadata["page_span"], "pages 1-2")
        self.assertEqual(passage.metadata["docname"], "Anthropic2024")
        self.assertEqual(passage.metadata["dockey"], "abc123md5")
        self.assertEqual(passage.metadata["citation"], "Anthropic (2024)")

    def test_parse_pdf_to_pages(self):
        """Test extracting text page-by-page from a generated PDF."""
        pdf_path = os.path.join(self.test_dir, "test_paper.pdf")
        p1_text = "RLHF training increases sycophantic responses in large models."
        p2_text = "Sparse autoencoders reduce deceptive alignment in deep networks."
        create_test_pdf_file(pdf_path, [p1_text, p2_text])

        parsed = parse_pdf_to_pages(pdf_path)
        self.assertIn("1", parsed.content)
        self.assertIn("2", parsed.content)
        self.assertEqual(parsed.content["1"], p1_text)
        self.assertEqual(parsed.content["2"], p2_text)
        self.assertGreater(parsed.metadata.total_parsed_text_length, 50)

        # Test single page range
        parsed_p1 = parse_pdf_to_pages(pdf_path, page_range=1)
        self.assertEqual(list(parsed_p1.content.keys()), ["1"])

        # Test page size limit validation
        with self.assertRaises(ValueError):
            parse_pdf_to_pages(pdf_path, page_size_limit=10)

    def test_chunk_pdf_sliding_window(self):
        """Test PDF slicing preserves exact page numbers and overlaps."""
        doc = Doc("Hubinger2024", "dockey123", "Hubinger et al. (2024)")
        parsed = ParsedText(content={
            "1": "Page 1: Reinforcement learning from human feedback causes sycophancy. " * 3,
            "2": "Page 2: Sleeper agents persist through safety training and fine-tuning. " * 3,
            "3": "Page 3: Mechanistic interpretability uncovers hidden deceptive reasoning. " * 3,
        })

        # Large chunk size -> spans all 3 pages
        chunks_large = chunk_pdf(parsed, doc, chunk_chars=5000, overlap=100)
        self.assertEqual(len(chunks_large), 1)
        self.assertEqual(chunks_large[0].name, "Hubinger2024 pages 1-3")

        # Small chunk size -> splits across individual pages and overlapping spans
        chunks_small = chunk_pdf(parsed, doc, chunk_chars=120, overlap=30)
        self.assertGreater(len(chunks_small), 1)
        chunk_names = [c.name for c in chunks_small]
        self.assertTrue(any("pages 1" in name for name in chunk_names))
        self.assertTrue(any("pages 2" in name or "pages 1-2" in name for name in chunk_names))

    def test_read_doc_various_formats(self):
        """Test read_doc on PDF, TXT, MD, and HTML files."""
        # 1. PDF
        pdf_path = os.path.join(self.test_dir, "doc1.pdf")
        create_test_pdf_file(pdf_path, ["Direct preference optimization mitigates sycophancy."])
        pdf_chunks = read_doc(pdf_path)
        self.assertEqual(len(pdf_chunks), 1)
        self.assertTrue(pdf_chunks[0].name.startswith("doc1 pages 1"))

        # 2. Plain Text
        txt_path = os.path.join(self.test_dir, "doc2.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Scalable oversight decreases reward hacking in reinforcement learning.")
        txt_chunks = read_doc(txt_path)
        self.assertEqual(len(txt_chunks), 1)
        self.assertIn("reward hacking", txt_chunks[0].text)

        # 3. Markdown
        md_path = os.path.join(self.test_dir, "doc3.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Safety Paper\n\nRLHF training increases sycophantic responses.")
        md_chunks = read_doc(md_path)
        self.assertEqual(len(md_chunks), 1)

        # 4. HTML
        html_path = os.path.join(self.test_dir, "doc4.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><body><p>Iterative alignment prevents goal misgeneralization.</p></body></html>")
        html_chunks = read_doc(html_path)
        self.assertEqual(len(html_chunks), 1)
        self.assertIn("goal misgeneralization", html_chunks[0].text)

    def test_process_pdf_to_graph_single_file(self):
        """Test end-to-end process_pdf_to_graph on a multi-page PDF file."""
        pdf_path = os.path.join(self.test_dir, "safety_findings.pdf")
        p1 = "RLHF training increases sycophantic responses in large language models."
        p2 = "Sparse autoencoders reduce deceptive alignment in deep networks."
        create_test_pdf_file(pdf_path, [p1, p2])

        result = process_pdf_to_graph(pdf_path, chunk_chars=3000, overlap=100)
        self.assertIn("statements", result)
        self.assertIn("cag", result)
        self.assertIn("cytoscapejs", result)
        self.assertIn("passages", result)

        stmts = result["statements"]
        self.assertEqual(len(stmts), 2)

        # Check statement claims and page-span provenance preservation
        for s in stmts:
            self.assertGreater(len(s.evidence), 0)
            ev = s.evidence[0]
            # pmid should contain document and page span (e.g. 'safety_findings pages 1-2')
            self.assertIn("safety_findings pages", ev.pmid)
            self.assertIn("pages", ev.annotations.get("page_span", ""))

    def test_process_pdf_to_graph_directory(self):
        """Test process_pdf_to_graph on a directory of mixed files."""
        # File 1: PDF with RLHF -> sycophancy
        pdf_path = os.path.join(self.test_dir, "paper1.pdf")
        create_test_pdf_file(pdf_path, ["RLHF training increases sycophantic responses."])

        # File 2: TXT with DPO -> sycophancy mitigation
        txt_path = os.path.join(self.test_dir, "paper2.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Direct preference optimization reduces sycophantic responses.")

        result = process_pdf_to_graph(self.test_dir)
        stmts = result["statements"]
        self.assertGreaterEqual(len(stmts), 2)
        cyjs = result["cytoscapejs"]
        self.assertIn("nodes", cyjs)
        self.assertIn("edges", cyjs)
        self.assertGreater(len(cyjs["nodes"]), 0)
        self.assertGreater(len(cyjs["edges"]), 0)


if __name__ == "__main__":
    unittest.main()
