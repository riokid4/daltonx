# -*- coding: utf-8 -*-
"""AI Safety Literature Reader and End-to-End Pipeline Package for DaltonX."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from daltonx.assembly.incremental import IncrementalAssembler
from daltonx.assembly.operations import AssemblyPipeline
from daltonx.exporters.cag import CAGAssembler
from daltonx.reader.extractor import LLMExtractor, Passage, chunk_text
from daltonx.reader.pdf_reader import (
    Doc,
    ParsedMetadata,
    ParsedText,
    Text,
    _make_chunk,
    chunk_pdf,
    clean_invalid_unicode,
    md5sum,
    parse_pdf_to_pages,
    parse_text,
    read_doc,
)
from daltonx.reader.processor import AISafetyProcessor


def process_text_to_graph(
    text_or_passages: Union[str, List[Any]],
    metadata: Optional[Dict[str, Any]] = None,
    assembler: Optional[IncrementalAssembler] = None,
    pipeline: Optional[AssemblyPipeline] = None,
    extractor: Optional[LLMExtractor] = None,
    processor: Optional[AISafetyProcessor] = None,
    grounding_ontology: str = 'AIS',
    grounding_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """High-level end-to-end pipeline: raw text/passages -> grounded statements -> assembly -> CAG/CytoscapeJS.

    Parameters
    ----------
    text_or_passages : Union[str, List[Any]]
        Raw scientific text, document abstract, or pre-chunked passages.
    metadata : Optional[Dict[str, Any]]
        Document metadata (paper_id, title, citation, section).
    assembler : Optional[IncrementalAssembler]
        Existing or new IncrementalAssembler instance.
    pipeline : Optional[AssemblyPipeline]
        Assembly preprocessing pipeline.
    extractor : Optional[LLMExtractor]
        Extractor instance. If None, default LLMExtractor is created.
    processor : Optional[AISafetyProcessor]
        Processor instance. If None, default AISafetyProcessor is created.
    grounding_ontology : str
        Ontology namespace for CAG node labels (default: 'AIS').
    grounding_threshold : Optional[float]
        Grounding threshold for CAG node labels.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'raw_statements': list of extracted statements before assembly
        - 'statements': list of assembled statements
        - 'delta': AssemblyDelta from the incremental assembler
        - 'assembler': the IncrementalAssembler instance
        - 'cag_assembler': CAGAssembler instance
        - 'cag': networkx.MultiDiGraph CAG model
        - 'cytoscapejs': CytoscapeJS JSON-compatible dictionary
    """
    if extractor is None:
        extractor = LLMExtractor(processor=processor)
    elif processor is not None:
        extractor.processor = processor

    # 1. Extraction: Text -> grounded Influence statements
    raw_stmts = extractor.extract_statements(text_or_passages, metadata=metadata)

    # 2. Pipeline preprocessing (if specified)
    if pipeline is not None:
        prepared_stmts = pipeline.run(raw_stmts)
    else:
        prepared_stmts = raw_stmts

    # 3. Incremental assembly
    if assembler is None:
        assembler = IncrementalAssembler()

    delta = assembler.add_statements(prepared_stmts)
    assembled_stmts = assembler.get_statements()

    # 4. CAG and CytoscapeJS export
    cag_assembler = CAGAssembler(assembled_stmts)
    cag_graph = cag_assembler.make_model(
        grounding_ontology=grounding_ontology,
        grounding_threshold=grounding_threshold,
    )
    cyjs = cag_assembler.export_to_cytoscapejs()

    return {
        'raw_statements': raw_stmts,
        'statements': assembled_stmts,
        'delta': delta,
        'assembler': assembler,
        'cag_assembler': cag_assembler,
        'cag': cag_graph,
        'cytoscapejs': cyjs,
    }


def process_pdf_to_graph(
    pdf_path_or_dir: Union[str, os.PathLike, Path],
    metadata: Optional[Dict[str, Any]] = None,
    assembler: Optional[IncrementalAssembler] = None,
    pipeline: Optional[AssemblyPipeline] = None,
    extractor: Optional[LLMExtractor] = None,
    processor: Optional[AISafetyProcessor] = None,
    grounding_ontology: str = 'AIS',
    grounding_threshold: Optional[float] = None,
    chunk_chars: int = 3000,
    overlap: int = 200,
    **reader_kwargs: Any,
) -> Dict[str, Any]:
    """End-to-end PDF/document ingestion: files/dirs -> page chunking -> extraction -> assembly -> CAG.

    Parameters
    ----------
    pdf_path_or_dir : Union[str, os.PathLike, Path]
        Path to a single .pdf, .txt, .md, or .html document, or directory containing documents.
    metadata : Optional[Dict[str, Any]]
        Optional default metadata overrides (e.g. title, citation).
    assembler : Optional[IncrementalAssembler]
        IncrementalAssembler instance to accumulate statements.
    pipeline : Optional[AssemblyPipeline]
        Assembly preprocessing pipeline.
    extractor : Optional[LLMExtractor]
        LLMExtractor instance.
    processor : Optional[AISafetyProcessor]
        AISafetyProcessor instance.
    grounding_ontology : str
        Ontology namespace for CAG nodes (default: 'AIS').
    grounding_threshold : Optional[float]
        Grounding threshold for CAG nodes.
    chunk_chars : int
        Chunk character size (default: 3000).
    overlap : int
        Character overlap between consecutive chunks (default: 200).
    **reader_kwargs : Any
        Additional keyword arguments passed to read_doc (e.g. page_range).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing assembled graph and components, with added 'passages'.
    """
    path_obj = Path(pdf_path_or_dir)
    if not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {pdf_path_or_dir}")

    supported_exts = {".pdf", ".txt", ".md", ".html"}
    if path_obj.is_dir():
        files = sorted(
            [p for p in path_obj.iterdir() if p.is_file() and p.suffix.lower() in supported_exts],
            key=lambda p: p.name,
        )
    else:
        files = [path_obj]

    all_passages: List[Passage] = []
    for file_path in files:
        doc_texts = read_doc(
            file_path,
            chunk_chars=chunk_chars,
            overlap=overlap,
            **reader_kwargs,
        )
        for t in doc_texts:
            passage = t.to_passage(chunk_index=len(all_passages))
            if metadata:
                # Merge any caller-provided global metadata without overwriting page_span
                for k, v in metadata.items():
                    if k not in passage.metadata or not passage.metadata[k]:
                        passage.metadata[k] = v
            all_passages.append(passage)

    result = process_text_to_graph(
        text_or_passages=all_passages,
        metadata=metadata,
        assembler=assembler,
        pipeline=pipeline,
        extractor=extractor,
        processor=processor,
        grounding_ontology=grounding_ontology,
        grounding_threshold=grounding_threshold,
    )
    result['passages'] = all_passages
    return result


__all__ = [
    'AISafetyProcessor',
    'LLMExtractor',
    'Passage',
    'chunk_text',
    'process_text_to_graph',
    'process_pdf_to_graph',
    'Doc',
    'Text',
    'ParsedText',
    'ParsedMetadata',
    'read_doc',
    'chunk_pdf',
    'parse_pdf_to_pages',
    'parse_text',
    '_make_chunk',
    'clean_invalid_unicode',
    'md5sum',
]
