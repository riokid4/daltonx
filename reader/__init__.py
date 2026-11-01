# -*- coding: utf-8 -*-
"""AI Safety Literature Reader and End-to-End Pipeline Package for DaltonX."""

from typing import Any, Dict, List, Optional, Union

from daltonx.assembly.incremental import IncrementalAssembler
from daltonx.assembly.operations import AssemblyPipeline
from daltonx.exporters.cag import CAGAssembler
from daltonx.reader.extractor import LLMExtractor, Passage, chunk_text
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


__all__ = [
    'AISafetyProcessor',
    'LLMExtractor',
    'Passage',
    'chunk_text',
    'process_text_to_graph',
]
