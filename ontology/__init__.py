"""daltonx.ontology – Ontology engine for the AI-Safety Knowledge Graph.

Exports
-------
AISafetyOntology
    The main ontology class (ported 1-to-1 from WorldOntology / IndraOntology).
load_ai_safety_ontology
    Convenience factory that loads the bundled YAML or a custom path/URL.
"""

from .ontology import AISafetyOntology, load_ai_safety_ontology

__all__ = ["AISafetyOntology", "load_ai_safety_ontology"]
