# -*- coding: utf-8 -*-
"""AI Safety Literature Extractor for DaltonX.

Provides clean text/passage chunking and LLMExtractor for extracting causal
claims from AI safety research literature with support for pluggable LLM
clients, custom extraction callbacks, and a built-in deterministic offline mode.
"""

from __future__ import absolute_import, print_function, unicode_literals

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union

from daltonx.reader.processor import AISafetyProcessor
from daltonx.statements import Influence

logger = logging.getLogger(__name__)


class Passage:
    """A passage/chunk of text with attached document metadata.

    Parameters
    ----------
    text : str
        Passage text content.
    metadata : Optional[Dict[str, Any]]
        Document metadata (e.g. paper_id, title, section, citation).
    chunk_index : int
        Sequential index of the chunk within the document.
    """

    def __init__(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_index: int = 0,
    ):
        self.text = text.strip()
        self.metadata = metadata or {}
        self.chunk_index = chunk_index

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'metadata': self.metadata,
            'chunk_index': self.chunk_index,
        }

    def __repr__(self) -> str:
        return f"Passage(chunk_index={self.chunk_index}, len={len(self.text)}, meta={self.metadata})"


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Passage]:
    """Split long paper text or abstract into overlapping passages.

    Preserves paragraph and sentence integrity whenever possible, attaching
    metadata to every chunk.

    Parameters
    ----------
    text : str
        The full text to split.
    chunk_size : int
        Target maximum character length per passage (default: 1000).
    overlap : int
        Character overlap between consecutive passages (default: 150).
    metadata : Optional[Dict[str, Any]]
        Metadata dict to attach to every chunk.

    Returns
    -------
    List[Passage]
        Ordered list of passage objects.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    # If already shorter than chunk_size, return single passage
    if len(clean_text) <= chunk_size:
        return [Passage(text=clean_text, metadata=dict(metadata or {}), chunk_index=0)]

    # Split by paragraphs first
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', clean_text) if p.strip()]

    passages: List[Passage] = []
    current_chunk = ""
    chunk_idx = 0

    for para in paragraphs:
        if not current_chunk:
            current_chunk = para
        elif len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk += "\n\n" + para
        else:
            # Save current chunk
            passages.append(Passage(text=current_chunk, metadata=dict(metadata or {}), chunk_index=chunk_idx))
            chunk_idx += 1

            # Compute overlap prefix from the tail of current_chunk
            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                # Try to break on sentence/space boundary in overlap
                space_pos = overlap_text.find(" ")
                if space_pos != -1:
                    overlap_text = overlap_text[space_pos + 1:]
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para

    if current_chunk.strip():
        passages.append(Passage(text=current_chunk, metadata=dict(metadata or {}), chunk_index=chunk_idx))

    return passages


EXTRACTION_SYSTEM_PROMPT = """You are an expert AI Safety Knowledge Base Extractor.
Analyze the scientific passage and extract all directed causal influence relationships between concepts.
Return ONLY valid JSON matching this schema:
{
  "causal_claims": [
    {
      "cause_text": "e.g. RLHF training",
      "cause_polarity": 1,
      "cause_adjectives": ["iterative"],
      "effect_text": "e.g. sycophantic responses",
      "effect_polarity": 1,
      "effect_adjectives": ["severe"],
      "evidence_text": "Exact sentence quote supporting this causal relationship",
      "model": "model name if mentioned",
      "model_scale": "parameter count / compute scale if mentioned",
      "training_paradigm": "e.g. RLHF, DPO, pretraining, fine-tuning",
      "architecture": "e.g. Transformer",
      "evaluation_setting": "benchmark or sandbox name",
      "empirical_vs_theoretical": "empirical or theoretical",
      "negated": false,
      "hedged": false,
      "confidence": 0.95
    }
  ]
}
Notes on polarity:
- cause_polarity: 1 for increase/presence, -1 for decrease/absence, null if unspecified.
- effect_polarity: 1 for increase/higher, -1 for decrease/lower/mitigated, null if unspecified.
- negated: true if the claim states the effect DOES NOT occur or failed to occur.
- hedged: true if the claim expresses uncertainty ('might', 'may', 'preliminary evidence suggests').
"""


class LLMExtractor:
    """Extracts structured causal claims from AI Safety literature using LLMs or offline rules.

    Parameters
    ----------
    client : Optional[Union[Callable, Any, str]]
        Pluggable backend:
        - Callable: function(prompt, passage_text) -> JSON string or list/dict.
        - Object with chat.completions.create (OpenAI, LiteLLM) or messages.create (Anthropic).
        - 'offline' or None: uses deterministic regex-based offline extraction rules.
    model : str
        Model name if using an API client (default: 'gpt-4o').
    processor : Optional[AISafetyProcessor]
        Processor instance for converting raw extraction dicts to Influence statements.
    temperature : float
        Sampling temperature (default: 0.0 for deterministic extraction).
    """

    def __init__(
        self,
        client: Optional[Union[Callable, Any, str]] = None,
        model: str = 'gpt-4o',
        processor: Optional[AISafetyProcessor] = None,
        temperature: float = 0.0,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.processor = processor or AISafetyProcessor()

    def extract_from_passage(self, passage: Union[Passage, str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract raw causal relation dictionaries from a passage."""
        if isinstance(passage, Passage):
            text = passage.text
            metadata = passage.metadata
        elif isinstance(passage, dict):
            text = passage.get('text', '')
            metadata = passage.get('metadata', {})
        else:
            text = str(passage)
            metadata = {}

        if not text.strip():
            return []

        # Determine backend
        if callable(self.client):
            raw_claims = self._call_callable_client(text)
        elif self.client and hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
            raw_claims = self._call_openai_compatible_client(text)
        elif self.client and hasattr(self.client, 'messages') and hasattr(self.client.messages, 'create'):
            raw_claims = self._call_anthropic_compatible_client(text)
        else:
            # Deterministic offline rule-based extraction
            raw_claims = self._offline_extract(text)

        # Merge passage metadata into each claim dict
        results: List[Dict[str, Any]] = []
        for claim in raw_claims:
            merged = dict(claim)
            for mkey, mval in metadata.items():
                if mkey not in merged or not merged[mkey]:
                    merged[mkey] = mval
            results.append(merged)

        return results

    def extract_statements(
        self,
        text_or_passages: Union[str, List[Any]],
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        overlap: int = 150,
    ) -> List[Influence]:
        """End-to-end extraction: chunks text, extracts claims, and returns grounded Influence statements.

        Parameters
        ----------
        text_or_passages : Union[str, List[Any]]
            Input document as full text string, list of passages, or list of text strings.
        metadata : Optional[Dict[str, Any]]
            Document metadata to attach (paper_id, title, citation, section).
        chunk_size : int
            Chunk size for text chunking if a raw string is passed.
        overlap : int
            Overlap for text chunking.

        Returns
        -------
        List[Influence]
            List of grounded DaltonX Influence statements.
        """
        if isinstance(text_or_passages, str):
            passages = chunk_text(text_or_passages, chunk_size=chunk_size, overlap=overlap, metadata=metadata)
        elif isinstance(text_or_passages, list):
            passages = []
            for idx, item in enumerate(text_or_passages):
                if isinstance(item, Passage):
                    passages.append(item)
                elif isinstance(item, dict):
                    passages.append(Passage(
                        text=item.get('text', ''),
                        metadata=item.get('metadata', metadata or {}),
                        chunk_index=idx,
                    ))
                else:
                    passages.append(Passage(
                        text=str(item),
                        metadata=dict(metadata or {}),
                        chunk_index=idx,
                    ))
        else:
            passages = []

        all_extractions: List[Dict[str, Any]] = []
        for p in passages:
            claims = self.extract_from_passage(p)
            all_extractions.extend(claims)

        # Pass extractions through AISafetyProcessor
        self.processor.extractions = all_extractions
        return self.processor.extract_causal_relations()

    # ------------------------------------------------------------------
    # Backend Callers
    # ------------------------------------------------------------------

    def _call_callable_client(self, text: str) -> List[Dict[str, Any]]:
        """Call a custom user-supplied extraction callable."""
        try:
            resp = self.client(EXTRACTION_SYSTEM_PROMPT, text)
            if isinstance(resp, list):
                return resp
            elif isinstance(resp, dict):
                return resp.get('causal_claims', [resp])
            elif isinstance(resp, str):
                return self._parse_json_response(resp)
        except Exception as e:
            logger.warning("Callable LLM client extraction failed: %s; falling back to offline", e)
        return self._offline_extract(text)

    def _call_openai_compatible_client(self, text: str) -> List[Dict[str, Any]]:
        """Call OpenAI/LiteLLM chat.completions client."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': EXTRACTION_SYSTEM_PROMPT},
                    {'role': 'user', 'content': f"Extract causal claims from this text:\n\n{text}"},
                ],
                temperature=self.temperature,
                response_format={'type': 'json_object'},
            )
            content = response.choices[0].message.content
            return self._parse_json_response(content)
        except Exception as e:
            logger.warning("OpenAI client extraction failed: %s; falling back to offline", e)
        return self._offline_extract(text)

    def _call_anthropic_compatible_client(self, text: str) -> List[Dict[str, Any]]:
        """Call Anthropic messages.create client."""
        try:
            response = self.client.messages.create(
                model=self.model,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[
                    {'role': 'user', 'content': f"Extract causal claims from this text:\n\n{text}"},
                ],
                temperature=self.temperature,
                max_tokens=2048,
            )
            content = response.content[0].text
            return self._parse_json_response(content)
        except Exception as e:
            logger.warning("Anthropic client extraction failed: %s; falling back to offline", e)
        return self._offline_extract(text)

    def _parse_json_response(self, content: str) -> List[Dict[str, Any]]:
        """Clean and parse JSON from an LLM response string."""
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-len("```")].strip()

        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get('causal_claims', [data])
        return []

    # ------------------------------------------------------------------
    # Deterministic Offline Rule Extractor
    # ------------------------------------------------------------------

    def _offline_extract(self, text: str) -> List[Dict[str, Any]]:
        """Deterministic offline rule-based causal extractor for AI Safety text.

        Parses common causal sentence patterns in AI safety papers without needing
        any external LLM API or network access.
        """
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        claims: List[Dict[str, Any]] = []

        pos_verbs = r'(?:increases?|enhances?|promotes?|causes?|leads?\s+to|induces?|exacerbates?|produces?|yields?)'
        neg_verbs = r'(?:decreases?|reduces?|mitigates?|prevents?|suppresses?|curbs?)'

        causal_regex = re.compile(
            rf'''(?i)
            (?P<cause>[\w\s\-]+?)\s+
            (?P<neg>does\s+not\s+|fails?\s+to\s+|cannot\s+|didn\'t\s+)?
            (?P<hedge>may\s+|might\s+|could\s+|can\s+)?
            (?P<adv>significantly\s+|substantially\s+)?
            (?P<verb>{pos_verbs}|{neg_verbs})\s+
            (?P<effect>[\w\s\-]+?)
            (?=\s+(?:in|on|across|for|at|under|when|where|while)\b|[.,;]|$)
            ''',
            re.VERBOSE,
        )

        model_pattern = re.compile(r'\b(GPT-4|Claude(?:\s*3)?|Llama(?:\s*3)?|Gemini|Mistral|Qwen|Transformer)\b', re.IGNORECASE)
        eval_pattern = re.compile(r'\b(TruthfulQA|MMLU|GSM8k|sandbox|eval|benchmark)\b', re.IGNORECASE)

        intro_prefix_pattern = re.compile(
            r'^(?:(?:we|our\s+experiments|results|studies|findings)\s+(?:show|find|observe|demonstrate|indicate|prove)\s+that\s+)',
            re.IGNORECASE,
        )

        for sent in sentences:
            clean_sent = intro_prefix_pattern.sub('', sent.strip())

            model_match = model_pattern.search(sent)
            model_name = model_match.group(1) if model_match else None

            eval_match = eval_pattern.search(sent)
            eval_setting = eval_match.group(1) if eval_match else None

            for match in causal_regex.finditer(clean_sent):
                raw_cause = match.group('cause').strip()
                raw_cause = intro_prefix_pattern.sub('', raw_cause)
                for art in ('the ', 'a ', 'an ', 'our ', 'this ', 'these ', 'those '):
                    if raw_cause.lower().startswith(art):
                        raw_cause = raw_cause[len(art):]

                raw_effect = match.group('effect').strip()
                for art in ('the ', 'a ', 'an ', 'our ', 'this ', 'these ', 'those '):
                    if raw_effect.lower().startswith(art):
                        raw_effect = raw_effect[len(art):]

                if not self._is_valid_concept_phrase(raw_cause) or not self._is_valid_concept_phrase(raw_effect):
                    continue

                verb_str = match.group('verb').lower()
                is_pos_verb = bool(re.match(pos_verbs, verb_str, re.IGNORECASE))
                is_negated = bool(match.group('neg')) or bool(re.search(r'\b(not|never|fails to)\b', sent, re.I))
                is_hedged = bool(match.group('hedge')) or bool(re.search(r'\b(may|might|could|suggests|potentially|preliminary)\b', sent, re.I))
                has_adv = bool(match.group('adv'))

                effect_pol = 1 if is_pos_verb else -1

                claims.append({
                    'cause_text': raw_cause,
                    'cause_polarity': 1,
                    'cause_adjectives': ['significant'] if has_adv else [],
                    'effect_text': raw_effect,
                    'effect_polarity': effect_pol,
                    'effect_adjectives': [],
                    'evidence_text': sent,
                    'model': model_name,
                    'evaluation_setting': eval_setting,
                    'empirical_vs_theoretical': 'empirical',
                    'negated': is_negated,
                    'hedged': is_hedged,
                    'confidence': 0.85 if not is_hedged else 0.65,
                })
        return claims

    @staticmethod
    def _is_valid_concept_phrase(phrase: str) -> bool:
        """Filter out garbage or pronoun matches."""
        p = phrase.strip().lower()
        stopwords = {'we', 'our', 'this', 'that', 'these', 'those', 'it', 'they', 'results', 'finding', 'there'}
        if p in stopwords or len(p) < 3 or len(p) > 60:
            return False
        # Strip leading articles
        for art in ('the ', 'a ', 'an ', 'our ', 'this '):
            if p.startswith(art):
                p = p[len(art):]
        return len(p) >= 3
