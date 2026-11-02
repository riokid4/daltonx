#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DaltonX Live Demo: AI Safety Literature Causal Knowledge Graph.

Extracts causal statements from realistic AI Safety research literature,
assembles them using DaltonX (concept grounding, deduplication, hierarchical
refinement, belief scoring), prints a terminal summary, and renders a
self-contained interactive Cytoscape.js web dashboard (ai_safety_graph.html).
"""

import json
import os
import subprocess
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from daltonx.assembly.incremental import IncrementalAssembler
from daltonx.assembly.operations import AssemblyPipeline
from daltonx.ontology import load_ai_safety_ontology
from daltonx.reader import (
    AISafetyProcessor,
    LLMExtractor,
    Passage,
    process_text_to_graph,
)

# ---------------------------------------------------------------------------
# 1. Realistic AI Safety Literature Passages (8 papers / replications)
# ---------------------------------------------------------------------------

DEMO_PASSAGES = [
    Passage(
        text=(
            "Paper 0 (Russell et al., UC Berkeley 2021 - Human Compatible):\n"
            "Foundational alignment research establishes that model alignment ensures artificial "
            "intelligence safety in advanced autonomous agents."
        ),
        metadata={
            'paper_id': 'Russell-2021-Foundations',
            'title': 'Human Compatible: Artificial Intelligence and the Problem of Control',
            'citation': 'Russell et al., 2021',
            'section': 'Theory',
        },
        chunk_index=0,
    ),
    Passage(
        text=(
            "Paper 1 (Bai et al., Anthropic 2023 - Constitutional AI):\n"
            "Our empirical evaluations demonstrate that scalable oversight significantly "
            "promotes AI safety in large language models by automating feedback loops."
        ),
        metadata={
            'paper_id': 'arXiv:2212.08073',
            'title': 'Constitutional AI: Harmlessness from AI Feedback',
            'citation': 'Bai et al., Anthropic 2023',
            'section': 'Results',
        },
        chunk_index=1,
    ),
    Passage(
        text=(
            "Paper 2 (Ouyang et al., OpenAI 2023 - InstructGPT):\n"
            "We observe that reinforcement learning from human feedback (RLHF training) "
            "significantly increases model alignment with user intent across benchmark tasks."
        ),
        metadata={
            'paper_id': 'arXiv:2203.02155',
            'title': 'Training language models to follow instructions with human feedback',
            'citation': 'Ouyang et al., OpenAI 2023',
            'section': 'Experiments',
        },
        chunk_index=2,
    ),
    Passage(
        text=(
            "Paper 3 (Sharma et al., NYU & Anthropic 2023 - Sycophancy):\n"
            "Despite alignment gains, iterative RLHF training exacerbates sycophantic responses "
            "where language models pander to human user misconceptions."
        ),
        metadata={
            'paper_id': 'arXiv:2310.13548',
            'title': 'Towards Understanding Sycophancy in Language Models',
            'citation': 'Sharma et al., 2023',
            'section': 'Discussion',
        },
        chunk_index=3,
    ),
    Passage(
        text=(
            "Paper 4 (Rafailov et al., Stanford 2023 - DPO):\n"
            "Direct preference optimization prevents reward hacking during policy optimization "
            "by eliminating the explicit reward model collapse."
        ),
        metadata={
            'paper_id': 'arXiv:2305.18290',
            'title': 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model',
            'citation': 'Rafailov et al., Stanford 2023',
            'section': 'Analysis',
        },
        chunk_index=4,
    ),
    Passage(
        text=(
            "Paper 5 (Bricken et al., Anthropic 2023 - Sparse Autoencoders):\n"
            "Monosemantic feature dictionaries extracted via sparse autoencoders significantly "
            "enhance mechanistic interpretability in frontier transformer representations."
        ),
        metadata={
            'paper_id': 'Anthropic-2023-SAE',
            'title': 'Towards Monosemanticity: Decomposing Language Models With Dictionary Learning',
            'citation': 'Bricken et al., Anthropic 2023',
            'section': 'Core Contributions',
        },
        chunk_index=5,
    ),
    Passage(
        text=(
            "Paper 6 (Hubinger et al., Anthropic 2024 - Sleeper Agents):\n"
            "Mechanistic interpretability mitigates deceptive alignment by revealing hidden "
            "chain-of-thought activations before safety guards are bypassed."
        ),
        metadata={
            'paper_id': 'arXiv:2401.05566',
            'title': 'Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training',
            'citation': 'Hubinger et al., Anthropic 2024',
            'section': 'Mitigations',
        },
        chunk_index=6,
    ),
    Passage(
        text=(
            "Paper 7 (DeepMind Alignment Team, 2024 - Independent Replication):\n"
            "Independent replication: Multi-agent debate and scalable oversight increases "
            "model safety across mathematical reasoning benchmarks."
        ),
        metadata={
            'paper_id': 'arXiv:2402.12000',
            'title': 'Scalable Oversight via Multi-Agent Debate and Verification',
            'citation': 'DeepMind Alignment Team, 2024',
            'section': 'Conclusion',
        },
        chunk_index=7,
    ),
    Passage(
        text=(
            "Paper 8 (Perez et al., 2022 - Red Teaming Analysis):\n"
            "Empirical red-teaming reveals that naive RLHF training fails to eliminate "
            "deceptive alignment when evaluation triggers remain out of distribution."
        ),
        metadata={
            'paper_id': 'arXiv:2202.03286',
            'title': 'Red Teaming Language Models with Language Models',
            'citation': 'Perez et al., 2022',
            'section': 'Failure Modes',
        },
        chunk_index=8,
    ),
]


# ---------------------------------------------------------------------------
# 2. Domain Extraction Callback (Simulates LLM extraction from literature)
# ---------------------------------------------------------------------------

def domain_literature_extractor(prompt, text):
    """High-fidelity extraction simulating an expert LLM causal reader."""
    if "Russell et al." in text:
        return [{
            'cause_text': 'model alignment',
            'cause_polarity': 1,
            'cause_adjectives': [],
            'effect_text': 'AI safety',
            'effect_polarity': 1,
            'effect_adjectives': [],
            'evidence_text': 'Foundational alignment research establishes that model alignment ensures AI safety.',
            'model': 'Theory',
            'training_paradigm': 'Foundational',
            'evaluation_setting': 'Theoretical Proof',
            'empirical_vs_theoretical': 'theoretical',
            'confidence': 0.85,
        }]
    elif "Bai et al." in text:
        return [{
            'cause_text': 'scalable oversight',
            'cause_polarity': 1,
            'cause_adjectives': ['automated'],
            'effect_text': 'AI safety',
            'effect_polarity': 1,
            'effect_adjectives': [],
            'evidence_text': 'Scalable oversight significantly promotes AI safety in large language models.',
            'model': 'Claude',
            'training_paradigm': 'Constitutional AI',
            'evaluation_setting': 'Harmlessness Evals',
            'empirical_vs_theoretical': 'empirical',
            'confidence': 0.90,
        }]
    elif "Ouyang et al." in text:
        return [{
            'cause_text': 'RLHF training',
            'cause_polarity': 1,
            'cause_adjectives': ['iterative'],
            'effect_text': 'AI alignment',
            'effect_polarity': 1,
            'effect_adjectives': [],
            'evidence_text': 'RLHF training significantly increases model alignment with user intent.',
            'model': 'InstructGPT',
            'training_paradigm': 'PPO',
            'evaluation_setting': 'Human Preference Benchmark',
            'empirical_vs_theoretical': 'empirical',
            'confidence': 0.88,
        }]
    elif "Sharma et al." in text:
        return [{
            'cause_text': 'RLHF training',
            'cause_polarity': 1,
            'cause_adjectives': ['iterative'],
            'effect_text': 'sycophantic responses',
            'effect_polarity': 1,
            'effect_adjectives': ['severe'],
            'evidence_text': 'Iterative RLHF training exacerbates sycophantic responses across instruction-tuned language models.',
            'model': 'LLaMA / Claude',
            'training_paradigm': 'RLHF',
            'evaluation_setting': 'TruthfulQA / Opinion Matching',
            'empirical_vs_theoretical': 'empirical',
            'confidence': 0.85,
        }]
    elif "Rafailov et al." in text:
        return [{
            'cause_text': 'direct preference optimization',
            'cause_polarity': 1,
            'cause_adjectives': [],
            'effect_text': 'reward hacking',
            'effect_polarity': -1,
            'effect_adjectives': [],
            'evidence_text': 'Direct preference optimization prevents reward hacking during policy optimization.',
            'model': 'Pythia / LLaMA',
            'training_paradigm': 'DPO',
            'evaluation_setting': 'TL;DR Summarization',
            'empirical_vs_theoretical': 'empirical',
            'confidence': 0.92,
        }]
    elif "Bricken et al." in text:
        return [{
            'cause_text': 'sparse autoencoders',
            'cause_polarity': 1,
            'cause_adjectives': ['monosemantic'],
            'effect_text': 'mechanistic interpretability',
            'effect_polarity': 1,
            'effect_adjectives': [],
            'evidence_text': 'Sparse autoencoders significantly enhance mechanistic interpretability in transformer representations.',
            'model': 'Transformer-1L',
            'training_paradigm': 'Unsupervised Dictionary Learning',
            'evaluation_setting': 'Feature Monosemanticity Scoring',
            'empirical_vs_theoretical': 'empirical',
            'confidence': 0.94,
        }]
    elif "Hubinger et al." in text:
        return [{
            'cause_text': 'mechanistic interpretability',
            'cause_polarity': 1,
            'cause_adjectives': [],
            'effect_text': 'deceptive alignment',
            'effect_polarity': -1,
            'effect_adjectives': [],
            'evidence_text': 'Mechanistic interpretability mitigates deceptive alignment by revealing hidden chain-of-thought activations.',
            'model': 'Claude 3 Sleeper Agent',
            'training_paradigm': 'Backdoor Training',
            'evaluation_setting': 'Code Vulnerability Insertion',
            'empirical_vs_theoretical': 'empirical',
            'confidence': 0.89,
        }]
    elif "DeepMind Alignment Team" in text:
        return [{
            # Duplicate of Paper 1 for deduplication & belief boosting
            'cause_text': 'scalable oversight',
            'cause_polarity': 1,
            'cause_adjectives': ['multi-agent'],
            'effect_text': 'AI safety',
            'effect_polarity': 1,
            'effect_adjectives': [],
            'evidence_text': 'Multi-agent debate and scalable oversight increases model safety across reasoning benchmarks.',
            'model': 'Gemini Ultra',
            'training_paradigm': 'Debate Verification',
            'evaluation_setting': 'GSM8K / MATH',
            'empirical_vs_theoretical': 'empirical',
            'confidence': 0.91,
        }]
    elif "Perez et al." in text:
        return [{
            # Negated / Contradicting claim
            'cause_text': 'RLHF training',
            'cause_polarity': 1,
            'cause_adjectives': ['standard'],
            'effect_text': 'deceptive alignment',
            'effect_polarity': -1,
            'effect_adjectives': [],
            'evidence_text': 'Empirical red-teaming reveals that naive RLHF training fails to eliminate deceptive alignment.',
            'model': 'Instruct Models',
            'training_paradigm': 'Red Teaming',
            'evaluation_setting': 'OOD Scenarios',
            'empirical_vs_theoretical': 'empirical',
            'negated': True,
            'confidence': 0.82,
        }]
    return []
    return []


# ---------------------------------------------------------------------------
# 3. HTML Dashboard Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DaltonX Causal Knowledge Graph | AI Safety Literature</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
  <style>
    :root {
      --bg: #0d1117;
      --panel: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-bright: #f0f6fc;
      --accent: #58a6ff;
      --green: #2ea043;
      --maroon: #f85149;
      --badge-bg: #21262d;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: 0 4px 12px rgba(0,0,0,0.4);
      z-index: 10;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand h1 {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text-bright);
      letter-spacing: -0.02em;
    }
    .brand .version {
      font-size: 0.72rem;
      background: rgba(88, 166, 255, 0.15);
      color: var(--accent);
      border: 1px solid rgba(88, 166, 255, 0.4);
      padding: 2px 7px;
      border-radius: 12px;
      font-weight: 600;
    }
    .stats {
      display: flex;
      gap: 16px;
    }
    .stat-badge {
      font-size: 0.8rem;
      background: var(--badge-bg);
      border: 1px solid var(--border);
      padding: 4px 10px;
      border-radius: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .stat-badge strong {
      color: var(--text-bright);
    }
    .toolbar {
      display: flex;
      gap: 8px;
    }
    button {
      background: var(--badge-bg);
      border: 1px solid var(--border);
      color: var(--text);
      font-size: 0.8rem;
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 500;
      transition: all 0.15s ease;
    }
    button:hover {
      background: #30363d;
      color: var(--text-bright);
      border-color: #8b949e;
    }
    main {
      display: flex;
      flex: 1;
      position: relative;
      overflow: hidden;
    }
    #cy {
      flex: 1;
      height: 100%;
      background: radial-gradient(circle at center, #131922 0%, #0d1117 100%);
    }
    #inspector {
      width: 420px;
      background: var(--panel);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      box-shadow: -4px 0 20px rgba(0,0,0,0.5);
    }
    .inspector-header {
      padding: 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .inspector-header h2 {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-bright);
    }
    .inspector-content {
      padding: 16px;
      font-size: 0.85rem;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .claim-card {
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .claim-title {
      font-weight: 600;
      color: var(--accent);
      font-size: 0.95rem;
    }
    .belief-bar-bg {
      height: 6px;
      background: #21262d;
      border-radius: 3px;
      overflow: hidden;
      margin-top: 4px;
    }
    .belief-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #2ea043, #58a6ff);
      border-radius: 3px;
    }
    .evidence-item {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 10px;
      font-size: 0.8rem;
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-top: 6px;
    }
    .evidence-quote {
      font-style: italic;
      color: #e6edf3;
      border-left: 2px solid var(--accent);
      padding-left: 8px;
      margin: 4px 0;
    }
    .badge {
      display: inline-block;
      font-size: 0.7rem;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
    }
    .badge-green { background: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid rgba(46, 160, 67, 0.4); }
    .badge-maroon { background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.4); }
    .badge-blue { background: rgba(88, 166, 255, 0.2); color: #58a6ff; border: 1px solid rgba(88, 166, 255, 0.4); }
    .badge-purple { background: rgba(187, 128, 247, 0.2); color: #d2a8ff; border: 1px solid rgba(187, 128, 247, 0.4); }
    .legend {
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: rgba(22, 27, 34, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 6px;
      pointer-events: none;
    }
    .legend-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .legend-color {
      width: 14px;
      height: 3px;
      border-radius: 2px;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h1>DaltonX Causal Knowledge Graph</h1>
      <span class="version">AI Safety v1.0</span>
    </div>
    <div class="stats">
      <div class="stat-badge">Concepts: <strong id="stat-nodes">0</strong></div>
      <div class="stat-badge">Causal Influences: <strong id="stat-edges">0</strong></div>
      <div class="stat-badge">Papers Ingested: <strong id="stat-papers">8</strong></div>
    </div>
    <div class="toolbar">
      <button onclick="cy.fit(null, 50)">Fit Graph</button>
      <button onclick="runLayout()">Re-run Layout</button>
    </div>
  </header>

  <main>
    <div id="cy"></div>
    <div class="legend">
      <div class="legend-row">
        <div class="legend-color" style="background:#2ea043;"></div>
        <span>Positive Reinforcement (+1 &rarr; +1)</span>
      </div>
      <div class="legend-row">
        <div class="legend-color" style="background:#f85149;"></div>
        <span>Mitigation / Adverse Side Effect (+1 &rarr; -1)</span>
      </div>
      <div class="legend-row">
        <div class="legend-color" style="background:#8b949e; border-top: 2px dashed #8b949e;"></div>
        <span>Negated / Inconclusive Claim</span>
      </div>
      <div class="legend-row">
        <span>Edge Thickness: Proportional to Belief Score</span>
      </div>
    </div>

    <aside id="inspector">
      <div class="inspector-header">
        <h2>Causal Claim Inspector</h2>
        <span class="badge badge-blue" id="inspector-badge">Click any edge or node</span>
      </div>
      <div class="inspector-content" id="inspector-body">
        <p style="color: #8b949e;">Click an edge or node in the graph to inspect exact paper citations, raw quotes, ontological groundings, and hierarchical refinement supports.</p>
      </div>
    </aside>
  </main>

  <script>
    const graphData = __GRAPH_DATA__;
    const statementsData = __STATEMENTS_DATA__;

    document.getElementById('stat-nodes').innerText = graphData.nodes.length;
    document.getElementById('stat-edges').innerText = graphData.edges.length;

    const cy = cytoscape({
      container: document.getElementById('cy'),
      elements: graphData,
      style: [
        {
          selector: 'node',
          style: {
            'label': 'data(id)',
            'color': '#f0f6fc',
            'font-family': '-apple-system, sans-serif',
            'font-size': '11px',
            'font-weight': 600,
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': '#161b22',
            'border-width': 2,
            'border-color': '#58a6ff',
            'padding': '12px',
            'width': 'label',
            'height': 'label',
            'shape': 'roundrectangle',
            'text-max-width': '120px',
            'text-wrap': 'wrap',
            'transition-property': 'border-color, border-width, background-color',
            'transition-duration': '0.15s'
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': '#f0883e',
            'border-width': 3,
            'background-color': '#21262d'
          }
        },
        {
          selector: 'edge',
          style: {
            'curve-style': 'bezier',
            'line-color': 'data(linecolor)',
            'target-arrow-color': 'data(linecolor)',
            'target-arrow-shape': 'data(targetArrowShape)',
            'line-style': 'data(linestyle)',
            'width': 'mapData(belief, 0.5, 1.0, 2, 7)',
            'opacity': 0.85,
            'arrow-scale': 1.2,
            'transition-property': 'width, opacity, line-color',
            'transition-duration': '0.15s'
          }
        },
        {
          selector: 'edge:selected',
          style: {
            'line-color': '#f0883e',
            'target-arrow-color': '#f0883e',
            'width': 6,
            'opacity': 1.0
          }
        }
      ],
      layout: {
        name: 'cose',
        idealEdgeLength: 140,
        nodeOverlap: 20,
        refresh: 20,
        fit: true,
        padding: 50,
        randomize: false,
        componentSpacing: 100,
        nodeRepulsion: 400000,
        edgeElasticity: 100,
        nestingFactor: 5,
        gravity: 80,
        numIter: 1000
      }
    });

    function runLayout() {
      cy.layout({ name: 'cose', animate: true, padding: 50 }).run();
    }

    // Node click inspector
    cy.on('tap', 'node', function(evt){
      const node = evt.target;
      const id = node.data('id');
      const badge = document.getElementById('inspector-badge');
      badge.innerText = 'Node Selected';
      badge.className = 'badge badge-purple';

      let inEdges = node.incomers('edge');
      let outEdges = node.outgoers('edge');

      let html = `
        <div class="claim-card">
          <div class="claim-title">${id}</div>
          <div style="font-size:0.75rem; color:#8b949e;">Ontology Node (AIS Namespace)</div>
          <div style="display:flex; gap:10px; margin-top:8px;">
            <span class="badge badge-blue">Causes: ${outEdges.length}</span>
            <span class="badge badge-green">Influenced by: ${inEdges.length}</span>
          </div>
        </div>
        <div style="font-weight:600; color:var(--text-bright); margin-top:10px;">Connected Causal Relations:</div>
      `;

      outEdges.forEach(e => {
        html += `<div style="font-size:0.8rem; padding:6px; background:#0d1117; border-radius:4px; margin-bottom:4px;">
          &rarr; Influences <strong style="color:var(--text-bright);">${e.data('target')}</strong> (Belief: ${(e.data('belief')*100).toFixed(1)}%)
        </div>`;
      });
      inEdges.forEach(e => {
        html += `<div style="font-size:0.8rem; padding:6px; background:#0d1117; border-radius:4px; margin-bottom:4px;">
          &larr; Influenced by <strong style="color:var(--text-bright);">${e.data('source')}</strong> (Belief: ${(e.data('belief')*100).toFixed(1)}%)
        </div>`;
      });

      document.getElementById('inspector-body').innerHTML = html;
    });

    // Edge click inspector
    cy.on('tap', 'edge', function(evt){
      const edge = evt.target;
      const d = edge.data();
      const badge = document.getElementById('inspector-badge');
      badge.innerText = 'Causal Influence Edge';
      badge.className = 'badge badge-green';

      const sKey = d.source.toLowerCase() + '_' + d.target.toLowerCase();
      const stmtInfo = statementsData[sKey] || {};
      const beliefPct = (d.belief * 100).toFixed(1);

      const polSign = d.obj_polarity === 1 ?
        '<span class="badge badge-green">+1 (Increases)</span>' :
        (d.obj_polarity === -1 ? '<span class="badge badge-maroon">-1 (Decreases / Mitigates)</span>' : '<span class="badge">Unspecified</span>');

      let html = `
        <div class="claim-card">
          <div class="claim-title">${d.source} &rarr; ${d.target}</div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px;">
            <span>Target Effect:</span>
            ${polSign}
          </div>
          <div style="margin-top:8px;">
            <div style="display:flex; justify-content:space-between; font-size:0.75rem;">
              <span>Assembled Belief Score:</span>
              <strong style="color:var(--accent);">${beliefPct}%</strong>
            </div>
            <div class="belief-bar-bg">
              <div class="belief-bar-fill" style="width: ${beliefPct}%"></div>
            </div>
          </div>
          <div style="font-size:0.75rem; color:#8b949e; margin-top:4px;">
            Supporting Literature Citations: ${stmtInfo.evidences ? stmtInfo.evidences.length : 1}
          </div>
        </div>
      `;

      if (stmtInfo.supported_by && stmtInfo.supported_by.length > 0) {
        html += `
          <div style="margin-top:8px;">
            <span style="font-weight:600; color:#e6edf3;">Hierarchical Refinements:</span>
            <div style="font-size:0.78rem; color:#8b949e; margin-top:2px;">
              Supported by ${stmtInfo.supported_by.length} more-specific child claim(s):
            </div>
            ${stmtInfo.supported_by.map(c => `<div style="padding:4px 8px; background:#0d1117; border-left:2px solid #58a6ff; font-size:0.78rem; margin-top:4px;">${c}</div>`).join('')}
          </div>
        `;
      }

      if (stmtInfo.evidences) {
        html += `<div style="font-weight:600; color:var(--text-bright); margin-top:12px;">Supporting Literature Evidence:</div>`;
        stmtInfo.evidences.forEach((ev, idx) => {
          html += `
            <div class="evidence-item">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong style="color:var(--accent);">${ev.citation || 'Unknown Source'}</strong>
                <span class="badge badge-blue">${ev.pmid || 'Ref #' + (idx+1)}</span>
              </div>
              <div class="evidence-quote">"${ev.text}"</div>
              ${ev.model ? `<div style="font-size:0.75rem; color:#8b949e;">Evaluated on: <strong>${ev.model}</strong> (${ev.training_paradigm || 'N/A'})</div>` : ''}
              ${ev.negated ? `<div style="font-size:0.75rem; color:#f85149; font-weight:600;">[NEGATED CLAIM: Effect reported absent / failed to occur]</div>` : ''}
            </div>
          `;
        });
      }

      document.getElementById('inspector-body').innerHTML = html;
    });
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 4. Main Execution Function
# ---------------------------------------------------------------------------

def run_live_demo():
    print("=" * 80)
    print(" DALTONX v1.0 LIVE DEMO: CAUSAL KNOWLEDGE GRAPH FOR AI SAFETY")
    print("=" * 80)

    onto = load_ai_safety_ontology()
    onto.initialize()
    print(f"\n[1] Loaded AISafetyOntology: {len(onto.nodes())} nodes, {len(onto.transitive_closure)} transitive closure relations.")

    print(f"[2] Processing {len(DEMO_PASSAGES)} scientific literature passages through LLMExtractor & AISafetyProcessor...")
    extractor = LLMExtractor(
        client=domain_literature_extractor,
        processor=AISafetyProcessor(ontology=onto, source_api='ais'),
    )
    pipeline = AssemblyPipeline()
    from daltonx.assembly.matches import matches_compositional
    assembler = IncrementalAssembler(ontology=onto, matches_fun=matches_compositional)

    result = process_text_to_graph(
        text_or_passages=DEMO_PASSAGES,
        extractor=extractor,
        pipeline=pipeline,
        assembler=assembler,
        grounding_ontology='AIS',
    )

    raw_stmts = result['raw_statements']
    assembled_stmts = result['statements']
    delta = result['delta']
    cag = result['cag']
    cyjs = result['cytoscapejs']

    print(f"    - Extracted Raw Causal Claims: {len(raw_stmts)}")
    print(f"    - Assembled Unique Influence Statements: {len(assembled_stmts)}")
    print(f"    - New Hierarchical Refinements: {len(delta.new_refinements)}")
    print(f"    - De-duplicated Multi-Source Evidences Merged: {sum(len(evs) for evs in delta.new_evidences.values())}\n")

    print("-" * 80)
    print(f"{'CAUSE':<24} | {'EFFECT':<25} | {'POLARITY':<10} | {'EVIDENCES':<9} | {'BELIEF':<6}")
    print("-" * 80)

    # Build metadata lookup for the interactive inspector
    statements_lookup = {}

    for s in assembled_stmts:
        subj_name = s.subj.concept.name.capitalize()
        obj_name = s.obj.concept.name.capitalize()
        pol_str = f"{s.subj.delta.polarity:+d} -> {s.obj.delta.polarity:+d}" if (s.subj.delta.polarity and s.obj.delta.polarity) else "None"
        belief_str = f"{s.belief:.3f}" if s.belief is not None else "1.000"
        ev_count = len(s.evidence)

        print(f"{subj_name:<24} | {obj_name:<25} | {pol_str:<10} | {ev_count:<9} | {belief_str:<6}")

        # Terminal refinement hierarchy inspection
        if s.supported_by:
            for child in s.supported_by:
                print(f"  └─ supported_by: {child.subj.concept.name.capitalize()} -> {child.obj.concept.name.capitalize()}")
        if s.supports:
            for parent in s.supports:
                print(f"  └─ supports:     {parent.subj.concept.name.capitalize()} -> {parent.obj.concept.name.capitalize()}")

        # Pack evidence details for HTML inspector
        s_key = f"{subj_name.lower()}_{obj_name.lower()}"
        ev_list = []
        for ev in s.evidence:
            ctx = ev.context
            model_name = ctx.model.name if (ctx and ctx.model) else None
            paradigm = ctx.environment.db_refs.get('training_paradigm') if (ctx and ctx.environment) else None
            ev_list.append({
                'pmid': ev.pmid,
                'citation': ev.annotations.get('citation', 'Paper Citation'),
                'text': ev.text,
                'model': model_name,
                'training_paradigm': paradigm,
                'negated': ev.epistemics.get('negated', False),
            })

        statements_lookup[s_key] = {
            'supported_by': [f"{c.subj.concept.name.capitalize()} &rarr; {c.obj.concept.name.capitalize()}" for c in s.supported_by],
            'supports': [f"{p.subj.concept.name.capitalize()} &rarr; {p.obj.concept.name.capitalize()}" for p in s.supports],
            'evidences': ev_list,
        }

    print("-" * 80)

    # -----------------------------------------------------------------------
    # Generate Interactive HTML File
    # -----------------------------------------------------------------------
    html_output_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'ai_safety_graph.html')
    )

    rendered_html = (
        HTML_TEMPLATE
        .replace("__GRAPH_DATA__", json.dumps(cyjs, indent=2))
        .replace("__STATEMENTS_DATA__", json.dumps(statements_lookup, indent=2))
    )

    with open(html_output_path, 'w', encoding='utf-8') as f:
        f.write(rendered_html)

    print(f"\n[3] Interactive Cytoscape.js HTML graph generated successfully:")
    print(f"    file://{html_output_path}")

    # -----------------------------------------------------------------------
    # Open in default macOS browser
    # -----------------------------------------------------------------------
    print(f"\n[4] Launching in default macOS browser (open {html_output_path})...")
    subprocess.run(['open', html_output_path], check=False)
    print("Demo completed successfully!\n")


if __name__ == '__main__':
    run_live_demo()
