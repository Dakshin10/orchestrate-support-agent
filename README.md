<div align="center">

# 🧠 Multi-Domain Support Triage Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20BM25-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Coverage-HackerRank%20%7C%20Claude%20%7C%20Visa-6C63FF?style=for-the-badge" alt="Domains"/>
  <img src="https://img.shields.io/badge/Mode-Terminal%20%E2%80%A2%20Zero%20Dependencies%20on%20LLMs-00C896?style=for-the-badge" alt="Terminal"/>
  <img src="https://img.shields.io/badge/HackerRank-Orchestrate%20May%202026-00EA64?style=for-the-badge&logo=hackerrank&logoColor=white" alt="HackerRank Orchestrate"/>
</p>

> **A production-grade, fully offline support-triage pipeline** that ingests raw support tickets, classifies them across three enterprise domains, evaluates risk and safety, retrieves grounded documentation, and generates safe — or escalates sensitive — responses. No LLM API calls. No hallucinations.

</div>

---

## ✨ What Makes This Agent Different

| Capability | Detail |
|---|---|
| 🏷️ **Multi-signal classification** | Weighted n-gram taxonomy with TF-IDF normalization, exclusivity bonus & confidence calibration |
| 🔍 **BM25-inspired retrieval** | Dual-signal ranking (TF-IDF cosine + Jaccard-like lexical overlap) with length normalization |
| 🛡️ **4-tier safety engine** | Critical → High → Medium → Low risk matrix with diminishing-returns scoring & audit trail |
| 📝 **Grounded responses** | Sentence-level extraction from corpus — zero fabricated policies |
| ⚡ **Fast & offline** | Pure NumPy/sklearn pipeline; processes 100+ tickets per minute on a laptop |
| 📊 **Full observability** | Per-ticket confidence scores, processing time (ms), and escalation reasons exported to CSV |

---

## 🗂️ Repository Layout

```
orchestrate-support-agent/
│
├── code/
│   ├── main.py          # 🚀 Entry point — orchestrates the full pipeline
│   ├── classifier.py    # 🏷️  Multi-signal domain & request-type classifier
│   ├── retriever.py     # 🔍 BM25-inspired TF-IDF retriever with query expansion
│   ├── safety.py        # 🛡️  4-tier risk engine with confidence-adjusted thresholds
│   └── responder.py     # 📝 Sentence-extraction response generator
│
├── data/
│   ├── hackerrank/      # Scraped HackerRank support corpus (.txt / .md / .html)
│   ├── claude/          # Claude Help Center corpus
│   └── visa/            # Visa Support corpus
│
├── support_tickets/
│   ├── sample_support_tickets.csv   # Labelled examples for validation
│   ├── support_tickets.csv          # 🎯 Competition input — run the agent against this
│   └── output.csv                   # ✅ Agent-generated results
│
├── .env.example         # Copy → .env and fill in any optional API keys
├── AGENTS.md            # Agent-framework rules & logging contract
└── README.md            # You are here
```

---

## 🔬 Pipeline Architecture

```
                     ┌─────────────────────────────────────┐
                     │         support_tickets.csv          │
                     │  issue | subject | company           │
                     └──────────────────┬──────────────────┘
                                        │  for each row
                                        ▼
                          ┌─────────────────────────┐
                          │      🏷️  CLASSIFIER       │
                          │  ─────────────────────   │
                          │  • n-gram keyword match  │
                          │  • TF-IDF score per       │
                          │    domain                 │
                          │  • Exclusivity bonus      │
                          │  • Confidence 0 → 1       │
                          └────────────┬────────────┘
                                       │
                     ┌─────────────────▼──────────────────┐
                     │  Confidence < threshold?            │
                     │  YES ──► ESCALATE                  │
                     │  NO  ──► continue                  │
                     └─────────────────┬──────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │      🛡️  SAFETY ENGINE    │
                          │  ─────────────────────   │
                          │  Critical / High →       │
                          │    auto-escalate         │
                          │  Medium + low conf →     │
                          │    escalate              │
                          │  Low → pass through      │
                          └────────────┬────────────┘
                                       │
                     ┌─────────────────▼──────────────────┐
                     │  Safety triggered?                  │
                     │  YES ──► ESCALATE (with reason)    │
                     │  NO  ──► continue                  │
                     └─────────────────┬──────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │      🔍 RETRIEVER         │
                          │  ─────────────────────   │
                          │  • Domain-specific index │
                          │  • Query expansion       │
                          │  • BM25 normalization    │
                          │  • Lexical blend         │
                          │  • top-k chunks          │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │      📝 RESPONDER         │
                          │  ─────────────────────   │
                          │  • Sentence extraction   │
                          │  • Quality filtering     │
                          │  • Token overlap scoring │
                          └────────────┬────────────┘
                                       │
                     ┌─────────────────▼──────────────────┐
                     │           output.csv                │
                     │  status | product_area | response   │
                     │  justification | request_type       │
                     │  classification_confidence          │
                     │  response_confidence | proc_ms      │
                     └─────────────────────────────────────┘
```

---

## 🧩 Module Deep-Dive

### `classifier.py` — Domain & Intent Classification

Classifies every ticket into a **product domain** and **request type** using a weighted keyword taxonomy — no LLM required.

- **Domains**: `hackerRank_assessments` · `claude_platform` · `visa_payments` · `general_support`
- **Request types**: `bug` · `feature_request` · `billing_dispute` · `security` · `product_issue` · `invalid`
- **Confidence model**: match density + exclusivity bonus → calibrated score [0, 0.98]

---

### `retriever.py` — BM25-Inspired Document Retrieval

Builds per-domain TF-IDF indices at startup and retrieves the most relevant corpus chunks for each ticket.

| Feature | Implementation |
|---|---|
| Indexing | `TfidfVectorizer` with bigrams & sublinear TF, 8 000 max features |
| Ranking | `(1 - 0.35) × TF-IDF cosine + 0.35 × lexical overlap` |
| Length norm | BM25 `b = 0.4` normalization to prevent long-chunk bias |
| Query expansion | Suffix-stemming & synonym seeds (e.g., `payment → pay`, `api → endpoint`) |
| Chunking | 200-word windows, 40-word overlap, 15-word minimum |
| Fallback | Primary domain → `general_support` if no docs found |

---

### `safety.py` — 4-Tier Risk Engine

Every ticket is scanned against a compiled pattern library before any response is generated.

| Tier | Auto-Escalate? | Example Triggers |
|---|---|---|
| 🔴 **Critical** | Always | account compromised, identity theft, data breach |
| 🟠 **High** | Always | fraud, stolen card, phishing, hack |
| 🟡 **Medium** | If confidence < 0.5 **or** composite score ≥ 4.0 | refund, account locked, double charge |
| 🟢 **Low** | Never (score reducer) | "how do I", "documentation", "what is" |

Diminishing returns prevent score inflation when multiple patterns of the same tier fire.

---

### `responder.py` — Grounded Response Generation

Extracts the single best-matching sentence from retrieved corpus chunks — never invents information.

- Filters out boilerplate, navigation fragments, headings, and questions
- Scores remaining sentences by token overlap with the query
- Falls back to any well-formed sentence if no high-overlap match is found
- Ensures capitalization and terminal punctuation

---

## 🚀 Quick Start

### 1 — Prerequisites

```bash
pip install pandas numpy scikit-learn
```

### 2 — Prepare the corpus

Place scraped support documentation under `data/`:

```
data/
  hackerrank/   ← HackerRank support pages (.txt / .md / .html)
  claude/       ← Claude Help Center pages
  visa/         ← Visa Support pages
```

> The retriever auto-detects the domain from the directory name. Any `.txt`, `.md`, or `.html` file is indexed.

### 3 — Run the agent

```bash
cd code
python main.py
```

The agent reads `../support_tickets/support_tickets.csv`, processes every row through the full pipeline, and writes results to `../support_tickets/output.csv`.

**Sample terminal output:**
```
📚 Loading domain-specific support corpus...
✅ Retriever ready — 1 842 chunks across 3 domains.

[1] Processing...
[2] Processing...
...
[N] Processing...

✅ output.csv updated successfully!
```

---

## 📤 Output Schema

| Column | Type | Values |
|---|---|---|
| `status` | string | `replied` · `escalated` |
| `product_area` | string | `hackerRank_assessments` · `claude_platform` · `visa_payments` · `general_support` |
| `response` | string | User-facing answer or escalation message |
| `justification` | string | Why this decision was made |
| `request_type` | string | `product_issue` · `feature_request` · `bug` · `invalid` |
| `classification_confidence` | float | 0.0 – 1.0 |
| `response_confidence` | float | 0.0 – 1.0 |
| `processing_ms` | int | Wall-clock time per ticket in milliseconds |

---

## 🧠 Design Principles

1. **Corpus-grounded only** — the agent never uses outside knowledge or pre-trained LLM completions. Every response is extracted verbatim or synthesized exclusively from the provided documentation corpus.
2. **Fail safe** — when in doubt, escalate. Low confidence, missing docs, and critical risk patterns all route to a human agent.
3. **Full auditability** — every output row carries confidence scores, risk tier, and processing time so evaluators can trace every decision.
4. **Zero external calls** — the entire pipeline runs locally with no network dependency at inference time.

---

## 📋 Evaluation Criteria Alignment

| Criterion | How this agent addresses it |
|---|---|
| Correct `status` | Safety engine + confidence thresholds provide deterministic escalation logic |
| Correct `product_area` | Weighted n-gram taxonomy with exclusivity calibration |
| Grounded `response` | Sentence extraction — only corpus text, never hallucinated |
| Meaningful `justification` | Escalation reason or domain attribution appended to every row |
| Correct `request_type` | Regex pattern hierarchy ordered from most → least specific |

---

## 📜 License & Attribution

Built for the **HackerRank Orchestrate** hackathon (May 2026).  
Support corpus sourced from:
- [HackerRank Support](https://support.hackerrank.com/)
- [Claude Help Center](https://support.claude.com/en/)
- [Visa Support India](https://www.visa.co.in/support.html)

---

<div align="center">
  <sub>Built with 🔍 TF-IDF · 🛡️ Safety-first design · 📝 Corpus-grounded responses</sub>
</div>
