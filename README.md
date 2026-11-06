# RAG Customer-Support Assistant + Evaluation

A Retrieval-Augmented Generation (RAG) system built from scratch in Python, plus
an automated evaluation suite that scores its quality with
[DeepEval](https://github.com/confident-ai/deepeval).

The goal of this project is to show **how a RAG system works from the inside** and
**how to test an AI system whose output is not simply right or wrong**. Every layer
(chunking, retrieval, generation) is written by hand rather than hidden behind a
framework, so each one can be tested on its own.

The knowledge base is a fictional e-commerce company, **NimbusCart** (returns,
shipping, accounts, orders, support policies): 5 documents, 32 chunks.

---

## How it works

```
Question -> [Retriever: find relevant chunks] -> [Generator: LLM writes the answer] -> Answer
```

| Layer | File | What it does |
|---|---|---|
| Documents | `documents/*.md` | The knowledge base |
| Chunking | `rag/chunker.py` | Splits each document into one chunk per `##` section |
| Retrieval | `rag/retriever.py` | Keyword scoring, returns the top-k chunks |
| Generation | `rag/generator.py` | `OllamaGenerator` (real LLM) and `MockGenerator` (fast fake) |
| Assembly | `rag/rag_system.py` | `RAGSystem.query()` returns the answer **and** the retrieved context |
| Evaluation | `tests/test_rag_evaluation.py` | Scores the live system against golden test cases |

### Design choices worth noting

- **Swappable generator (dependency injection):** `RAGSystem` accepts any object with
  a `.generate(question, chunks)` method. The mock is fast and deterministic for
  plumbing checks; the Ollama generator produces real answers for quality evaluation.
- **Testable by design:** `query()` returns both the answer and `retrieval_context`,
  because faithfulness and contextual relevancy cannot be scored without the context.
- **Goldens kept separate from test code:** test questions live in
  `datasets/faq_goldens.py`; adding a case means adding a row, not writing a test.

---

## Evaluation

Each golden question is run through the **real** system, then scored with the RAG triad:

| Metric | Checks |
|---|---|
| Contextual Relevancy | Did retrieval pull the right chunks? |
| Faithfulness | Is the answer supported by the retrieved context? |
| Answer Relevancy | Does the answer address the question? |

The judge is a local `llama3.1` model through Ollama (no paid API key needed).

### Results (5 golden questions, threshold 0.4)

2 passed, 3 failed. The failures are the useful part: they exposed three real
defects.

1. **Generation contradicted the source** (shipping question).
2. **Specific facts buried inside general chunks** (electronics return window).
3. **Weak handling of out-of-scope questions** (should say "I don't know").

See [`TEST_FINDINGS.md`](TEST_FINDINGS.md) for the full defect report with root
causes and recommended fixes, and [`STUDY_GUIDE.md`](STUDY_GUIDE.md) for the
architecture and the OOP concepts used.

> **Caveat:** the judge is a small local model, so scores are less reliable than a
> hosted model and can vary between runs. Read the judge's stated reason, not just
> the number. Results will not be identical on every machine.

---

## Run it

**Prerequisites:** Python 3.12, and [Ollama](https://ollama.com/download) installed.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the judge/generator model (about 5 GB, one time)
ollama pull llama3.1

# 3. Try each layer on its own
python rag/chunker.py        # shows the 32 chunks
python rag/retriever.py      # retrieval for a sample question
python rag/generator.py      # mock vs real generation side by side
python rag/rag_system.py     # the full pipeline

# 4. Run the evaluation (slow: every metric calls the local LLM)
python -m pytest tests/test_rag_evaluation.py -v
```

The evaluation can take several minutes because each golden question triggers a
real generation plus three LLM-judged metrics.

---

## Project structure

```
rag-project/
├── documents/            # NimbusCart knowledge base (5 markdown files)
├── rag/                  # chunker, retriever, generator, rag_system
├── datasets/             # golden test questions
├── tests/                # DeepEval RAG-triad evaluation
├── config.py             # judge model (single place to swap it)
├── conftest.py           # pytest import path setup
├── requirements.txt
├── STUDY_GUIDE.md        # architecture + OOP concepts
└── TEST_FINDINGS.md      # defect report from the evaluation
```

## Known limitations

- Retrieval is keyword based, so it matches words, not meaning. A question phrased
  with synonyms ("get my money back" vs "refund") can miss the right chunk.
  Embedding-based retrieval is the natural next step.
- The golden set is small (5 cases) and meant for demonstration.
- The three findings above are documented but not yet fixed in code.
