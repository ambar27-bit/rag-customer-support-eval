# RAG Project — Study Guide (Architecture + OOP Concepts)

An interview-ready walkthrough of the RAG system we built and the Python/OOP
fundamentals it demonstrates.

---

## PART A — How RAG Works (the architecture)

**RAG = Retrieval-Augmented Generation.** It answers questions using YOUR
documents, something a plain LLM can't do (it wasn't trained on your docs).

Two core steps:
1. **Retrieval** — search the documents, pull the few relevant pieces.
2. **Generation** — feed those pieces + the question to an LLM, which WRITES
   the answer.

```
Question -> [RETRIEVE relevant chunks] -> [GENERATE answer from them] -> Answer
```

### The 5 layers we built

| Layer | File | Job |
|---|---|---|
| 0. Documents | `documents/*.md` | the knowledge base (5 docs → 32 chunks) |
| 1. Chunking | `chunker.py` | split docs into small, self-contained pieces |
| 2. Retrieval | `retriever.py` | given a question, find the most relevant chunks |
| 3. Generation | `generator.py` | LLM writes the answer from the chunks |
| 4. Assembly | `rag_system.py` | wire retrieve + generate into one `.query()` |
| 5. Evaluation | `tests/` | score the system with the RAG triad + goldens |

### Layer 1 — Chunking
- A **chunk** is one coherent piece of a document (we split on `## ` headers,
  so one section = one chunk).
- **Why:** you can't feed whole documents to the LLM per question — too much
  irrelevant text. Chunks let retrieval return only what's relevant.
- **Chunk size matters:** too big = irrelevant text drags in; too small = a
  fact gets split across chunks. A QA tests whether chunking keeps facts whole.

### Layer 2 — Retrieval
- **Keyword matching (what we built):** score each chunk by how many
  question-words it contains; return the top-k. Simple, no dependencies.
  - Limitation: matches WORDS, not MEANING. "Get my money back" won't match
    "refund" — a false negative.
- **Semantic / embedding search (what production uses):** convert text to
  vectors that capture meaning; match by meaning, so synonyms/paraphrases work.
- **top_k:** how many chunks to return (we used 3). A tuning decision: too few
  misses context, too many adds noise.

### Layer 3 — Generation
- The LLM takes a **prompt** = instruction + retrieved context + question, and
  writes the answer.
- The prompt wording ("use ONLY the context, else say you don't know") is a
  design choice that controls faithfulness/hallucination — and is testable.
- We built TWO generators (see Mocking below): a fast MockGenerator and a real
  OllamaGenerator.

### Layer 4 — Assembly (`RAGSystem`)
- `rag.query(question)` runs: retrieve → generate → return answer + context.
- Returns BOTH the answer and the `retrieval_context`, because evaluation needs
  the context to score faithfulness/contextual relevancy. "Testable by design."

### Layer 5 — Evaluation (the RAG triad)
- **Contextual Relevancy** → did retrieval pull the RIGHT chunks?
- **Faithfulness** → is the answer grounded in those chunks (no contradictions)?
- **Answer Relevancy** → does the answer address the question?
- Driven by a **golden dataset** (test questions + expected answers), looped
  over with pytest `@parametrize`. `actual_output` + `retrieval_context` come
  from running the REAL system live.

---

## PART B — OOP Concepts Used (interview favourites)

This project is a great OOP showcase. Know these terms and point to where each
appears.

### 1. Class & Object (instance)
- A **class** is a blueprint; an **object** is a thing built from it.
- `class Chunk:` is a blueprint; `Chunk(text=..., source=...)` builds an object.
- Python: no `new` keyword — just call the class name. Constructor is `__init__`.
- Examples in project: `Chunk`, `Retriever`, `MockGenerator`, `OllamaGenerator`,
  `RAGSystem`, `RAGResult`.

### 2. Constructor (`__init__`) and `self`
- `__init__` runs automatically when you create an object; it sets up initial
  data. `self` = "this particular object" (like `this` in Java/TS), and must be
  the first parameter of every method.
- Example:
  ```python
  class Retriever:
      def __init__(self, chunks):
          self.chunks = chunks        # store data on this object
  ```

### 3. Encapsulation
- Bundling data + behaviour together and hiding internal details behind a simple
  interface.
- `RAGSystem.query()` hides the retrieve→generate steps; the caller just asks a
  question. The `_score()` method is marked "internal" with a leading underscore.

### 4. Composition ("has-a")
- Building a bigger object out of smaller ones, instead of doing everything in
  one class.
- `RAGSystem` HAS-A retriever and HAS-A generator; it coordinates them rather
  than re-implementing their logic. Each piece has one job.

### 5. Dependency Injection
- Passing a dependency INTO an object instead of hardcoding it, so it can be
  swapped.
- `RAGSystem(docs, generator=OllamaGenerator())` — the generator is injected.
  Pass the mock or the real one; `RAGSystem` doesn't care.

### 6. Polymorphism / "duck typing"
- Different classes that share the same method can be used interchangeably.
- `MockGenerator` and `OllamaGenerator` both have `.generate(question, chunks)`,
  so `RAGSystem` uses either without knowing which. "If it has `.generate()`, I
  can use it."

### 7. Mocking (a testing application of the above)
- A **mock** is a fake stand-in with the same interface as the real thing, used
  to make tests fast/reliable/controllable.
- `MockGenerator` replaces the slow, sometimes-crashing real LLM for fast demos
  and plumbing tests. Use the real one when evaluating actual answer quality.
- **When to mock vs use the real thing** is a QA judgement:
  - mock → structural/integration tests, demos (fast, deterministic)
  - real → quality evaluation with the RAG triad (needs genuine output)

### 8. Dunder methods (`__init__`, `__repr__`)
- Special double-underscore methods Python calls automatically.
- `__init__` = constructor. `__repr__` = how the object prints (we used it so
  `print(chunk)` shows a readable preview instead of `<object at 0x...>`).

---

## PART C — Python fundamentals reinforced

- **File I/O:** `with open(path, "r", encoding="utf-8") as f: f.read()` — the
  `with` auto-closes the file. Same pattern used to load goldens from files.
- **String methods:** `.split("## ")`, `.lower()`, `.strip()`, `.join()`,
  `.endswith(".md")`, slicing `text[:50]`, `.replace("\n", " ")`.
- **Lists & comprehensions:** `[chunk for (score, chunk) in scored if score>0]`.
- **Sorting with a key:** `scored.sort(key=lambda pair: pair[0], reverse=True)`
  — `lambda` is a one-line nameless function; `key` says what to sort by.
- **Slicing for top-k:** `results[:top_k]` — first N items; safe even if fewer.
- **Guard clauses / `continue`:** skip junk (non-.md files, empty sections) and
  keep looping. `continue` = skip this item; `break` = stop the whole loop.
- **Default + optional args:** `def retrieve(self, question, top_k=3)`.

---

## PART D — Interview talking points

- "I built a RAG system in 5 layers — chunking, retrieval, generation,
  assembly, evaluation — so I understand each piece, not a black box."
- "Retrieval was keyword-based for clarity; I can explain why production uses
  embeddings (meaning vs words) and the false-negative problem keywords have."
- "I made generation swappable via dependency injection — a MockGenerator for
  fast, reliable tests and a real Ollama generator for quality evaluation.
  Knowing when to mock is a QA skill."
- "The system returns both the answer and the retrieval_context, so it's
  testable by design — the RAG triad needs the context."
- "I evaluated with the RAG triad over a golden dataset and the evaluation
  caught real issues (faithfulness contradiction, chunking problem, weak
  out-of-scope handling) — see the Test Findings report."
