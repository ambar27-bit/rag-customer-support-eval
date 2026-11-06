# generator.py - LAYER 3 of the RAG system: WRITE the answer.
#
# Generation = the step where an LLM takes the question + the retrieved chunks
# and WRITES a natural-language answer.
#
# SWAPPABLE DESIGN (dependency injection + mocking):
# We define TWO generators with the SAME method `.generate(question, chunks)`:
#   - MockGenerator   -> no LLM, instant canned answer (fast, reliable demos)
#   - OllamaGenerator -> calls the real local llama3.1 (real answers, slower)
# Because they share the same method, the RAGSystem can use EITHER without
# caring which — that's the whole point of a mock: a stand-in with the same
# interface as the real thing.


def _build_prompt(question: str, chunks: list) -> str:
    """Combine the question + retrieved chunks into a prompt for the LLM."""
    # Join each chunk's text into one context block, separated by blank lines.
    context = "\n\n".join(chunk.text for chunk in chunks)

    # The instruction. "Use ONLY the context" is what keeps the answer grounded
    # in the docs (reduces hallucination). This wording is a design choice a QA
    # would test — a weak prompt leads to unfaithful answers.
    prompt = (
        "You are a helpful customer-support assistant. "
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt


class MockGenerator:
    """A FAKE generator: no LLM. Returns a simple canned answer instantly.
    Great for fast demos and for testing the PLUMBING (does retrieval wire
    into generation correctly?) without waiting on a slow model."""

    def generate(self, question: str, chunks: list) -> str:
        if not chunks:
            return "I'm sorry, I couldn't find anything relevant to answer that."
        # Just echo back the most relevant chunk's text as the "answer".
        # (A real LLM would rephrase it; the mock keeps it trivial + instant.)
        return f"Based on our policy: {chunks[0].text}"


class OllamaGenerator:
    """The REAL generator: calls the local llama3.1 model via Ollama to write
    a proper answer from the context. Slower, but produces genuine output you
    can evaluate for faithfulness/relevancy."""

    def __init__(self, model_name: str = "llama3.1"):
        # Import inside __init__ so the ollama library is only needed when you
        # actually use the real generator (the mock works without it).
        import ollama
        self._ollama = ollama
        self._model_name = model_name

    def generate(self, question: str, chunks: list) -> str:
        if not chunks:
            return "I'm sorry, I couldn't find anything relevant to answer that."
        prompt = _build_prompt(question, chunks)
        # Send the prompt to the local model and return its written answer.
        response = self._ollama.generate(model=self._model_name, prompt=prompt)
        return response["response"].strip()


# ---- Quick manual check: compare mock vs real generation ----
if __name__ == "__main__":
    import os
    from chunker import load_and_chunk
    from retriever import Retriever

    here = os.path.dirname(os.path.abspath(__file__))
    docs = os.path.join(here, "..", "documents")

    chunks = load_and_chunk(docs)
    retriever = Retriever(chunks)

    question = "How long do I have to return electronics?"
    retrieved = retriever.retrieve(question, top_k=3)

    print(f"Question: {question}\n")

    print("--- MockGenerator (instant, canned) ---")
    mock = MockGenerator()
    print(mock.generate(question, retrieved))

    print("\n--- OllamaGenerator (real LLM, slower) ---")
    real = OllamaGenerator()
    print(real.generate(question, retrieved))
