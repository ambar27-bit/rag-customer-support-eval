# rag_system.py - LAYER 4: tie retrieval + generation into ONE system.
#
# This is the "assembly" of the RAG pipeline. It hides the internal steps
# (retrieve -> generate) behind a single simple method: query(question).
# The caller just asks a question and gets an answer + the context used.

from chunker import load_and_chunk
from retriever import Retriever
from generator import MockGenerator


class RAGResult:
    """Holds everything produced for one query: the answer AND the context
    that was retrieved. We keep the context because the EVALUATION step
    (RAG triad) needs retrieval_context to score faithfulness/contextual
    relevancy. Returning both is what makes the system testable."""
    def __init__(self, answer: str, retrieval_context: list):
        self.answer = answer                      # the generated answer (string)
        self.retrieval_context = retrieval_context  # list of chunk texts used


class RAGSystem:
    def __init__(self, documents_folder: str, generator=None, top_k: int = 3):
        # 1. Load + chunk the documents ONCE when the system is built.
        self.chunks = load_and_chunk(documents_folder)

        # 2. Build the retriever over those chunks.
        self.retriever = Retriever(self.chunks)

        # 3. The generator is INJECTED (swappable). Default to the mock so the
        #    system works fast/offline out of the box; pass OllamaGenerator()
        #    for real answers.
        self.generator = generator if generator is not None else MockGenerator()

        self.top_k = top_k

    def query(self, question: str) -> RAGResult:
        """The full RAG flow for one question: retrieve -> generate."""
        # STEP 1: RETRIEVAL — find the most relevant chunks.
        retrieved_chunks = self.retriever.retrieve(question, top_k=self.top_k)

        # STEP 2: GENERATION — write an answer from those chunks.
        answer = self.generator.generate(question, retrieved_chunks)

        # Package the answer + the context (as plain text) for the caller.
        # The evaluation step will use retrieval_context to score the system.
        context_texts = [chunk.text for chunk in retrieved_chunks]
        return RAGResult(answer=answer, retrieval_context=context_texts)


# ---- Quick manual check: ask the full system a few questions ----
if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    docs = os.path.join(here, "..", "documents")

    # Build the system (uses the fast MockGenerator by default).
    rag = RAGSystem(docs)

    questions = [
        "How long do I have to return electronics?",
        "What payment methods do you accept?",
        "How do I reset my password?",
    ]

    for q in questions:
        result = rag.query(q)
        print(f"\nQ: {q}")
        print(f"A: {result.answer[:150]}...")
        print(f"   (used {len(result.retrieval_context)} chunks)")
