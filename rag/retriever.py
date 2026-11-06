# retriever.py - LAYER 2 of the RAG system: find the relevant chunks.
#
# Given a question, search all chunks and return the few MOST RELEVANT ones.
# Those returned chunks become the "retrieval_context" for generation.
#
# APPROACH: keyword matching (simple + no dependencies, great for learning).
# We score each chunk by how many of the question's words appear in it, then
# return the top-scoring chunks. Real systems use embeddings (semantic search)
# for meaning-based matching -- we discuss that after you understand this.


class Retriever:
    def __init__(self, chunks: list):
        # Store the chunks we'll search through. The retriever is built ONCE
        # with all the chunks, then answers many questions against them.
        self.chunks = chunks

    def _score(self, question: str, chunk_text: str) -> int:
        """Count how many words from the question appear in the chunk."""
        # Lowercase both so matching is case-insensitive
        # ("Refund" should match "refund").
        question_words = question.lower().split()   # list of words in the question
        chunk_lower = chunk_text.lower()

        score = 0
        for word in question_words:
            # Skip very short/common words ("a", "to", "do") — they match
            # everything and add noise. (A simple "stop word" filter.)
            if len(word) <= 2:
                continue
            if word in chunk_lower:      # is this question-word present in the chunk?
                score += 1               # yes -> add 1 to the chunk's score
        return score

    def retrieve(self, question: str, top_k: int = 3) -> list:
        """Return the top_k most relevant chunks for the question."""
        # 1. Score every chunk. We build a list of (score, chunk) pairs.
        scored = []
        for chunk in self.chunks:
            score = self._score(question, chunk.text)
            scored.append((score, chunk))

        # 2. Sort by score, highest first.
        #    key=lambda pair: pair[0]  -> sort using the score (the 1st item of the pair).
        #    reverse=True              -> descending (biggest score first).
        scored.sort(key=lambda pair: pair[0], reverse=True)

        # 3. Keep only chunks that actually matched something (score > 0),
        #    then take the top_k of them.
        results = [chunk for (score, chunk) in scored if score > 0]
        return results[:top_k]      # slice the first top_k chunks


# ---- Quick manual check: run this file directly to test retrieval ----
if __name__ == "__main__":
    import os
    from chunker import load_and_chunk

    here = os.path.dirname(os.path.abspath(__file__))
    docs = os.path.join(here, "..", "documents")

    chunks = load_and_chunk(docs)
    retriever = Retriever(chunks)

    question = "How long do I have to return electronics?"
    results = retriever.retrieve(question, top_k=3)

    print(f"Question: {question}\n")
    print(f"Top {len(results)} chunks retrieved:")
    for i, chunk in enumerate(results, start=1):
        print(f"\n{i}. [{chunk.source} -> {chunk.heading}]")
        print(f"   {chunk.text[:120]}...")
