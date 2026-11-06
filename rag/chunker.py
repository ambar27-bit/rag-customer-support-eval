# chunker.py - LAYER 1 of the RAG system: turn documents into chunks.
#
# WHY CHUNKING?
# A RAG system can't feed whole documents to the LLM for every question --
# that's too much text and mostly irrelevant. So we split documents into
# smaller, self-contained pieces called "chunks". Later, retrieval finds
# only the few chunks relevant to a question.
#
# OUR STRATEGY:
# Our docs are Markdown with "## Section Headers". Each section is already a
# coherent topic (e.g. "Return window", "Refund processing time"), so we split
# on the "## " headers. One section = one chunk. This is called
# "header-based" (or "semantic") chunking.

import os                        # to work with file paths and folders


# A Chunk is one piece of a document. We use a small class so each chunk can
# carry BOTH its text AND where it came from (useful for debugging/testing:
# "which document did this answer come from?").
class Chunk:
    def __init__(self, text: str, source: str, heading: str):
        self.text = text         # the actual content of this chunk
        self.source = source     # which file it came from (e.g. "returns_refunds.md")
        self.heading = heading   # the section title (e.g. "Return window")

    def __repr__(self):
        # __repr__ controls how the object prints. This makes debugging readable:
        # instead of "<Chunk object at 0x...>" we see the heading + a preview.

        # self.text[:50] -> SLICE the first 50 characters of the chunk text
        #   (so long chunks don't flood the screen; we show just a preview).
        # .replace("\n", " ") -> swap any newline characters for spaces, so the
        #   preview stays on ONE line instead of wrapping to multiple lines.
        preview = self.text[:50].replace("\n", " ")

        # Build a readable one-line label using an f-string. {self.source},
        # {self.heading}, and {preview} get inserted into the text. The "..."
        # at the end hints that the text was cut off at 50 characters.
        return f"Chunk(source={self.source}, heading='{self.heading}', text='{preview}...')"


def chunk_markdown(text: str, source: str) -> list:
    """Split ONE markdown document's text into chunks, one per '## ' section."""
    chunks = []

    # Split the whole document wherever a "## " header appears.
    # Example: "## A\n...\n## B\n..."  ->  ["", "A\n...", "B\n..."]
    # (The first piece is whatever came before the first "## ", often the
    #  "# Title" line; we skip empty/irrelevant pieces below.)
    sections = text.split("## ")

    for section in sections:
        section = section.strip()        # remove leading/trailing whitespace
        if not section:                  # skip empty pieces (guard clause)
            continue

        # The first line of each section is its heading; the rest is the body.
        lines = section.split("\n", 1)   # split into [heading, rest] at most once
        heading = lines[0].strip()

        # If there's a body after the heading, use it; otherwise use the heading.
        body = lines[1].strip() if len(lines) > 1 else ""

        # Skip the document's top "# Title" line (it has no useful body and
        # isn't a real section). We detect it: no body means it's just a title.
        if not body:
            continue

        # The chunk text includes the heading + body so it's self-contained
        # (retrieval and the LLM both benefit from knowing the section topic).
        chunk_text = f"{heading}: {body}"
        chunks.append(Chunk(text=chunk_text, source=source, heading=heading))

    return chunks


def load_and_chunk(documents_folder: str) -> list:
    """Read every .md file in the folder and return ALL chunks combined."""
    all_chunks = []

    # List every file in the documents folder.
    for filename in os.listdir(documents_folder):
        if not filename.endswith(".md"):     # only process markdown files
            continue

        full_path = os.path.join(documents_folder, filename)

        # Read the whole file's text.
        with open(full_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Chunk this file and add its chunks to the master list.
        file_chunks = chunk_markdown(text, source=filename)
        all_chunks.extend(file_chunks)       # extend adds all items from a list

    return all_chunks


# ---- Quick manual check: run this file directly to see the chunks ----
if __name__ == "__main__":
    # Build the path to the documents folder (sits next to this rag/ folder).
    here = os.path.dirname(os.path.abspath(__file__))        # .../rag-project/rag
    docs = os.path.join(here, "..", "documents")             # .../rag-project/documents

    chunks = load_and_chunk(docs)
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks:
        print(c)
