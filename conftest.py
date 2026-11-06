# conftest.py - pytest auto-loads this. Makes the project root (and the rag/
# package) importable from tests, so we don't repeat sys.path lines everywhere.

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAG_PACKAGE = os.path.join(PROJECT_ROOT, "rag")

for path in (PROJECT_ROOT, RAG_PACKAGE):
    if path not in sys.path:
        sys.path.insert(0, path)
