# config.py - Shared configuration: the judge model for evaluation.
# Single source of truth — swap the judge here and every test picks it up.

from deepeval.models import OllamaModel


def get_judge_model():
    """The local Ollama model used to JUDGE the RAG system's answers."""
    return OllamaModel(
        model="llama3.1",
        base_url="http://localhost:11434",
        temperature=0,
    )
