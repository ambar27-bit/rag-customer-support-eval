# test_rag_evaluation.py - LAYER 5: evaluate the REAL RAG system.
#
# This is the whole project coming together:
#   - We run the ACTUAL RAG system (real Ollama generator) on each golden.
#   - actual_output + retrieval_context come from the LIVE system (not stored).
#   - We score with the RAG TRIAD: contextual relevancy, faithfulness,
#     answer relevancy.
#   - @parametrize loops over the golden dataset, one test per question.

import pytest

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

from config import get_judge_model
from rag_system import RAGSystem
from generator import OllamaGenerator
from datasets.faq_goldens import FAQ_GOLDENS

import os

# Build the RAG system ONCE for the whole test module, using the REAL LLM
# generator (so we're evaluating genuine generated answers, not the mock).
DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "documents")
rag = RAGSystem(DOCS, generator=OllamaGenerator(), top_k=3)


@pytest.mark.parametrize("golden", FAQ_GOLDENS)
def test_rag_quality(golden):
    # 1. Run the REAL RAG system on the golden's input.
    #    Both the answer AND the retrieved context come from the live system.
    result = rag.query(golden.input)

    # 2. Build the test case from the golden + the live system output.
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=result.answer,                 # LIVE answer from the LLM
        expected_output=golden.expected_output,      # what we expected (the golden)
        retrieval_context=result.retrieval_context,  # chunks the system retrieved
    )

    # 3. Score with the RAG triad. Thresholds are set a bit lenient (0.4)
    #    because our local judge + keyword retrieval aren't production-grade;
    #    the goal is to SEE the scores and learn, not to hit perfection.
    metrics = [
        ContextualRelevancyMetric(threshold=0.4, model=get_judge_model()),
        FaithfulnessMetric(threshold=0.4, model=get_judge_model()),
        AnswerRelevancyMetric(threshold=0.4, model=get_judge_model()),
    ]
    assert_test(test_case, metrics)
