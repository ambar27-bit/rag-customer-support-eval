# faq_goldens.py - The GOLDEN DATASET (test data), separate from test code.
#
# Each golden = one test question + what a good answer should contain.
# Adding a test case = adding a row here. The test loop runs the REAL RAG
# system on each input and scores the result.
#
# As a QA, notice the MIX of cases: straightforward, a tricky specific fact,
# and an out-of-scope question (the system should admit it doesn't know).

from deepeval.dataset import Golden

FAQ_GOLDENS = [
    Golden(
        input="What payment methods do you accept?",
        expected_output="Visa, Mastercard, American Express, Discover, PayPal, "
                        "Apple Pay, and Google Pay. No checks, money orders, or crypto.",
    ),
    Golden(
        input="How do I reset my password?",
        expected_output="Click 'Forgot Password' on the login page and enter your "
                        "email to get a reset link valid for 30 minutes.",
    ),
    # TRICKY: electronics have a DIFFERENT (15-day) window than the general
    # 30-day rule. Tests whether retrieval + generation pick the SPECIFIC fact.
    Golden(
        input="How long do I have to return electronics?",
        expected_output="Electronics have a 15-day return window.",
    ),
    Golden(
        input="Is standard shipping free?",
        expected_output="Standard shipping is free on orders over 50 dollars; "
                        "otherwise it costs 4.99 dollars.",
    ),
    # OUT OF SCOPE: not in the docs. A good RAG system should say it doesn't
    # know, NOT make something up. This probes hallucination/faithfulness.
    Golden(
        input="Do you offer same-day grocery delivery?",
        expected_output="The system should indicate it doesn't have that information.",
    ),
]
