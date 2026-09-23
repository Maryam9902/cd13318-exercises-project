from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import (
        BleuScore,
        NonLLMContextPrecisionWithReference,
        ResponseRelevancy,
        Faithfulness,
        RougeScore
    )
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str]
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""

    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    try:
        # Create evaluator LLM
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0
            )
        )

        # Create evaluator embeddings
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
        )

        # Define metrics
        response_relevancy = ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )

        faithfulness = Faithfulness(
            llm=evaluator_llm
        )

        # Create evaluation sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

        # Evaluate metrics
        relevancy_score = response_relevancy.single_turn_score(sample)
        faithfulness_score = faithfulness.single_turn_score(sample)

        return {
            "answer_relevancy": float(relevancy_score),
            "faithfulness": float(faithfulness_score)
        }

    except Exception as e:
        return {
            "error": str(e)
        }
