from typing import Dict, List

RAGAS_AVAILABLE = False
RAGAS_IMPORT_ERROR = None

try:
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import ResponseRelevancy, Faithfulness

    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings

    RAGAS_AVAILABLE = True

except Exception as e:
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = str(e)


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str]
) -> Dict[str, float]:
    """
    Evaluate response quality using RAGAS metrics.
    """

    if not RAGAS_AVAILABLE:
        return {
            "error": (
                "RAGAS not available: "
                f"{RAGAS_IMPORT_ERROR}"
            )
        }

    if not question:
        return {
            "error": "Question is empty"
        }

    if not answer:
        return {
            "error": "Answer is empty"
        }

    if not contexts:
        return {
            "error": "No retrieved contexts available"
        }

    try:

        # Evaluator LLM
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0
            )
        )

        # Evaluator embeddings
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
        )

        # Response Relevancy metric
        response_relevancy = ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )

        # Faithfulness metric
        faithfulness = Faithfulness(
            llm=evaluator_llm
        )

        # Create RAGAS evaluation sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

        # Calculate scores
        relevancy_score = (
            response_relevancy.single_turn_score(
                sample
            )
        )

        faithfulness_score = (
            faithfulness.single_turn_score(
                sample
            )
        )

        return {
            "answer_relevancy":
                float(relevancy_score),

            "faithfulness":
                float(faithfulness_score)
        }

    except Exception as e:

        return {
            "error":
                f"Evaluation failed: {str(e)}"
        }
