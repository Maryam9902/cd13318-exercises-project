from typing import Dict, List, Optional

RAGAS_AVAILABLE = False
RAGAS_IMPORT_ERROR = None

try:
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from ragas.metrics import (
        ResponseRelevancy,
        Faithfulness,
        BleuScore
    )

    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings

    RAGAS_AVAILABLE = True

except Exception as e:
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = str(e)


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    reference_answer: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a RAG response using:

    1. Response Relevancy
    2. Faithfulness
    3. BLEU Score when a reference answer is available
    """

    if not RAGAS_AVAILABLE:
        return {
            "error": (
                "RAGAS not available: "
                f"{RAGAS_IMPORT_ERROR}"
            )
        }

    if not question:
        return {"error": "Question is empty"}

    if not answer:
        return {"error": "Answer is empty"}

    if not contexts:
        return {"error": "No retrieved contexts available"}

    try:
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0
            )
        )

        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
        )

        response_relevancy = ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )

        faithfulness = Faithfulness(
            llm=evaluator_llm
        )

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference_answer
        )

        relevancy_score = (
            response_relevancy.single_turn_score(sample)
        )

        faithfulness_score = (
            faithfulness.single_turn_score(sample)
        )

        scores = {
            "answer_relevancy": float(relevancy_score),
            "faithfulness": float(faithfulness_score)
        }

        # Additional documented RAGAS metric
        if reference_answer:
            bleu = BleuScore()

            bleu_score = bleu.single_turn_score(
                sample
            )

            scores["bleu_score"] = float(
                bleu_score
            )

        return scores

    except Exception as e:
        return {
            "error": (
                f"Evaluation failed: {str(e)}"
            )
        }
