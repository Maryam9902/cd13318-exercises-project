import os
import json
from statistics import mean

import rag_client
import llm_client
import ragas_evaluator


def main():
    """Run batch RAG evaluation using the test question dataset."""

    # Get OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError(
            "Please set the OPENAI_API_KEY environment variable."
        )

    # Required by Chroma OpenAI embedding function
    os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

    # -----------------------------------------------------
    # Load evaluation dataset
    # -----------------------------------------------------

    with open(
        "test_questions.json",
        "r",
        encoding="utf-8"
    ) as f:
        test_questions = json.load(f)

    if not test_questions:
        raise ValueError(
            "No test questions found in test_questions.json."
        )

    print(
        f"Loaded {len(test_questions)} evaluation questions."
    )

    # -----------------------------------------------------
    # Discover ChromaDB backend
    # -----------------------------------------------------

    backends = rag_client.discover_chroma_backends()

    if not backends:
        raise RuntimeError(
            "No ChromaDB backend found. "
            "Run embedding_pipeline.py first."
        )

    # Use first available collection
    backend = next(
        iter(backends.values())
    )

    print(
        f"Using collection: "
        f"{backend['display_name']}"
    )

    # -----------------------------------------------------
    # Initialize collection
    # -----------------------------------------------------

    collection = rag_client.initialize_rag_system(
        backend["dir_path"],
        backend["collection"]
    )

    results = []

    # -----------------------------------------------------
    # Evaluate every question
    # -----------------------------------------------------

    for item in test_questions:

        question = item["question"]
        mission = item.get("mission")
        reference_answer = item.get(
            "reference_answer"
        )

        print("\n" + "=" * 70)

        print(
            f"Question {item['id']}: "
            f"{question}"
        )

        print(
            f"Category: "
            f"{item.get('category', 'unknown')}"
        )

        print(
            f"Mission: "
            f"{mission}"
        )

        # -------------------------------------------------
        # Retrieve documents
        # -------------------------------------------------

        retrieved = rag_client.retrieve_documents(
            collection,
            question,
            n_results=3,
            mission_filter=mission
        )

        documents = []
        metadatas = []

        if (
            retrieved
            and retrieved.get("documents")
            and retrieved["documents"][0]
        ):
            documents = (
                retrieved["documents"][0]
            )

        if (
            retrieved
            and retrieved.get("metadatas")
            and retrieved["metadatas"][0]
        ):
            metadatas = (
                retrieved["metadatas"][0]
            )

        # -------------------------------------------------
        # Build retrieved context
        # -------------------------------------------------

        context = rag_client.format_context(
            documents,
            metadatas
        )

        # -------------------------------------------------
        # Generate answer
        # -------------------------------------------------

        answer = llm_client.generate_response(
            openai_key=openai_key,
            user_message=question,
            context=context,
            conversation_history=[]
        )

        # -------------------------------------------------
        # RAGAS evaluation
        #
        # Metrics:
        # - Response Relevancy
        # - Faithfulness
        # - BLEU
        # -------------------------------------------------

        scores = (
            ragas_evaluator
            .evaluate_response_quality(
                question,
                answer,
                documents,
                reference_answer=reference_answer
            )
        )

        # -------------------------------------------------
        # Print individual result
        # -------------------------------------------------

        print("\nAnswer:")
        print(answer)

        print("\nReference Answer:")
        print(reference_answer)

        print("\nMetric Scores:")

        if "error" in scores:
            print(
                f"Evaluation error: "
                f"{scores['error']}"
            )
        else:
            for metric, score in scores.items():

                if isinstance(
                    score,
                    (int, float)
                ):
                    print(
                        f"{metric}: "
                        f"{score:.4f}"
                    )

        # Save result
        results.append(
            {
                "id": item["id"],
                "category": item.get(
                    "category"
                ),
                "mission": mission,
                "question": question,
                "reference_answer":
                    reference_answer,
                "answer": answer,
                "contexts": documents,
                "scores": scores
            }
        )

    # -----------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------

    metric_values = {}

    for result in results:

        scores = result.get(
            "scores",
            {}
        )

        for metric, value in scores.items():

            if isinstance(
                value,
                (int, float)
            ):

                metric_values.setdefault(
                    metric,
                    []
                ).append(value)

    averages = {}

    for metric, values in (
        metric_values.items()
    ):

        if values:

            averages[metric] = mean(
                values
            )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "BATCH EVALUATION SUMMARY"
    )

    print("=" * 70)

    print(
        f"Questions evaluated: "
        f"{len(results)}"
    )

    for metric, score in (
        averages.items()
    ):

        print(
            f"{metric} mean: "
            f"{score:.4f}"
        )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    output = {
        "questions_evaluated":
            len(results),

        "results":
            results,

        "average_metrics":
            averages
    }

    with open(
        "evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nResults saved to "
        "evaluation_results.json"
    )


if __name__ == "__main__":
    main()
