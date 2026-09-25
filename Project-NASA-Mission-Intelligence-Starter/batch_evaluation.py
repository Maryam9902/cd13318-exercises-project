import os
import json
from statistics import mean

import rag_client
import llm_client
import ragas_evaluator


def main():
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError(
            "Please set the OPENAI_API_KEY environment variable."
        )

    os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

    with open(
        "test_questions.json",
        "r",
        encoding="utf-8"
    ) as f:
        test_questions = json.load(f)

    backends = rag_client.discover_chroma_backends()

    if not backends:
        raise RuntimeError(
            "No ChromaDB backend found. Run embedding_pipeline.py first."
        )

    backend = next(iter(backends.values()))

    collection = rag_client.initialize_rag_system(
        backend["dir_path"],
        backend["collection"]
    )

    results = []

    for item in test_questions:
        question = item["question"]
        mission = item.get("mission")

        print("\n" + "=" * 70)
        print(f"Question {item['id']}: {question}")
        print(f"Category: {item['category']}")
        print(f"Mission: {mission}")

        retrieved = rag_client.retrieve_documents(
            collection,
            question,
            n_results=3,
            mission_filter=mission
        )

        documents = []
        metadatas = []

        if retrieved and retrieved.get("documents"):
            documents = retrieved["documents"][0]

        if retrieved and retrieved.get("metadatas"):
            metadatas = retrieved["metadatas"][0]

        context = rag_client.format_context(
            documents,
            metadatas
        )

        answer = llm_client.generate_response(
            openai_key=openai_key,
            user_message=question,
            context=context,
            conversation_history=[]
        )

        scores = ragas_evaluator.evaluate_response_quality(
            question,
            answer,
            documents
        )

        print("\nAnswer:")
        print(answer)

        print("\nScores:")
        print(scores)

        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "mission": mission,
                "question": question,
                "answer": answer,
                "scores": scores
            }
        )

    metric_values = {}

    for result in results:
        for metric, value in result["scores"].items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(
                    metric,
                    []
                ).append(value)

    averages = {
        metric: mean(values)
        for metric, values in metric_values.items()
        if values
    }

    print("\n" + "=" * 70)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 70)

    for metric, score in averages.items():
        print(f"{metric}: {score:.4f}")

    output = {
        "questions_evaluated": len(results),
        "results": results,
        "average_metrics": averages
    }

    with open(
        "evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2
        )

    print("\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    main()
