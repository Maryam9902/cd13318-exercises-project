import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and "chroma" in d.name.lower()
    ]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            collections = client.list_collections()

            for collection in collections:
                collection_name = (
                    collection.name
                    if hasattr(collection, "name")
                    else str(collection)
                )

                key = f"{chroma_dir.name}_{collection_name}"

                try:
                    count = client.get_collection(collection_name).count()
                except Exception:
                    count = 0

                backends[key] = {
                    "dir_path": str(chroma_dir),
                    "collection": collection_name,
                    "display_name": f"{chroma_dir.name} - {collection_name}",
                    "doc_count": count
                }

        except Exception as e:
            backends[chroma_dir.name] = {
                "dir_path": str(chroma_dir),
                "collection": "",
                "display_name": f"{chroma_dir.name} - Error: {str(e)[:50]}",
                "doc_count": 0
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    return client.get_collection(name=collection_name)


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    where_filter = None

    if mission_filter and mission_filter.lower() != "all":
        where_filter = {"mission": mission_filter}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    context_parts = ["Relevant NASA Mission Context:"]

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1
    ):
        metadata = metadata or {}

        mission = metadata.get("mission", "Unknown Mission")
        mission = str(mission).replace("_", " ").title()

        source = metadata.get("source", "Unknown Source")

        category = metadata.get(
            "document_category",
            metadata.get("category", "Unknown")
        )
        category = str(category).replace("_", " ").title()

        source_header = (
            f"\nSource {i} | "
            f"Mission: {mission} | "
            f"Category: {category} | "
            f"Source: {source}"
        )

        context_parts.append(source_header)

        if len(document) > 2000:
            document = document[:2000] + "..."

        context_parts.append(document)

    return "\n".join(context_parts)
