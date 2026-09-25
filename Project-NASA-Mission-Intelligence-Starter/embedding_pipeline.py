#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders
and creates a permanent ChromaDB collection with OpenAI embeddings
for RAG applications.

Optimized to process only text files to avoid duplication with JSON versions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import time
from datetime import datetime
import argparse
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chroma_embedding_text_only.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Embedding Pipeline
# ---------------------------------------------------------

class ChromaEmbeddingPipelineTextOnly:
    """
    Pipeline for creating ChromaDB collections
    with OpenAI embeddings - Text files only.
    """

    def __init__(
        self,
        openai_api_key: str,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the embedding pipeline.
        """

        self.openai_api_key = openai_api_key
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # OpenAI client
        self.openai_client = OpenAI(
            api_key=openai_api_key
        )

        # Chroma embedding function
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=embedding_model
        )

        # Persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        # Create or retrieve collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

        logger.info(
            f"Initialized collection: {collection_name}"
        )


    # -----------------------------------------------------
    # Text Chunking
    # -----------------------------------------------------

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks with overlap
        and metadata.
        """

        if not text or not text.strip():
            return []

        text = text.strip()

        # Short text
        if len(text) <= self.chunk_size:

            chunk_metadata = metadata.copy()

            chunk_metadata["chunk_index"] = 0
            chunk_metadata["chunk_start"] = 0
            chunk_metadata["chunk_end"] = len(text)
            chunk_metadata["chunk_size"] = len(text)
            chunk_metadata["total_chunks"] = 1

            return [
                (text, chunk_metadata)
            ]

        chunks = []

        start = 0
        chunk_index = 0

        while start < len(text):

            end = min(
                start + self.chunk_size,
                len(text)
            )

            # Try to stop at sentence boundary
            if end < len(text):

                search_start = start + int(
                    self.chunk_size * 0.5
                )

                sentence_positions = [
                    text.rfind(". ", search_start, end),
                    text.rfind("? ", search_start, end),
                    text.rfind("! ", search_start, end),
                    text.rfind("\n", search_start, end)
                ]

                best_position = max(
                    sentence_positions
                )

                if best_position > start:
                    end = best_position + 1

            chunk = text[start:end].strip()

            if chunk:

                chunk_metadata = metadata.copy()

                chunk_metadata["chunk_index"] = chunk_index
                chunk_metadata["chunk_start"] = start
                chunk_metadata["chunk_end"] = end
                chunk_metadata["chunk_size"] = len(chunk)

                chunks.append(
                    (
                        chunk,
                        chunk_metadata
                    )
                )

                chunk_index += 1

            if end >= len(text):
                break

            next_start = end - self.chunk_overlap

            if next_start <= start:
                next_start = end

            start = next_start

        total_chunks = len(chunks)

        for _, chunk_metadata in chunks:
            chunk_metadata["total_chunks"] = total_chunks

        return chunks


    # -----------------------------------------------------
    # Document Exists
    # -----------------------------------------------------

    def check_document_exists(
        self,
        doc_id: str
    ) -> bool:
        """
        Check whether a document exists.
        """

        try:

            result = self.collection.get(
                ids=[doc_id]
            )

            return bool(
                result
                and result.get("ids")
            )

        except Exception as e:

            logger.error(
                f"Error checking document "
                f"{doc_id}: {e}"
            )

            return False


    # -----------------------------------------------------
    # Update Document
    # -----------------------------------------------------

    def update_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update an existing document.
        """

        try:

            embedding = self.get_embedding(
                text
            )

            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )

            logger.debug(
                f"Updated document: {doc_id}"
            )

            return True

        except Exception as e:

            logger.error(
                f"Error updating document "
                f"{doc_id}: {e}"
            )

            return False


    # -----------------------------------------------------
    # Delete by Source
    # -----------------------------------------------------

    def delete_documents_by_source(
        self,
        source_pattern: str
    ) -> int:
        """
        Delete documents matching a source pattern.
        """

        try:

            all_docs = self.collection.get()

            ids_to_delete = []

            metadatas = all_docs.get(
                "metadatas",
                []
            )

            ids = all_docs.get(
                "ids",
                []
            )

            for i, metadata in enumerate(
                metadatas
            ):

                if (
                    metadata
                    and
                    source_pattern
                    in metadata.get(
                        "source",
                        ""
                    )
                ):

                    ids_to_delete.append(
                        ids[i]
                    )

            if ids_to_delete:

                self.collection.delete(
                    ids=ids_to_delete
                )

                logger.info(
                    f"Deleted "
                    f"{len(ids_to_delete)} "
                    f"documents matching "
                    f"source pattern: "
                    f"{source_pattern}"
                )

                return len(
                    ids_to_delete
                )

            logger.info(
                "No documents found matching "
                f"source pattern: {source_pattern}"
            )

            return 0

        except Exception as e:

            logger.error(
                "Error deleting documents "
                f"by source: {e}"
            )

            return 0


    # -----------------------------------------------------
    # Get File Documents
    # -----------------------------------------------------

    def get_file_documents(
        self,
        file_path: Path
    ) -> List[str]:
        """
        Get all document IDs belonging
        to one exact file.
        """

        try:

            all_docs = self.collection.get()

            file_doc_ids = []

            target_path = str(
                file_path
            )

            for i, metadata in enumerate(
                all_docs.get(
                    "metadatas",
                    []
                )
            ):

                if (
                    metadata
                    and
                    metadata.get(
                        "file_path"
                    ) == target_path
                ):

                    file_doc_ids.append(
                        all_docs["ids"][i]
                    )

            return file_doc_ids

        except Exception as e:

            logger.error(
                "Error getting file "
                f"documents: {e}"
            )

            return []


    # -----------------------------------------------------
    # OpenAI Embedding
    # -----------------------------------------------------

    def get_embedding(
        self,
        text: str
    ) -> List[float]:
        """
        Get OpenAI embedding.
        """

        try:

            response = (
                self.openai_client
                .embeddings
                .create(
                    model=self.embedding_model,
                    input=text
                )
            )

            return (
                response
                .data[0]
                .embedding
            )

        except Exception as e:

            logger.error(
                "Error generating "
                f"embedding: {e}"
            )

            raise


    # -----------------------------------------------------
    # Stable Document ID
    # -----------------------------------------------------

    def generate_document_id(
        self,
        file_path: Path,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Generate stable document ID.

        Format:
        mission_source_chunk_0001
        """

        mission = metadata.get(
            "mission",
            "unknown"
        )

        source = metadata.get(
            "source",
            file_path.stem
        )

        chunk_index = metadata.get(
            "chunk_index",
            0
        )

        mission = str(
            mission
        ).replace(
            " ",
            "_"
        )

        source = str(
            source
        ).replace(
            " ",
            "_"
        )

        return (
            f"{mission}_"
            f"{source}_"
            f"chunk_"
            f"{chunk_index:04d}"
        )


    # -----------------------------------------------------
    # Process Text File
    # -----------------------------------------------------

    def process_text_file(
        self,
        file_path: Path
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Read and process a text file.
        """

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                content = f.read()

            if not content.strip():
                return []

            metadata = {

                "source":
                    file_path.stem,

                "file_path":
                    str(file_path),

                "file_type":
                    "text",

                "content_type":
                    "full_text",

                "mission":
                    self.extract_mission_from_path(
                        file_path
                    ),

                "data_type":
                    self.extract_data_type_from_path(
                        file_path
                    ),

                "document_category":
                    self.extract_document_category_from_filename(
                        file_path.name
                    ),

                "file_size":
                    len(content),

                "processed_timestamp":
                    datetime.now().isoformat()
            }

            return self.chunk_text(
                content,
                metadata
            )

        except Exception as e:

            logger.error(
                f"Error processing text "
                f"file {file_path}: {e}"
            )

            return []


    # -----------------------------------------------------
    # Mission Extraction
    # -----------------------------------------------------

    def extract_mission_from_path(
        self,
        file_path: Path
    ) -> str:

        path_str = str(
            file_path
        ).lower()

        if (
            "apollo11" in path_str
            or
            "apollo_11" in path_str
        ):
            return "apollo_11"

        elif (
            "apollo13" in path_str
            or
            "apollo_13" in path_str
        ):
            return "apollo_13"

        elif "challenger" in path_str:
            return "challenger"

        return "unknown"


    # -----------------------------------------------------
    # Data Type
    # -----------------------------------------------------

    def extract_data_type_from_path(
        self,
        file_path: Path
    ) -> str:

        path_str = str(
            file_path
        ).lower()

        if "transcript" in path_str:
            return "transcript"

        elif "textract" in path_str:
            return "textract_extracted"

        elif "audio" in path_str:
            return "audio_transcript"

        elif "flight_plan" in path_str:
            return "flight_plan"

        return "document"


    # -----------------------------------------------------
    # Document Category
    # -----------------------------------------------------

    def extract_document_category_from_filename(
        self,
        filename: str
    ) -> str:

        filename_lower = filename.lower()

        if "pao" in filename_lower:
            return "public_affairs_officer"

        elif "cm" in filename_lower:
            return "command_module"

        elif "tec" in filename_lower:
            return "technical"

        elif "flight_plan" in filename_lower:
            return "flight_plan"

        elif "mission_audio" in filename_lower:
            return "mission_audio"

        elif "ntrs" in filename_lower:
            return "nasa_archive"

        elif "19900066485" in filename_lower:
            return "technical_report"

        elif "19710015566" in filename_lower:
            return "mission_report"

        elif "full_text" in filename_lower:
            return "complete_document"

        return "general_document"


    # -----------------------------------------------------
    # Scan Files
    # -----------------------------------------------------

    def scan_text_files_only(
        self,
        base_path: str
    ) -> List[Path]:

        base_path = Path(
            base_path
        )

        files_to_process = []

        data_dirs = [
            "apollo11",
            "apollo13",
            "challenger"
        ]

        for data_dir in data_dirs:

            dir_path = (
                base_path
                / data_dir
            )

            if dir_path.exists():

                logger.info(
                    f"Scanning directory: "
                    f"{dir_path}"
                )

                text_files = list(
                    dir_path.glob(
                        "**/*.txt"
                    )
                )

                files_to_process.extend(
                    text_files
                )

                logger.info(
                    f"Found "
                    f"{len(text_files)} "
                    f"text files in "
                    f"{data_dir}"
                )

        filtered_files = []

        for file_path in files_to_process:

            if (
                file_path.name.startswith(".")
                or
                "summary"
                in file_path.name.lower()
                or
                file_path.suffix.lower()
                != ".txt"
            ):
                continue

            filtered_files.append(
                file_path
            )

        logger.info(
            f"Total text files to process: "
            f"{len(filtered_files)}"
        )

        mission_counts = {}

        for file_path in filtered_files:

            mission = (
                self.extract_mission_from_path(
                    file_path
                )
            )

            mission_counts[mission] = (
                mission_counts.get(
                    mission,
                    0
                )
                + 1
            )

        logger.info(
            "Files by mission:"
        )

        for mission, count in (
            mission_counts.items()
        ):

            logger.info(
                f"  {mission}: "
                f"{count} files"
            )

        return filtered_files


    # -----------------------------------------------------
    # Add Documents
    # -----------------------------------------------------

    def add_documents_to_collection(
        self,
        documents: List[
            Tuple[
                str,
                Dict[str, Any]
            ]
        ],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = "skip"
    ) -> Dict[str, int]:
        """
        Add documents to ChromaDB.

        update_mode:
        - skip
        - update
        - replace
        """

        if not documents:

            return {
                "added": 0,
                "updated": 0,
                "skipped": 0
            }

        stats = {
            "added": 0,
            "updated": 0,
            "skipped": 0
        }


        # -------------------------------------------------
        # SAFE REPLACE MODE
        # Delete only documents belonging to exact file.
        # -------------------------------------------------

        if update_mode == "replace":

            existing_docs = (
                self.collection.get()
            )

            ids_to_delete = []

            target_file_path = str(
                file_path
            )

            metadatas = (
                existing_docs.get(
                    "metadatas",
                    []
                )
            )

            existing_ids = (
                existing_docs.get(
                    "ids",
                    []
                )
            )

            for i, metadata in enumerate(
                metadatas
            ):

                if (
                    metadata
                    and
                    metadata.get(
                        "file_path"
                    ) == target_file_path
                ):

                    ids_to_delete.append(
                        existing_ids[i]
                    )

            if ids_to_delete:

                self.collection.delete(
                    ids=ids_to_delete
                )

                logger.info(
                    f"Deleted "
                    f"{len(ids_to_delete)} "
                    f"existing chunks "
                    f"for exact file: "
                    f"{file_path}"
                )


        batch_ids = []
        batch_documents = []
        batch_metadatas = []
        batch_embeddings = []


        def flush_batch():

            if not batch_ids:
                return

            self.collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings
            )

            stats["added"] += len(
                batch_ids
            )

            batch_ids.clear()
            batch_documents.clear()
            batch_metadatas.clear()
            batch_embeddings.clear()


        for text, metadata in documents:

            doc_id = (
                self.generate_document_id(
                    file_path,
                    metadata
                )
            )

            exists = (
                self.check_document_exists(
                    doc_id
                )
            )

            if exists:

                if update_mode == "skip":

                    stats[
                        "skipped"
                    ] += 1

                    continue


                elif update_mode == "update":

                    success = (
                        self.update_document(
                            doc_id,
                            text,
                            metadata
                        )
                    )

                    if success:

                        stats[
                            "updated"
                        ] += 1

                    continue


            try:

                embedding = (
                    self.get_embedding(
                        text
                    )
                )

                batch_ids.append(
                    doc_id
                )

                batch_documents.append(
                    text
                )

                batch_metadatas.append(
                    metadata
                )

                batch_embeddings.append(
                    embedding
                )


                if (
                    len(batch_ids)
                    >= batch_size
                ):

                    flush_batch()


            except Exception as e:

                logger.error(
                    f"Failed processing "
                    f"document {doc_id}: {e}"
                )


        flush_batch()

        return stats


    # -----------------------------------------------------
    # Process All Data
    # -----------------------------------------------------

    def process_all_text_data(
        self,
        base_path: str,
        update_mode: str = "skip",
        batch_size: int = 50
    ) -> Dict[str, Any]:

        stats = {

            "files_processed": 0,

            "documents_added": 0,

            "documents_updated": 0,

            "documents_skipped": 0,

            "errors": 0,

            "total_chunks": 0,

            "missions": {}
        }


        files_to_process = (
            self.scan_text_files_only(
                base_path
            )
        )


        if not files_to_process:

            logger.warning(
                "No text files found under "
                f"{base_path}"
            )

            return stats


        for file_path in files_to_process:

            try:

                logger.info(
                    f"Processing file: "
                    f"{file_path}"
                )


                documents = (
                    self.process_text_file(
                        file_path
                    )
                )


                mission = (
                    self.extract_mission_from_path(
                        file_path
                    )
                )


                if mission not in stats[
                    "missions"
                ]:

                    stats[
                        "missions"
                    ][mission] = {

                        "files": 0,

                        "chunks": 0,

                        "added": 0,

                        "updated": 0,

                        "skipped": 0
                    }


                result = (
                    self.add_documents_to_collection(
                        documents,
                        file_path,
                        batch_size=batch_size,
                        update_mode=update_mode
                    )
                )


                stats[
                    "files_processed"
                ] += 1


                stats[
                    "total_chunks"
                ] += len(
                    documents
                )


                stats[
                    "documents_added"
                ] += result[
                    "added"
                ]


                stats[
                    "documents_updated"
                ] += result[
                    "updated"
                ]


                stats[
                    "documents_skipped"
                ] += result[
                    "skipped"
                ]


                mission_stats = (
                    stats[
                        "missions"
                    ][mission]
                )


                mission_stats[
                    "files"
                ] += 1


                mission_stats[
                    "chunks"
                ] += len(
                    documents
                )


                mission_stats[
                    "added"
                ] += result[
                    "added"
                ]


                mission_stats[
                    "updated"
                ] += result[
                    "updated"
                ]


                mission_stats[
                    "skipped"
                ] += result[
                    "skipped"
                ]


            except Exception as e:

                logger.error(
                    f"Error processing "
                    f"{file_path}: {e}"
                )

                stats[
                    "errors"
                ] += 1


        return stats


    # -----------------------------------------------------
    # Collection Info
    # -----------------------------------------------------

    def get_collection_info(
        self
    ) -> Dict[str, Any]:

        try:

            return {

                "collection_name":
                    self.collection.name,

                "document_count":
                    self.collection.count(),

                "metadata":
                    self.collection.metadata,

                "chroma_directory":
                    self.chroma_persist_directory,

                "embedding_model":
                    self.embedding_model
            }

        except Exception as e:

            logger.error(
                "Error getting "
                f"collection info: {e}"
            )

            return {
                "error": str(e)
            }


    # -----------------------------------------------------
    # Query Collection
    # -----------------------------------------------------

    def query_collection(
        self,
        query_text: str,
        n_results: int = 5
    ) -> Dict[str, Any]:

        try:

            query_embedding = (
                self.get_embedding(
                    query_text
                )
            )

            results = (
                self.collection.query(
                    query_embeddings=[
                        query_embedding
                    ],
                    n_results=n_results,
                    include=[
                        "documents",
                        "metadatas",
                        "distances"
                    ]
                )
            )

            return results

        except Exception as e:

            logger.error(
                "Error querying "
                f"collection: {e}"
            )

            return {}


    # -----------------------------------------------------
    # Collection Statistics
    # -----------------------------------------------------

    def get_collection_stats(
        self
    ) -> Dict[str, Any]:

        try:

            all_docs = (
                self.collection.get()
            )


            if not all_docs.get(
                "metadatas"
            ):

                return {
                    "error":
                    "No documents in collection"
                }


            stats = {

                "total_documents":
                    len(
                        all_docs[
                            "metadatas"
                        ]
                    ),

                "missions": {},

                "data_types": {},

                "document_categories": {},

                "file_types": {}
            }


            for metadata in (
                all_docs[
                    "metadatas"
                ]
            ):

                mission = metadata.get(
                    "mission",
                    "unknown"
                )

                data_type = metadata.get(
                    "data_type",
                    "unknown"
                )

                doc_category = metadata.get(
                    "document_category",
                    "unknown"
                )

                file_type = metadata.get(
                    "file_type",
                    "unknown"
                )


                stats[
                    "missions"
                ][mission] = (

                    stats[
                        "missions"
                    ].get(
                        mission,
                        0
                    )
                    + 1
                )


                stats[
                    "data_types"
                ][data_type] = (

                    stats[
                        "data_types"
                    ].get(
                        data_type,
                        0
                    )
                    + 1
                )


                stats[
                    "document_categories"
                ][doc_category] = (

                    stats[
                        "document_categories"
                    ].get(
                        doc_category,
                        0
                    )
                    + 1
                )


                stats[
                    "file_types"
                ][file_type] = (

                    stats[
                        "file_types"
                    ].get(
                        file_type,
                        0
                    )
                    + 1
                )


            return stats


        except Exception as e:

            logger.error(
                "Error getting "
                f"collection stats: {e}"
            )

            return {
                "error": str(e)
            }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ChromaDB Embedding Pipeline "
            "for NASA Data"
        )
    )


    parser.add_argument(
        "--data-path",
        default=".",
        help="Path to data directories"
    )


    parser.add_argument(
        "--openai-key",
        required=True,
        help="OpenAI API key"
    )


    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db_openai",
        help="ChromaDB persist directory"
    )


    parser.add_argument(
        "--collection-name",
        default="nasa_space_missions_text",
        help="Collection name"
    )


    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model"
    )


    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Text chunk size"
    )


    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap size"
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size"
    )


    parser.add_argument(
        "--update-mode",
        choices=[
            "skip",
            "update",
            "replace"
        ],
        default="skip",
        help=(
            "How to handle existing documents: "
            "skip, update, or replace"
        )
    )


    parser.add_argument(
        "--test-query",
        help="Test query after processing"
    )


    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show collection statistics"
    )


    parser.add_argument(
        "--delete-source",
        help=(
            "Delete all documents from "
            "a source pattern"
        )
    )


    args = parser.parse_args()


    logger.info(
        "Initializing ChromaDB "
        "Embedding Pipeline..."
    )


    pipeline = (
        ChromaEmbeddingPipelineTextOnly(

            openai_api_key=
                args.openai_key,

            chroma_persist_directory=
                args.chroma_dir,

            collection_name=
                args.collection_name,

            embedding_model=
                args.embedding_model,

            chunk_size=
                args.chunk_size,

            chunk_overlap=
                args.chunk_overlap
        )
    )


    if args.delete_source:

        deleted_count = (
            pipeline.delete_documents_by_source(
                args.delete_source
            )
        )

        logger.info(
            f"Deleted "
            f"{deleted_count} documents "
            f"matching source pattern: "
            f"{args.delete_source}"
        )

        return


    if args.stats_only:

        logger.info(
            "Collection Statistics:"
        )

        stats = (
            pipeline.get_collection_stats()
        )

        for key, value in (
            stats.items()
        ):

            logger.info(
                f"{key}: {value}"
            )

        return


    logger.info(
        "Starting text data processing "
        f"with update mode: "
        f"{args.update_mode}"
    )


    start_time = time.time()


    stats = (
        pipeline.process_all_text_data(

            args.data_path,

            update_mode=
                args.update_mode,

            batch_size=
                args.batch_size
        )
    )


    processing_time = (
        time.time()
        - start_time
    )


    logger.info(
        "=" * 60
    )

    logger.info(
        "PROCESSING COMPLETE"
    )

    logger.info(
        "=" * 60
    )


    logger.info(
        "Files processed: "
        f"{stats['files_processed']}"
    )


    logger.info(
        "Total chunks created: "
        f"{stats['total_chunks']}"
    )


    logger.info(
        "Documents added: "
        f"{stats['documents_added']}"
    )


    logger.info(
        "Documents updated: "
        f"{stats['documents_updated']}"
    )


    logger.info(
        "Documents skipped: "
        f"{stats['documents_skipped']}"
    )


    logger.info(
        f"Errors: "
        f"{stats['errors']}"
    )


    logger.info(
        "Processing time: "
        f"{processing_time:.2f} seconds"
    )


    logger.info(
        "\nMission breakdown:"
    )


    for mission, mission_stats in (
        stats[
            "missions"
        ].items()
    ):

        logger.info(
            f"  {mission}: "
            f"{mission_stats['files']} files, "
            f"{mission_stats['chunks']} chunks"
        )

        logger.info(
            f"    Added: "
            f"{mission_stats['added']}, "
            f"Updated: "
            f"{mission_stats['updated']}, "
            f"Skipped: "
            f"{mission_stats['skipped']}"
        )


    collection_info = (
        pipeline.get_collection_info()
    )


    logger.info(
        "\nCollection: "
        f"{collection_info.get('collection_name', 'N/A')}"
    )


    logger.info(
        "Total documents in collection: "
        f"{collection_info.get('document_count', 'N/A')}"
    )


    if args.test_query:

        logger.info(
            f"\nTesting query: "
            f"'{args.test_query}'"
        )


        results = (
            pipeline.query_collection(
                args.test_query
            )
        )


        if (
            results
            and
            results.get(
                "documents"
            )
        ):

            logger.info(
                f"Found "
                f"{len(results['documents'][0])} "
                f"results:"
            )


            for i, doc in enumerate(
                results[
                    "documents"
                ][0][:3]
            ):

                logger.info(
                    f"Result "
                    f"{i + 1}: "
                    f"{doc[:200]}..."
                )


    logger.info(
        "Pipeline completed successfully!"
    )


if __name__ == "__main__":
    main()
