#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation Integration

Enhanced version of the simple RAG chat that includes
real-time evaluation and feedback collection.
"""

import streamlit as st
import os
from typing import Dict, List, Optional

import ragas_evaluator
import rag_client
import llm_client


# ---------------------------------------------------------
# RAGAS availability
# ---------------------------------------------------------

try:
    from ragas import SingleTurnSample
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)


# ---------------------------------------------------------
# ChromaDB backend discovery
# ---------------------------------------------------------

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """
    Discover available ChromaDB backends
    in the project directory.
    """

    return rag_client.discover_chroma_backends()


# ---------------------------------------------------------
# Initialize RAG system
# ---------------------------------------------------------

def initialize_rag_system(
    chroma_dir: str,
    collection_name: str
):
    """
    Initialize the RAG system using the
    selected ChromaDB backend.
    """

    try:

        collection = (
            rag_client.initialize_rag_system(
                chroma_dir,
                collection_name
            )
        )

        return collection, True, None

    except Exception as e:

        return None, False, str(e)


# ---------------------------------------------------------
# Retrieve documents
# ---------------------------------------------------------

def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:
    """
    Retrieve relevant documents from ChromaDB
    with optional filtering.
    """

    try:

        return rag_client.retrieve_documents(
            collection,
            query,
            n_results,
            mission_filter
        )

    except Exception as e:

        st.error(
            f"Error retrieving documents: {e}"
        )

        return None


# ---------------------------------------------------------
# Format retrieved context
# ---------------------------------------------------------

def format_context(
    documents: List[str],
    metadatas: List[Dict]
) -> str:
    """
    Format retrieved documents into context.
    """

    return rag_client.format_context(
        documents,
        metadatas
    )


# ---------------------------------------------------------
# Generate LLM response
# ---------------------------------------------------------

def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> str:
    """
    Generate response using OpenAI
    with retrieved context.
    """

    try:

        return llm_client.generate_response(
            openai_key,
            user_message,
            context,
            conversation_history,
            model
        )

    except Exception as e:

        return (
            f"Error generating response: {e}"
        )


# ---------------------------------------------------------
# RAGAS Evaluation
# ---------------------------------------------------------

def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str]
) -> Dict[str, float]:
    """
    Evaluate response quality using
    RAGAS metrics.
    """

    try:

        return (
            ragas_evaluator
            .evaluate_response_quality(
                question,
                answer,
                contexts
            )
        )

    except Exception as e:

        return {
            "error":
            f"Evaluation failed: {str(e)}"
        }


# ---------------------------------------------------------
# Display evaluation metrics
# ---------------------------------------------------------

def display_evaluation_metrics(
    scores: Dict[str, float]
):
    """
    Display RAGAS evaluation metrics
    in the sidebar.
    """

    if "error" in scores:

        st.sidebar.error(
            f"Evaluation Error: "
            f"{scores['error']}"
        )

        return

    st.sidebar.subheader(
        "📊 Response Quality"
    )

    for metric_name, score in (
        scores.items()
    ):

        if isinstance(
            score,
            (int, float)
        ):

            st.sidebar.metric(
                label=(
                    metric_name
                    .replace("_", " ")
                    .title()
                ),
                value=f"{score:.3f}"
            )

            progress_value = max(
                0.0,
                min(
                    1.0,
                    float(score)
                )
            )

            st.sidebar.progress(
                progress_value
            )


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

def main():

    st.title(
        "🚀 NASA Space Mission Chat with Evaluation"
    )

    st.markdown(
        "Chat with AI about NASA space missions "
        "with real-time quality evaluation."
    )


    # -----------------------------------------------------
    # Session state
    # -----------------------------------------------------

    if "messages" not in st.session_state:

        st.session_state.messages = []


    if "current_backend" not in st.session_state:

        st.session_state.current_backend = None


    if "last_evaluation" not in st.session_state:

        st.session_state.last_evaluation = None


    if "last_contexts" not in st.session_state:

        st.session_state.last_contexts = []


    # -----------------------------------------------------
    # Sidebar
    # -----------------------------------------------------

    with st.sidebar:

        st.header(
            "🔧 Configuration"
        )


        # -------------------------------------------------
        # Discover ChromaDB backends
        # -------------------------------------------------

        with st.spinner(
            "Discovering ChromaDB backends..."
        ):

            available_backends = (
                discover_chroma_backends()
            )


        if not available_backends:

            st.error(
                "No ChromaDB backends found!"
            )

            st.info(
                "Please run the embedding pipeline "
                "first using `embedding_pipeline.py`."
            )

            st.stop()


        # -------------------------------------------------
        # Backend selection
        # -------------------------------------------------

        st.subheader(
            "📊 ChromaDB Backend"
        )


        backend_options = {
            key: value["display_name"]
            for key, value
            in available_backends.items()
        }


        selected_backend_key = (
            st.selectbox(
                "Select Document Collection",
                options=list(
                    backend_options.keys()
                ),
                format_func=lambda x:
                    backend_options[x],
                help=(
                    "Choose which document "
                    "collection to use."
                )
            )
        )


        selected_backend = (
            available_backends[
                selected_backend_key
            ]
        )


        # -------------------------------------------------
        # OpenAI API key
        # -------------------------------------------------

        st.subheader(
            "🔑 OpenAI Settings"
        )


        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv(
                "OPENAI_API_KEY",
                ""
            ),
            help=(
                "Enter your OpenAI API key."
            )
        )


        if not openai_key:

            st.warning(
                "Please enter your "
                "OpenAI API key."
            )

            st.stop()


        # Set API key for OpenAI and Chroma
        os.environ[
            "OPENAI_API_KEY"
        ] = openai_key

        os.environ[
            "CHROMA_OPENAI_API_KEY"
        ] = openai_key


        # -------------------------------------------------
        # Model selection
        # -------------------------------------------------

        model_choice = st.selectbox(
            "OpenAI Model",
            options=[
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo-preview"
            ],
            help=(
                "Choose the OpenAI model "
                "for responses."
            )
        )


        # -------------------------------------------------
        # Retrieval settings
        # -------------------------------------------------

        st.subheader(
            "🔍 Retrieval Settings"
        )


        n_docs = st.slider(
            "Documents to retrieve",
            min_value=1,
            max_value=10,
            value=3
        )


        # -------------------------------------------------
        # Evaluation settings
        # -------------------------------------------------

        st.subheader(
            "📊 Evaluation Settings"
        )


        if not RAGAS_AVAILABLE:

            st.warning(
                "RAGAS is not available."
            )


        enable_evaluation = (
            st.checkbox(
                "Enable RAGAS Evaluation",
                value=RAGAS_AVAILABLE,
                disabled=not RAGAS_AVAILABLE
            )
        )


        # -------------------------------------------------
        # Backend change
        # -------------------------------------------------

        if (
            st.session_state.current_backend
            != selected_backend_key
        ):

            st.session_state.current_backend = (
                selected_backend_key
            )


    # -----------------------------------------------------
    # Initialize selected ChromaDB collection
    # -----------------------------------------------------

    with st.spinner(
        "Initializing RAG system..."
    ):

        collection, success, error = (
            initialize_rag_system(
                selected_backend[
                    "dir_path"
                ],
                selected_backend[
                    "collection"
                ]
            )
        )


    if not success:

        st.error(
            f"Failed to initialize "
            f"RAG system: {error}"
        )

        st.stop()


    # -----------------------------------------------------
    # Display collection information
    # -----------------------------------------------------

    with st.sidebar:

        st.success(
            "RAG system initialized"
        )

        doc_count = (
            selected_backend.get(
                "doc_count",
                0
            )
        )

        st.metric(
            "Documents",
            doc_count
        )


    # -----------------------------------------------------
    # Display previous evaluation
    # -----------------------------------------------------

    if (
        st.session_state.last_evaluation
        and
        enable_evaluation
    ):

        display_evaluation_metrics(
            st.session_state.last_evaluation
        )


    # -----------------------------------------------------
    # Display previous chat messages
    # -----------------------------------------------------

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # -----------------------------------------------------
    # User input
    # -----------------------------------------------------

    prompt = st.chat_input(
        "Ask about NASA space missions..."
    )


    if prompt:

        # Add user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                prompt
            )


        # -------------------------------------------------
        # Assistant response
        # -------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching documents and "
                "generating response..."
            ):

                # -----------------------------------------
                # Retrieve relevant documents
                # -----------------------------------------

                docs_result = (
                    retrieve_documents(
                        collection,
                        prompt,
                        n_docs
                    )
                )


                context = ""

                contexts_list = []


                # -----------------------------------------
                # Prepare retrieved context
                # -----------------------------------------

                if (
                    docs_result
                    and
                    docs_result.get(
                        "documents"
                    )
                ):

                    documents = (
                        docs_result[
                            "documents"
                        ][0]
                    )


                    metadatas = (
                        docs_result.get(
                            "metadatas",
                            [[]]
                        )[0]
                    )


                    context = format_context(
                        documents,
                        metadatas
                    )


                    contexts_list = (
                        documents
                    )


                    st.session_state.last_contexts = (
                        contexts_list
                    )


                # -----------------------------------------
                # Generate OpenAI response
                # -----------------------------------------

                response = generate_response(
                    openai_key,
                    prompt,
                    context,
                    st.session_state.messages[
                        :-1
                    ],
                    model_choice
                )


                st.markdown(
                    response
                )


                # -----------------------------------------
                # RAGAS evaluation
                # -----------------------------------------

                if (
                    enable_evaluation
                    and
                    RAGAS_AVAILABLE
                    and
                    contexts_list
                ):

                    with st.spinner(
                        "Evaluating response "
                        "quality..."
                    ):

                        evaluation_scores = (
                            evaluate_response_quality(
                                prompt,
                                response,
                                contexts_list
                            )
                        )


                        st.session_state[
                            "last_evaluation"
                        ] = evaluation_scores


        # -------------------------------------------------
        # Save assistant response
        # -------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )


        # Refresh application
        st.rerun()


# ---------------------------------------------------------
# Run application
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
