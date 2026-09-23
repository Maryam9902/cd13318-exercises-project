from typing import Dict, List
from openai import OpenAI


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    system_prompt = """
You are a NASA Mission Intelligence Assistant.

Answer the user's questions using the provided NASA mission context.
Be accurate, concise, and factual.

If the answer cannot be found in the provided context,
say that the available mission data does not contain enough information.
"""

    # Set context in messages
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "system",
            "content": f"Retrieved NASA Mission Context:\n{context}"
        }
    ]

    # Add chat history
    for message in conversation_history:
        if (
            isinstance(message, dict)
            and "role" in message
            and "content" in message
        ):
            messages.append({
                "role": message["role"],
                "content": message["content"]
            })

    # Add current user message
    messages.append({
        "role": "user",
        "content": user_message
    })

    # Create OpenAI client
    client = OpenAI(api_key=openai_key)

    # Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )

    # Return response
    return response.choices[0].message.content
