from typing import Dict, List
from openai import OpenAI


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> str:
    """Generate response using OpenAI with retrieved NASA context."""

    system_prompt = """
You are a NASA mission expert assistant.

Answer the user's question using the retrieved NASA mission context provided to you.

Important instructions:
- Base your answer on the retrieved context.
- Cite the relevant retrieved source labels in your answer.
- Use citation labels such as [Source 1], [Source 2], [Source 3].
- Do not invent source citations.
- If the retrieved context does not contain enough information, clearly say so.
- Keep the answer factual, clear, and concise.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "system",
            "content": (
                "Retrieved NASA Mission Context:\n\n"
                f"{context}"
            )
        }
    ]

    for message in conversation_history:
        if (
            isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and "content" in message
        ):
            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"]
                }
            )

    messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    client = OpenAI(
        api_key=openai_key
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message.content
