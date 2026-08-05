"""Groq API client wrapper for LexAI."""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

AVAILABLE_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B (Recommended)",
    "llama-3.1-8b-instant": "Llama 3.1 8B (Fastest)",
    "mixtral-8x7b-32768": "Mixtral 8x7B (Long Context)",
    "gemma2-9b-it": "Gemma 2 9B",
}


def get_groq_client(
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> ChatGroq:
    """
    Initialize and return a ChatGroq client.

    Args:
        api_key: Groq API key.
        model: Model name to use.
        temperature: Sampling temperature (0=deterministic, 1=creative).
        max_tokens: Maximum response tokens.

    Returns:
        Configured ChatGroq instance.
    """
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def simple_chat(
    client: ChatGroq,
    system_prompt: str,
    user_message: str,
) -> str:
    """
    Perform a simple chat completion.

    Args:
        client: ChatGroq client instance.
        system_prompt: System instruction for the LLM.
        user_message: User's input message.

    Returns:
        LLM response as string.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    response = client.invoke(messages)
    return response.content


def validate_api_key(api_key: str) -> bool:
    """
    Validate a Groq API key by making a minimal test call.

    Args:
        api_key: Groq API key to validate.

    Returns:
        True if valid, False otherwise.
    """
    try:
        client = get_groq_client(api_key, model="llama-3.1-8b-instant", max_tokens=5)
        client.invoke([HumanMessage(content="Hi")])
        return True
    except Exception:
        return False
