import json
import base64
from langchain_ollama import ChatOllama
from app.config import settings


def get_llm(model: str = None, temperature: float = 0.1):
    return ChatOllama(
        model=model or settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
    )


def get_light_llm(temperature: float = 0.1):
    return get_llm(model=settings.LIGHT_LLM_MODEL, temperature=temperature)


def invoke_llm(prompt: str, model: str = None) -> str:
    llm = get_llm(model=model)
    response = llm.invoke(prompt)
    return response.content


def invoke_llm_with_image(prompt: str, image_path: str) -> str:
    """Send an image + prompt to the LLM for vision-based extraction."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    llm = get_llm()
    from langchain_core.messages import HumanMessage

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
        ]
    )
    response = llm.invoke([message])
    return response.content


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response that may contain markdown fences."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {}
