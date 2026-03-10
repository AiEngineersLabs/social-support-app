"""
LLM Service — Ollama integration with Langfuse observability.

Provides:
  - get_llm / get_light_llm / get_vision_llm  (model factories)
  - invoke_llm                                 (text generation)
  - invoke_light_llm                           (faster model)
  - invoke_llm_with_image                      (vision OCR via llava)
  - get_langfuse_callback                      (optional tracing)
  - extract_json_from_response                 (JSON parsing helper)
"""
import json
import base64
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Langfuse callback (optional — degrades gracefully if keys are not set)
# ---------------------------------------------------------------------------

def get_langfuse_callback():
    """Return a Langfuse LangChain callback handler, or None if not configured."""
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        return None
    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    except Exception as exc:
        logger.warning("Langfuse callback unavailable: %s", exc)
        return None


def _build_config(name: str = None) -> dict:
    """Build LangChain invoke config with optional Langfuse tracing."""
    callbacks = []
    lf = get_langfuse_callback()
    if lf:
        callbacks.append(lf)
    cfg = {}
    if callbacks:
        cfg["callbacks"] = callbacks
    if name:
        cfg["run_name"] = name
    return cfg


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def get_llm(model: str = None, temperature: float = 0.1) -> ChatOllama:
    return ChatOllama(
        model=model or settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
    )


def get_light_llm(temperature: float = 0.1) -> ChatOllama:
    return get_llm(model=settings.LIGHT_LLM_MODEL, temperature=temperature)


def get_vision_llm(temperature: float = 0.1) -> ChatOllama:
    """Return the llava vision model for image OCR."""
    vision_model = getattr(settings, "VISION_MODEL", "llava:7b")
    return ChatOllama(
        model=vision_model,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Invocation helpers
# ---------------------------------------------------------------------------

def invoke_llm(prompt: str, model: str = None, name: str = None) -> str:
    """Invoke text LLM with optional Langfuse tracing."""
    llm = get_llm(model=model)
    response = llm.invoke(prompt, config=_build_config(name))
    return response.content


def invoke_light_llm(prompt: str, name: str = None) -> str:
    """Invoke the lighter/faster model with optional Langfuse tracing."""
    llm = get_light_llm()
    response = llm.invoke(prompt, config=_build_config(name))
    return response.content


def invoke_llm_with_image(prompt: str, image_path: str) -> str:
    """
    Vision OCR: send image + prompt to llava for extraction.

    Strategy:
      1. llava:7b via Ollama (primary — locally hosted)
      2. pytesseract + text LLM (fallback if Tesseract binary present)
      3. Empty-field placeholder (final fallback)
    """
    # Primary: llava via Ollama
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = image_path.lower().rsplit(".", 1)[-1]
        mime = {"png": "image/png", "jpg": "image/jpeg",
                "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")

        vision_llm = get_vision_llm()
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{image_data}"}},
            ]
        )
        response = vision_llm.invoke([message], config=_build_config("vision_ocr"))
        return response.content

    except Exception as llava_err:
        logger.warning("llava OCR failed (%s), trying pytesseract", llava_err)

    # Secondary: pytesseract
    try:
        import pytesseract
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        text = pytesseract.image_to_string(img)
        if text.strip():
            fallback_prompt = (
                f"{prompt}\n\nOCR text:\n{text[:2000]}\n\n"
                "Extract the requested fields from the OCR text and return ONLY JSON."
            )
            return invoke_llm(fallback_prompt, name="ocr_text_extraction")
    except Exception as tess_err:
        logger.warning("pytesseract fallback failed (%s)", tess_err)

    # Final fallback
    logger.error("All vision methods failed for %s", image_path)
    return '{"error": "Vision extraction unavailable", "id_number": "", "full_name": ""}'


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response that may contain markdown fences."""
    if not text:
        return {}
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
