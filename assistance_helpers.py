import base64
import json
import mimetypes
import os
from typing import Any, Dict, Optional

import requests


DEFAULT_LOCATION = {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "place": "New Delhi, India",
}


def get_location() -> Dict[str, Any]:
    """
    Try a lightweight IP-based lookup first.
    Falls back to a stable mock location if the API is unavailable.
    """
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "success":
            city = data.get("city") or "Unknown City"
            region = data.get("regionName") or data.get("country") or "Unknown Region"
            return {
                "latitude": data.get("lat"),
                "longitude": data.get("lon"),
                "place": f"{city}, {region}",
            }
    except Exception:
        pass

    return DEFAULT_LOCATION.copy()


def get_travel_assistance(location: Dict[str, Any]) -> str:
    """
    Fetch emergency travel assistance from a free LLM provider when configured.
    Falls back to a deterministic local message so prediction flow never breaks.
    """
    place = location.get("place", "the accident location")
    latitude = location.get("latitude", "unknown")
    longitude = location.get("longitude", "unknown")

    prompt = (
        f"An accident has occurred at {place} "
        f"(latitude: {latitude}, longitude: {longitude}). "
        "Suggest 3 nearby hospitals or trauma centers, immediate emergency steps, "
        "and the safest travel route guidance for responders and family members. "
        "Keep the answer concise, practical, and easy to follow."
    )

    groq_api_key = os.getenv("GROQ_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    try:
        if groq_api_key:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an emergency travel assistant for road accidents.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            if content:
                return content

        if openrouter_api_key:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an emergency travel assistant for road accidents.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            if content:
                return content
    except Exception:
        pass

    return (
        f"Emergency guidance for {place}: call local emergency services immediately, "
        "share the exact coordinates with responders, head to the nearest multi-specialty "
        "hospital or trauma center, and prefer major well-lit roads while avoiding blocked "
        "or high-traffic routes when moving the patient."
    )


def _image_to_data_url(image_path: str) -> Optional[str]:
    if not image_path or not os.path.exists(image_path):
        return None

    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _fallback_scene_analysis(localization_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    position = "image center"
    scene_note = "A road accident appears visible in the uploaded image."
    visible_evidence = "Vehicle damage or collision activity is likely visible."

    if localization_hint:
        position = localization_hint.get("position_label") or position
        bbox = localization_hint.get("bbox")
        if bbox:
            visible_evidence = (
                f"Most accident-related evidence appears around x={bbox.get('x')}, "
                f"y={bbox.get('y')}, width={bbox.get('width')}, height={bbox.get('height')}."
            )

    return {
        "accident_position_in_image": position,
        "likely_scene": "Roadside or traffic scene",
        "visible_evidence": visible_evidence,
        "summary": scene_note,
        "confidence_note": (
            "Exact real-world address cannot be confirmed from the image alone unless "
            "clear landmarks, signboards, or metadata are visible."
        ),
    }


def analyze_accident_scene(
    image_path: str, localization_hint: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze the uploaded image with a vision-capable LLM when configured.
    Falls back to a deterministic local description when no API key/model is available.
    """
    fallback = _fallback_scene_analysis(localization_hint)
    image_data_url = _image_to_data_url(image_path)
    if not image_data_url:
        return fallback

    prompt = (
        "Analyze this uploaded accident-related image and respond in strict JSON only. "
        'Use these keys: "accident_position_in_image", "likely_scene", '
        '"visible_evidence", "summary", "confidence_note". '
        "Keep values short and practical. "
        "Describe where the accident is visible inside the image like center-left, "
        "top-right, foreground, intersection area, roadside, etc. "
        "If the exact real-world location cannot be known from the image alone, clearly say that."
    )

    groq_api_key = os.getenv("GROQ_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    try:
        if groq_api_key:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv(
                        "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
                    ),
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_data_url}},
                            ],
                        }
                    ],
                    "temperature": 0.2,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            parsed = _extract_json_object(content)
            if parsed:
                return {**fallback, **parsed}

        if openrouter_api_key:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv(
                        "OPENROUTER_VISION_MODEL", "openai/gpt-4.1-mini"
                    ),
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_data_url}},
                            ],
                        }
                    ],
                    "temperature": 0.2,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            parsed = _extract_json_object(content)
            if parsed:
                return {**fallback, **parsed}
    except Exception:
        pass

    return fallback
