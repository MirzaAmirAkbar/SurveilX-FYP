from __future__ import annotations

import os
from typing import Dict, Optional

import requests

from .templates import cache_key_for_event, template_message
from .types import TalkdownEvent, TalkdownMessage, ToneLevel


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqTalkdownClient:
    """
    Minimal Groq client wrapper with:
    - small in-memory cache
    - strict timeout
    - deterministic fallback to templates when unavailable
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        timeout_seconds: float = 0.8,
        max_tokens: int = 64,
    ) -> None:
        self._model = model
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens
        self._cache: Dict[str, TalkdownMessage] = {}

    def _build_prompt(self, event: TalkdownEvent, tone: ToneLevel) -> str:
        attrs = event.attributes
        accessories = ", ".join(attrs.accessories) if attrs.accessories else "none"
        return (
            "You are a security announcement system for a surveillance camera. "
            "Generate ONE short sentence (max 22 words) to be spoken aloud over speakers.\n\n"
            "Rules:\n"
            "- Use ONLY the provided clothing and accessories details.\n"
            "- Refer to the person in a neutral way (e.g., 'the individual', 'the person').\n"
            "- Do NOT mention gender, age, race, or any sensitive traits.\n"
            "- Use the requested tone: "
            f"{tone.value.upper()}.\n"
            "- Mention at most the upper clothing and one accessory.\n"
            "- Speak directly to the person.\n\n"
            f"Context:\n"
            f"- Event type: {event.event_type}\n"
            f"- Upper clothing: {attrs.upper_clothing}\n"
            f"- Lower clothing: {attrs.lower_clothing}\n"
            f"- Footwear: {attrs.footwear}\n"
            f"- Accessories: {accessories}\n\n"
            "Return only the sentence, without quotes."
        )

    def generate(self, event: TalkdownEvent, tone: ToneLevel) -> TalkdownMessage:
        """
        Best-effort LLM generation with fast fallback.
        """
        cache_key = cache_key_for_event(event, tone)
        if cache_key in self._cache:
            return self._cache[cache_key]

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            msg = template_message(event, tone)
            self._cache[cache_key] = msg
            return msg

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": 0.5,
            "messages": [
                {
                    "role": "system",
                    "content": "You generate concise security loudspeaker announcements.",
                },
                {"role": "user", "content": self._build_prompt(event, tone)},
            ],
        }

        try:
            resp = requests.post(
                GROQ_API_URL, headers=headers, json=payload, timeout=self._timeout
            )
            resp.raise_for_status()
            data = resp.json()
            content: Optional[str] = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                if isinstance(data, dict)
                else ""
            )
            if not content:
                raise ValueError("empty Groq response")
            text = content.strip().replace("\n", " ")
            msg = TalkdownMessage(text=text, tone=tone, event=event)
        except Exception:
            msg = template_message(event, tone)

        self._cache[cache_key] = msg
        return msg

