from __future__ import annotations

from hashlib import sha256
from typing import Dict

from .types import PersonAttributes, TalkdownEvent, TalkdownMessage, ToneLevel


def _attributes_signature(attrs: PersonAttributes) -> str:
    """
    Stable, compact signature for caching and deduplication.
    """
    raw = "|".join(
        [
            attrs.upper_clothing or "",
            attrs.lower_clothing or "",
            ",".join(sorted(a for a in attrs.accessories if a)),
            attrs.footwear or "",
            attrs.gender or "",
        ]
    )
    return sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_accessory_clause(attrs: PersonAttributes) -> str:
    if not attrs.accessories:
        return ""
    # pick the first accessory only to keep sentences short
    first = attrs.accessories[0]
    return f" with the {first}"


def build_subject_phrase(attrs: PersonAttributes) -> str:
    """
    Build a short, identifying subject phrase using clothing/accessories only.
    """
    upper = attrs.upper_clothing or "clothing"
    accessory_clause = build_accessory_clause(attrs)
    return f"the individual in the {upper}{accessory_clause}"


def template_message(event: TalkdownEvent, tone: ToneLevel) -> TalkdownMessage:
    subject = build_subject_phrase(event.attributes)

    if event.event_type == "restricted_area_breach":
        if tone == ToneLevel.POLITE:
            text = (
                f"{subject}, you have entered a restricted area. "
                "Please leave the area immediately."
            )
        elif tone == ToneLevel.FIRM:
            text = (
                f"{subject}, this is a security notice. "
                "Leave the restricted area at once."
            )
        else:
            text = (
                f"Final warning to {subject}: leave the restricted area now. "
                "Security personnel are being dispatched."
            )
    elif event.event_type == "shoplifting_suspected":
        if tone == ToneLevel.POLITE:
            text = (
                f"{subject}, please return any unpaid items to the nearest counter "
                "or contact store staff immediately."
            )
        elif tone == ToneLevel.FIRM:
            text = (
                f"{subject}, you are under observation. "
                "Return any unpaid items to the counter now."
            )
        else:
            text = (
                f"Security notice to {subject}: "
                "return unpaid items immediately or security will intervene."
            )
    else:
        # generic safety notice
        if tone == ToneLevel.POLITE:
            text = f"{subject}, please follow the on-site safety instructions."
        elif tone == ToneLevel.FIRM:
            text = (
                f"{subject}, this is a security instruction. "
                "Comply with the safety rules immediately."
            )
        else:
            text = (
                f"Security notice to {subject}: "
                "comply with the safety rules now or further action will be taken."
            )

    return TalkdownMessage(text=text, tone=tone, event=event)


def cache_key_for_event(event: TalkdownEvent, tone: ToneLevel) -> str:
    """
    Public helper: stable cache key using event type, person, tone, and attributes.
    """
    attrs_sig = _attributes_signature(event.attributes)
    return f"{event.event_type}:{event.person_id}:{tone.value}:{attrs_sig}"


def explain_cache_contents(cache: Dict[str, TalkdownMessage]) -> str:
    """
    Small helper for debugging / tests.
    """
    parts = []
    for key, msg in cache.items():
        parts.append(f"{key} -> {msg.text[:60]}...")
    return "\n".join(parts)

