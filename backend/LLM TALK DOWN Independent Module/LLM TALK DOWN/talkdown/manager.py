from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .groq_client import GroqTalkdownClient
from .templates import template_message
from .tts_windows import WindowsTTSWorker
from .types import TalkdownEvent, ToneLevel


@dataclass
class TalkdownState:
    event: TalkdownEvent
    started_at: float
    last_tone: Optional[ToneLevel] = None
    last_spoken_at: float = 0.0


class TalkdownManager:
    """
    Per-person talkdown state machine.

    Usage (real time):
        manager = TalkdownManager()
        # called from your event loop when a violation is detected
        manager.handle_event(event)
        # periodically (e.g. every frame or every 0.5s)
        manager.tick()
    """

    def __init__(
        self,
        polite_after: float = 0.0,
        firm_after: float = 5.0,
        strict_after: float = 10.0,
        cooldown_seconds: float = 4.0,
        expiry_seconds: float = 15.0,
    ) -> None:
        self._polite_after = polite_after
        self._firm_after = firm_after
        self._strict_after = strict_after
        self._cooldown = cooldown_seconds
        self._expiry = expiry_seconds

        self._states: Dict[str, TalkdownState] = {}
        self._groq = GroqTalkdownClient()
        self._tts = WindowsTTSWorker()

    def handle_event(self, event: TalkdownEvent) -> None:
        """
        Register or refresh an active violation for a person.
        """
        key = self._key(event)
        now = time.time()
        state = self._states.get(key)
        if state is None:
            self._states[key] = TalkdownState(event=event, started_at=now)
        else:
            state.event = event
            state.event.last_seen_ts = event.last_seen_ts

    def tick(self) -> None:
        """
        Progress all active states and trigger speech when needed.
        Call this regularly from your main loop.
        """
        now = time.time()
        to_delete = []

        for key, state in self._states.items():
            # expiry: no recent observations
            if now - state.event.last_seen_ts > self._expiry:
                to_delete.append(key)
                continue

            elapsed = now - state.started_at
            desired_tone: Optional[ToneLevel] = None

            if elapsed >= self._strict_after:
                desired_tone = ToneLevel.STRICT
            elif elapsed >= self._firm_after:
                desired_tone = ToneLevel.FIRM
            elif elapsed >= self._polite_after:
                desired_tone = ToneLevel.POLITE

            if desired_tone is None:
                continue

            # do not repeat same tone or speak too frequently
            if state.last_tone == desired_tone and now - state.last_spoken_at < self._cooldown:
                continue

            msg = self._generate_message(state.event, desired_tone)
            self._tts.speak(msg)
            state.last_tone = desired_tone
            state.last_spoken_at = now

        for key in to_delete:
            del self._states[key]

    def _generate_message(self, event: TalkdownEvent, tone: ToneLevel):
        """
        Try Groq first, fall back to templates.
        """
        try:
            return self._groq.generate(event, tone)
        except Exception:
            return template_message(event, tone)

    @staticmethod
    def _key(event: TalkdownEvent) -> str:
        return f"{event.event_type}:{event.person_id}"

