from __future__ import annotations

import queue
import threading
from typing import Optional

try:
    import win32com.client  # type: ignore
    import pythoncom
except ImportError:  # pragma: no cover - only on non-Windows or missing lib
    win32com = None  # type: ignore
    pythoncom = None

from .types import TalkdownMessage


class WindowsTTSWorker:
    """
    Simple background TTS worker using Windows SAPI.
    Non-blocking: messages are enqueued and spoken sequentially.
    """

    def __init__(self, voice_name: Optional[str] = None, rate: int = 0) -> None:
        self._queue: "queue.Queue[TalkdownMessage]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._running = False
        self._voice_name = voice_name
        self._rate = rate

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        # enqueue sentinel to unblock queue.get
        self._queue.put(None)  # type: ignore[arg-type]
        self._thread.join(timeout=1.0)

    def speak(self, message: TalkdownMessage) -> None:
        """
        Enqueue a message to be spoken asynchronously.
        """
        if not self._running:
            self.start()
        self._queue.put(message)

    def _run(self) -> None:
        if win32com is None:
            # Library not available; gracefully drop speech.
            while self._running:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    break
            return

        if pythoncom is not None:
            pythoncom.CoInitialize()

        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Rate = self._rate

        if self._voice_name:
            for voice in speaker.GetVoices():
                if self._voice_name.lower() in voice.GetDescription().lower():
                    speaker.Voice = voice
                    break

        while self._running:
            try:
                msg = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg is None:
                break
            print(f"[ALARM] Speaking: {msg.text}")
            speaker.Speak(msg.text)

