from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional, Tuple


class ToneLevel(str, Enum):
    POLITE = "polite"
    FIRM = "firm"
    STRICT = "strict"


EventType = Literal["restricted_area_breach", "shoplifting_suspected", "other"]


@dataclass
class PersonAttributes:
    upper_clothing: str
    lower_clothing: str
    accessories: List[str] = field(default_factory=list)
    footwear: str = ""
    gender: Optional[str] = None


@dataclass
class TalkdownEvent:
    person_id: str
    event_type: EventType
    attributes: PersonAttributes
    first_seen_ts: float
    last_seen_ts: float
    camera_id: Optional[str] = None
    zone_id: Optional[str] = None


@dataclass
class TalkdownMessage:
    text: str
    tone: ToneLevel
    event: TalkdownEvent


CacheKey = Tuple[str, str, ToneLevel, str]

