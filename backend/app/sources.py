from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .config import COOKIE_DIR
from .youtube import (
    is_bilibili_url,
    is_local_upload_url,
    is_youtube_url,
    local_upload_direction,
    url_direction,
)


LANG_NAMES = {"en": "English", "ja": "Japanese", "zh": "Simplified Chinese"}

DIRECTION_LANGUAGES: dict[str, tuple[str, str]] = {
    "en-zh": ("en", "zh"),
    "ja-zh": ("ja", "zh"),
    "zh-en": ("zh", "en"),
}


@dataclass(frozen=True)
class SourceConfig:
    name: str
    matches: Callable[[str], bool]
    use_proxy: bool
    cookie_filename: str | None
    asr_language: str
    target_language: str
    default_direction: str = ""

    @property
    def cookie_path(self) -> Path | None:
        if not self.cookie_filename:
            return None
        return COOKIE_DIR / self.cookie_filename

    @property
    def asr_language_name(self) -> str:
        return LANG_NAMES[self.asr_language]

    @property
    def target_language_name(self) -> str:
        return LANG_NAMES[self.target_language]


SOURCES: list[SourceConfig] = [
    SourceConfig(
        name="youtube",
        matches=is_youtube_url,
        use_proxy=True,
        cookie_filename="youtube.txt",
        asr_language="en",
        target_language="zh",
        default_direction="en-zh",
    ),
    SourceConfig(
        name="local",
        matches=is_local_upload_url,
        use_proxy=False,
        cookie_filename=None,
        asr_language="en",
        target_language="zh",
        default_direction="en-zh",
    ),
    SourceConfig(
        name="bilibili",
        matches=is_bilibili_url,
        use_proxy=False,
        cookie_filename="bilibili.txt",
        asr_language="zh",
        target_language="en",
        default_direction="zh-en",
    ),
]


def _url_direction_for(source: SourceConfig, url: str) -> str:
    if source.name == "local":
        return local_upload_direction(url)
    return url_direction(url)


def detect_source(url: str) -> SourceConfig:
    for source in SOURCES:
        if not source.matches(url):
            continue
        direction = _url_direction_for(source, url)
        languages = DIRECTION_LANGUAGES.get(direction)
        if languages and direction != source.default_direction:
            return replace(source, asr_language=languages[0], target_language=languages[1])
        return source
    raise ValueError(f"No source matches URL: {url}")
