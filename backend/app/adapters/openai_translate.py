from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from ..sources import SourceConfig
from ._translate_prompts import (
    CONTENT_ONLY_TRANSLATION_RULES,
    PREPROCESS_PROMPT,
    TRANSLATE_RULES,
)
from .openai_client import normalize_openai_base_url

log = logging.getLogger(__name__)

API_SETTING_KEYS = ("base_url", "api_key", "model")
PREPROCESS_RETRY = 2
TRANSLATE_RETRY = 2
DESCRIPTION_LIMIT = 500
DEFAULT_CONCURRENCY = 50
CONTEXT_WINDOW = 2


class HotwordItem(BaseModel):
    src: str
    dst: str


class CorrectionItem(BaseModel):
    wrong: str
    correct: str


class PreprocessResponse(BaseModel):
    summary: str = ""
    hotwords: list[HotwordItem] = Field(default_factory=list)
    corrections: list[CorrectionItem] = Field(default_factory=list)


class TranslationItem(BaseModel):
    dst: str
    audio_mode: Literal["tts", "original"]

    @model_validator(mode="after")
    def validate_tts_text(self) -> "TranslationItem":
        if self.audio_mode == "tts" and not self.dst.strip():
            raise ValueError("dst must be non-empty when audio_mode is tts")
        return self


def list_models(*, base_url: str, api_key: str) -> list[str]:
    if not api_key:
        raise ValueError("OpenAI API key is not configured.")
    client = OpenAI(api_key=api_key, base_url=normalize_openai_base_url(base_url))
    response = client.models.list()
    seen: set[str] = set()
    models: list[str] = []
    for item in response.data:
        model_id = getattr(item, "id", "")
        if model_id and model_id not in seen:
            seen.add(model_id)
            models.append(model_id)
    return models


def _client(base_url: str, api_key: str) -> OpenAI:
    if not api_key:
        raise ValueError("OpenAI API key is not configured.")
    return OpenAI(api_key=api_key, base_url=normalize_openai_base_url(base_url))


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
# 翻译请求中说话人前缀「说话人N：」；译文里若被模型误带回则剥掉。
_SPEAKER_PREFIX_RE = re.compile(r"^说话人\d+：")


def _extract_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise json.JSONDecodeError(f"no JSON object found; raw[:300]={raw[:300]!r}", raw, 0)
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"{exc.msg}; len={len(raw)}; raw[:300]={raw[:300]!r}; raw[-200:]={raw[-200:]!r}",
            raw,
            exc.pos,
        ) from None


def _call_json(client: OpenAI, model: str, system: str, user: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content or "{}"
    return _extract_json(raw)


def _format_terms(items: list, fmt: str, empty: str) -> str:
    if not items:
        return empty
    return "\n".join(fmt.format(**item.model_dump()) for item in items)


def _meta_view(meta: dict[str, Any]) -> dict[str, str]:
    description = (meta.get("description") or "").strip()
    if len(description) > DESCRIPTION_LIMIT:
        description = description[:DESCRIPTION_LIMIT] + "..."
    return {
        "title": str(meta.get("title") or "").strip() or "(unknown)",
        "uploader": str(meta.get("uploader") or "").strip() or "(unknown)",
        "description": description or "(none)",
    }


def preprocess(
    full_text: str,
    meta: dict[str, Any],
    source: SourceConfig,
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> PreprocessResponse:
    user = PREPROCESS_PROMPT.format(
        src_language_name=source.asr_language_name,
        dst_language_name=source.target_language_name,
        full_text=full_text,
        **_meta_view(meta),
    )
    client = _client(base_url, api_key)
    last_error: Exception | None = None
    for attempt in range(PREPROCESS_RETRY + 1):
        try:
            data = _call_json(client, model, "You output strict JSON only.", user)
            return PreprocessResponse.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            log.warning("preprocess attempt %d failed: %s", attempt + 1, exc)
    log.error("preprocess gave up, returning empty: %s", last_error)
    return PreprocessResponse()


def _translate_system(source: SourceConfig, meta: dict[str, Any], pre: PreprocessResponse) -> str:
    rules = TRANSLATE_RULES[(source.asr_language, source.target_language)]
    formatted = rules.format(
        summary=pre.summary or "(none)",
        hotwords=_format_terms(pre.hotwords, "{src} -> {dst}", "(none)"),
        corrections=_format_terms(pre.corrections, "{wrong} -> {correct}", "(none)"),
        **_meta_view(meta),
    )
    return f"{formatted}\n\n{CONTENT_ONLY_TRANSLATION_RULES}"


def _post_process(text: str, target_language: str) -> str:
    cleaned = text.strip()
    cleaned = _SPEAKER_PREFIX_RE.sub("", cleaned)
    if target_language == "zh":
        cleaned = cleaned.replace("——", "，")
    return cleaned


def _user_message(
    text: str,
    prev_sentences: list[str],
    next_sentences: list[str],
    speaker: str | None = None,
) -> str:
    """构造带邻句上下文的 user 消息；无上下文时直接返回原文。

    多说话人时给待翻译句加「说话人N：」前缀，上下文由调用方预先打好前缀。
    """
    tag = f"说话人{speaker}：" if speaker else ""
    if not prev_sentences and not next_sentences:
        return f"{tag}{text}"
    lines = ["# 上下文（仅供参考，不要翻译，不要输出）"]
    if prev_sentences:
        lines.append("[前文]\n" + "\n".join(prev_sentences))
    if next_sentences:
        lines.append("[后文]\n" + "\n".join(next_sentences))
    lines.append("# 待翻译（只翻译这一句）\n" + tag + text)
    return "\n\n".join(lines)


def translate_sentence(
    text: str,
    target_language: str,
    client: OpenAI,
    model: str,
    system: str,
    prev_sentences: list[str] | None = None,
    next_sentences: list[str] | None = None,
    speaker: str | None = None,
) -> TranslationItem:
    last_error: Exception | None = None
    user = _user_message(text, prev_sentences or [], next_sentences or [], speaker)
    for attempt in range(TRANSLATE_RETRY):
        try:
            data = _call_json(client, model, system, user)
            item = TranslationItem.model_validate(data)
            return item.model_copy(update={"dst": _post_process(item.dst, target_language)})
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            log.warning("translate attempt %d failed for %r: %s", attempt + 1, text[:60], exc)
    raise RuntimeError(f"translate_sentence failed after {TRANSLATE_RETRY} attempts: {last_error}")


def translate_batch(
    texts: list[str],
    source: SourceConfig,
    meta: dict[str, Any],
    pre: PreprocessResponse,
    *,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    speakers: list[str] | None = None,
) -> list[TranslationItem]:
    if not texts:
        return []
    system = _translate_system(source, meta, pre)
    client = _client(base_url, api_key)
    log.info(
        "translate_batch: %d sentences, concurrency=%d", len(texts), concurrency,
    )
    # 多说话人时给每句（含上下文）带「说话人N：」前缀，帮助 LLM 理解对话人物关系。
    tagged = (
        [f"说话人{s}：{t}" for s, t in zip(speakers, texts)]
        if speakers
        else texts
    )
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [
            pool.submit(
                translate_sentence,
                text,
                source.target_language,
                client,
                model,
                system,
                tagged[max(0, idx - CONTEXT_WINDOW):idx],
                tagged[idx + 1:idx + 1 + CONTEXT_WINDOW],
                speakers[idx] if speakers else None,
            )
            for idx, text in enumerate(texts)
        ]
        return [future.result() for future in futures]


def _read_meta(session: Path) -> dict[str, Any]:
    info_file = session / "metadata" / "ytdlp_info.json"
    if not info_file.exists():
        return {}
    return json.loads(info_file.read_text(encoding="utf-8"))


def _speaker(utt: dict[str, Any]) -> str:
    additions = utt.get("additions") or {}
    if isinstance(additions, dict):
        return str(additions.get("speaker") or "1")
    return "1"


# 无空格书写的语言：转录兜底拼接时不插空格，避免预处理输入出现多余分隔。
_CJK_JOIN_LANGUAGES = {"ja", "zh"}


def _full_text(data: dict[str, Any], texts: list[str], language: str) -> str:
    raw = data.get("result", {}).get("text") or ""
    if raw.strip():
        return raw
    sep = "" if language in _CJK_JOIN_LANGUAGES else " "
    return sep.join(texts)


def preprocess_artifact_path(session: Path) -> Path:
    return session / "metadata" / "translation_preprocess.json"


def write_preprocess_artifact(session: Path, pre: PreprocessResponse) -> Path:
    path = preprocess_artifact_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pre.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_preprocess_artifact(session: Path) -> PreprocessResponse | None:
    path = preprocess_artifact_path(session)
    if not path.exists():
        return None
    return PreprocessResponse.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _concurrency_from(settings: dict[str, str]) -> int:
    raw = str(settings.get("translate_concurrency") or "").strip()
    if not raw or not all("0" <= char <= "9" for char in raw):
        return DEFAULT_CONCURRENCY
    concurrency = int(raw)
    if concurrency < 1 or concurrency > 200:
        return DEFAULT_CONCURRENCY
    return concurrency


def translate_asr(
    asr_file: Path,
    session: Path,
    settings: dict[str, str],
    source: SourceConfig,
) -> Path:
    output_file = session / "metadata" / f"translation.{source.target_language}.json"
    if output_file.exists():
        return output_file

    data = json.loads(asr_file.read_text(encoding="utf-8"))
    utterances = data["result"]["utterances"]
    texts = [u["text"].strip() for u in utterances]
    speakers = [_speaker(u) for u in utterances]
    # 仅多说话人时标注；单一说话人加前缀只是噪声。
    if len(set(speakers)) <= 1:
        speakers = None
    full_text = _full_text(data, texts, source.asr_language)
    meta = _read_meta(session)

    api = {key: settings[key] for key in API_SETTING_KEYS if key in settings}
    pre = load_preprocess_artifact(session)
    if pre is None:
        pre = preprocess(full_text, meta, source, **api)
        write_preprocess_artifact(session, pre)
        log.info("Wrote translation preprocess artifact to %s", preprocess_artifact_path(session))
    else:
        log.info("Reusing translation preprocess artifact from %s", preprocess_artifact_path(session))
    translated_items = translate_batch(
        texts, source, meta, pre, **api,
        concurrency=_concurrency_from(settings), speakers=speakers,
    )

    translation = [
        {
            "src": text,
            "dst": translated.dst,
            "audio_mode": translated.audio_mode,
            "src_lang": source.asr_language,
            "dst_lang": source.target_language,
            "start_time": utt["start_time"],
            "end_time": utt["end_time"],
            "speaker": _speaker(utt),
        }
        for text, translated, utt in zip(texts, translated_items, utterances)
    ]
    output_file.write_text(
        json.dumps({"translation": translation}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file
