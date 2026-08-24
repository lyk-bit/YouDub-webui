from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from pydub import AudioSegment

from ..config import MODEL_CACHE_DIR
from ..devices import resolve_device

_MODEL = None
_KOTOBA_PIPE = None

KOTOBA_WHISPER_DEFAULT_MODEL = "kotoba-tech/kotoba-whisper-v2.0"
_KOTOBA_DISABLED_VALUES = {"0", "false", "no", "off", "none"}
KOTOBA_WHISPER_LANGUAGE = "ja"


def _whisper_cache_file(whisper, name: str, download_root: str | None) -> Path | None:
    if not download_root:
        return None
    model_url = getattr(whisper, "_MODELS", {}).get(name)
    if not model_url:
        return None
    filename = Path(urlparse(model_url).path).name
    if not filename:
        return None
    return Path(download_root).expanduser() / filename


def _is_checksum_error(exc: RuntimeError) -> bool:
    return "sha256 checksum" in str(exc).lower()


def _remove_corrupt_whisper_cache(whisper, name: str, download_root: str | None) -> bool:
    cache_file = _whisper_cache_file(whisper, name, download_root)
    if not cache_file or not cache_file.exists():
        return False
    cache_file.unlink()
    return True


def release_model() -> None:
    global _MODEL, _KOTOBA_PIPE
    _MODEL = None
    _KOTOBA_PIPE = None


def _kotoba_model_name() -> str:
    value = os.getenv("KOTOBA_WHISPER_MODEL", KOTOBA_WHISPER_DEFAULT_MODEL).strip()
    if value.lower() in _KOTOBA_DISABLED_VALUES:
        return ""
    return value


def _use_kotoba(language: str) -> bool:
    return language == KOTOBA_WHISPER_LANGUAGE and bool(_kotoba_model_name())


def _load_kotoba():
    global _KOTOBA_PIPE
    if _KOTOBA_PIPE is not None:
        return _KOTOBA_PIPE

    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as exc:  # pragma: no cover - depends on heavy deps
        raise RuntimeError(
            "KOTOBA_WHISPER_MODEL requires transformers; install requirements.txt first."
        ) from exc

    import torch

    name = _kotoba_model_name()
    device = resolve_device("whisper").selected
    cache_dir = str(MODEL_CACHE_DIR / "huggingface")
    dtype = torch.float16 if device == "cuda" else None

    def from_pretrained(loader, *, local_only: bool):
        kwargs = {"cache_dir": cache_dir, "local_files_only": local_only}
        if dtype is not None:
            try:
                return loader(name, dtype=dtype, **kwargs)
            except TypeError:
                # transformers v4 spells this argument torch_dtype.
                return loader(name, torch_dtype=dtype, **kwargs)
        return loader(name, **kwargs)

    # from_pretrained contacts the hub for per-file etag checks even when the
    # cache is complete; when huggingface.co is unreachable each HEAD request
    # hangs until timeout (minutes in total, WinError 10060). Try the local
    # cache first and only go online when files are actually missing.
    try:
        model = from_pretrained(AutoModelForSpeechSeq2Seq.from_pretrained, local_only=True)
        processor = from_pretrained(AutoProcessor.from_pretrained, local_only=True)
    except Exception:
        model = from_pretrained(AutoModelForSpeechSeq2Seq.from_pretrained, local_only=False)
        processor = from_pretrained(AutoProcessor.from_pretrained, local_only=False)
    device_arg = 0 if device == "cuda" else device
    _KOTOBA_PIPE = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=15,
        batch_size=int(os.getenv("KOTOBA_WHISPER_BATCH_SIZE", "16")),
        device=device_arg,
    )
    return _KOTOBA_PIPE


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import whisper

    name = os.getenv("WHISPER_MODEL", "large-v3-turbo")
    whisper_device = resolve_device("whisper").selected
    download_root = os.getenv("WHISPER_DOWNLOAD_ROOT") or None
    try:
        _MODEL = whisper.load_model(name, device=whisper_device, download_root=download_root)
    except RuntimeError as exc:
        if not _is_checksum_error(exc):
            raise
        if not _remove_corrupt_whisper_cache(whisper, name, download_root):
            raise
        _MODEL = whisper.load_model(name, device=whisper_device, download_root=download_root)

    return _MODEL


def _to_ms(seconds: float) -> int:
    return int(round(float(seconds) * 1000))


def _convert_words(words: list) -> list:
    return [
        {
            "text": w.get("word", ""),
            "start_time": _to_ms(w.get("start", 0.0)),
            "end_time": _to_ms(w.get("end", 0.0)),
        }
        for w in words or []
    ]


def _convert_segments(segments: list) -> list:
    return [
        {
            "text": seg.get("text", "").strip(),
            "start_time": _to_ms(seg.get("start", 0.0)),
            "end_time": _to_ms(seg.get("end", 0.0)),
            "words": _convert_words(seg.get("words", [])),
        }
        for seg in segments
    ]


def _convert_chunks(chunks: list, duration_ms: int) -> list:
    utterances = []
    for index, chunk in enumerate(chunks or []):
        text = str(chunk.get("text", "")).strip()
        timestamp = chunk.get("timestamp") or (None, None)
        start, end = timestamp[0], timestamp[1]
        if not text or start is None:
            continue
        if end is None:
            # An open-ended timestamp means "until the end of the audio", but a
            # mid-stream chunk must never extend to the full duration: that
            # would swallow the later gaps that merge_audio backfills with
            # original non-speech vocals. Clamp it to the next chunk's start.
            next_timestamp = (
                chunks[index + 1].get("timestamp") if index + 1 < len(chunks) else None
            )
            next_start = next_timestamp[0] if next_timestamp else None
            end_ms = (
                min(_to_ms(next_start), duration_ms)
                if next_start is not None
                else duration_ms
            )
        else:
            end_ms = _to_ms(end)
        if end_ms <= _to_ms(start):
            continue
        utterances.append(
            {
                "text": text,
                "start_time": _to_ms(start),
                "end_time": end_ms,
                "words": [],
            }
        )
    return utterances


def _load_mono_audio(file: Path):
    import soundfile as sf

    data, rate = sf.read(str(file), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) == 2:
        data = data.mean(axis=1, dtype="float32")
    return data, rate


def recognize_speech(vocals_file: Path, session: Path, language: str) -> Path:
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_file = metadata_dir / "asr.json"
    if output_file.exists():
        return output_file

    duration_ms = len(AudioSegment.from_file(vocals_file))
    if _use_kotoba(language):
        audio, sample_rate = _load_mono_audio(vocals_file)
        result = _load_kotoba()(
            {"array": audio, "sampling_rate": sample_rate},
            return_timestamps=True,
        )
        utterances = _convert_chunks(result.get("chunks", []), duration_ms)
        text = (result.get("text") or "").strip()
    else:
        model = _load_model()
        result = model.transcribe(
            str(vocals_file),
            language=language,
            word_timestamps=True,
            verbose=False,
        )
        utterances = _convert_segments(result.get("segments", []))
        text = (result.get("text") or "").strip()

    if not utterances:
        raise RuntimeError("Whisper did not return any segments.")

    payload = {
        "audio_info": {"duration": duration_ms},
        "result": {
            "text": text,
            "utterances": utterances,
        },
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file
