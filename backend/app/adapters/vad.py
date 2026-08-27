from __future__ import annotations

import os
from pathlib import Path

# silero-vad 采样率与默认参数（可用环境变量覆盖）。
# VAD_THRESHOLD 低于该语音概率的片段视为非语音：静音、BGM 分离残留，
# 以及大部分无词义的呻吟/喘息会落入此区间，收缩后交给回填保留原声。
VAD_SAMPLING_RATE = 16000
VAD_DISABLED_VALUES = {"0", "false", "no", "off", "none"}
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "100"))

# 收缩后短于该时长的句子视为退化片段直接丢弃。
MIN_UTTERANCE_MS = 80


def _vad_enabled() -> bool:
    value = os.getenv("VAD_ENABLED", "").strip().lower()
    return value not in VAD_DISABLED_VALUES


def _load_mono_16k(file: Path):
    import soundfile as sf

    data, rate = sf.read(str(file), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1, dtype="float32")
    if rate != VAD_SAMPLING_RATE:
        import librosa

        data = librosa.resample(data, orig_sr=rate, target_sr=VAD_SAMPLING_RATE)
    return data, VAD_SAMPLING_RATE


def speech_intervals(vocals_file: Path) -> list[tuple[int, int]] | None:
    """在 Demucs 人声轨上运行 silero-vad，返回语音区间（毫秒，升序不重叠）。

    silero-vad 未安装或模型加载失败时返回 None，调用方跳过收缩；
    模型文件随 pip 包内置，加载过程不联网。
    """
    if not _vad_enabled():
        return None
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError:
        return None

    try:
        model = load_silero_vad()
        audio, rate = _load_mono_16k(vocals_file)
        speech = get_speech_timestamps(
            torch.from_numpy(audio),
            model,
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            sampling_rate=rate,
            return_seconds=False,
        )
    except Exception:
        return None

    return [
        (int(chunk["start"]) * 1000 // rate, int(chunk["end"]) * 1000 // rate)
        for chunk in speech
    ]


def _clip_to_speech(
    start: int, end: int, speech: list[tuple[int, int]]
) -> tuple[int, int] | None:
    """把 [start, end] 收缩到与其重叠的语音区间：仅收缩、永不外扩。"""
    clipped_start: int | None = None
    clipped_end: int | None = None
    for speech_start, speech_end in speech:
        if speech_start >= end:
            break
        if speech_end <= start:
            continue
        overlap_start = max(speech_start, start)
        overlap_end = min(speech_end, end)
        if clipped_start is None or overlap_start < clipped_start:
            clipped_start = overlap_start
        if clipped_end is None or overlap_end > clipped_end:
            clipped_end = overlap_end
    if clipped_start is None or clipped_end is None:
        return None
    return clipped_start, clipped_end


def clamp_utterances_to_speech(
    utterances: list, speech: list[tuple[int, int]]
) -> tuple[list, int]:
    """按语音区间收缩每句时间戳；无重叠（幻觉文本/纯非语音）的句子被丢弃。

    返回 (收缩后的句子列表, 丢弃数量)。全部被丢弃时视为 VAD 失效，
    原样返回输入以保护字幕。
    """
    if not speech:
        return list(utterances), 0

    result: list[dict] = []
    dropped = 0
    for utt in utterances:
        clipped = _clip_to_speech(int(utt["start_time"]), int(utt["end_time"]), speech)
        if clipped is None or clipped[1] - clipped[0] < MIN_UTTERANCE_MS:
            dropped += 1
            continue
        clipped_start, clipped_end = clipped
        result.append({**utt, "start_time": clipped_start, "end_time": clipped_end})

    if not result:
        return list(utterances), 0
    return result, dropped
