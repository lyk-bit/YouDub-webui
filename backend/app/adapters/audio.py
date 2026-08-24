from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from ..audio_mode import is_original_audio

BASE_FACTOR_MIN = 0.8
BASE_FACTOR_MAX = 1.2
BASE_FACTOR_SAFETY = 0.99
LOCAL_FACTOR_MIN = 0.9
LOCAL_FACTOR_MAX = 1.1
SPEED_NOOP_EPSILON = 1e-2

# Non-speech vocals (moans, laughter, breathing) fall between ASR segments.
# Gaps are backfilled with the original vocal track so those sounds survive
# dubbing, while anything quieter than this RMS level is treated as Demucs
# separation bleed and stays silent.
NON_SPEECH_RMS_MIN = 0.01
GAP_FADE_SEC = 0.01


def split_audio_by_translation(vocals_file: Path, translation_file: Path, session: Path) -> Path:
    output_dir = session / "segments" / "vocals"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(translation_file.read_text(encoding="utf-8"))
    audio = AudioSegment.from_file(vocals_file)

    for index, item in enumerate(data["translation"], start=1):
        output_file = output_dir / f"{index:04d}.wav"
        if output_file.exists():
            continue
        start = max(0, int(item["start_time"]) - 80)
        end = min(len(audio), int(item["end_time"]) + 160)
        audio[start:end].export(output_file, format="wav")

    return output_dir


def _audio_duration(file: Path) -> tuple[float, int]:
    import librosa

    y, sr = librosa.load(str(file), sr=None)
    return len(y) / sr, sr


def _load_audio(file: Path) -> tuple[np.ndarray, int]:
    import librosa

    return librosa.load(str(file), sr=None)


def _resample(y: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return y
    import librosa

    return librosa.resample(y, orig_sr=source_rate, target_sr=target_rate)


def _base_speed_factor(translation: list[dict], tts_files: list[Path]) -> float:
    cur_total = 0.0
    des_total = 0.0
    for segment, tts_file in zip(translation, tts_files):
        if is_original_audio(segment):
            continue
        dur, _ = _audio_duration(tts_file)
        cur_total += dur
        des_total += max(0.0, (segment["end_time"] - segment["start_time"]) / 1000.0)
    if cur_total <= 0:
        return 1.0
    factor = des_total / cur_total * BASE_FACTOR_SAFETY
    return max(min(factor, BASE_FACTOR_MAX), BASE_FACTOR_MIN)


def _stretch_segment(audio_file: Path, ratio: float, target_sec: float, cache_dir: Path) -> tuple[np.ndarray, int]:
    import librosa

    if abs(ratio - 1.0) < SPEED_NOOP_EPSILON:
        y, sr = librosa.load(str(audio_file), sr=None)
        return y, sr
    from audiostretchy.stretch import stretch_audio

    out_path = cache_dir / audio_file.name
    stretch_audio(str(audio_file), str(out_path), ratio=ratio)
    y, sr = librosa.load(str(out_path), sr=None)
    return y[: int(target_sec * sr)], sr


def _local_factor(current_sec: float, base: float, desired_sec: float) -> float:
    first = current_sec * base
    if first <= 1e-3:
        return 1.0
    return max(min(desired_sec / first, LOCAL_FACTOR_MAX), LOCAL_FACTOR_MIN)


def _silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def _load_mono_audio(file: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(file), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1, dtype=np.float32)
    return data, rate


def _uncovered_intervals(
    translation: list[dict], start_ms: float, end_ms: float
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    cursor = start_ms
    for item in translation:
        segment_start = float(item["start_time"])
        segment_end = float(item["end_time"])
        if segment_end <= cursor:
            continue
        if segment_start >= end_ms:
            break
        if segment_start > cursor:
            intervals.append((cursor, min(segment_start, end_ms)))
        cursor = max(cursor, segment_end)
        if cursor >= end_ms:
            break
    if cursor < end_ms:
        intervals.append((cursor, end_ms))
    return intervals


def _vocals_slice(
    vocals: np.ndarray,
    vocals_rate: int,
    start_ms: float,
    end_ms: float,
    sample_rate: int,
) -> np.ndarray | None:
    start = int(start_ms * vocals_rate / 1000.0)
    end = min(len(vocals), int(end_ms * vocals_rate / 1000.0))
    if end <= start:
        return None
    y = _resample(vocals[start:end], vocals_rate, sample_rate).astype(np.float32)
    if float(np.sqrt(np.mean(np.square(y)))) < NON_SPEECH_RMS_MIN:
        return None
    fade = min(int(GAP_FADE_SEC * sample_rate), len(y) // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    return y


def _gap_audio(
    vocals: np.ndarray | None,
    vocals_rate: int,
    translation: list[dict],
    start_ms: float,
    end_ms: float,
    sample_rate: int,
) -> tuple[np.ndarray, bool]:
    gap = _silence((end_ms - start_ms) / 1000.0, sample_rate)
    filled = False
    if vocals is None or end_ms <= start_ms:
        return gap, filled
    for uncovered_start, uncovered_end in _uncovered_intervals(translation, start_ms, end_ms):
        y = _vocals_slice(vocals, vocals_rate, uncovered_start, uncovered_end, sample_rate)
        if y is None:
            continue
        offset = int((uncovered_start - start_ms) * sample_rate / 1000.0)
        stop = min(len(gap), offset + len(y))
        if stop > offset:
            gap[offset:stop] = y[: stop - offset]
            filled = True
    return gap, filled


def merge_tts_audio(
    translation_file: Path,
    tts_dir: Path,
    session: Path,
    *,
    original_vocals_file: Path | None = None,
) -> tuple[Path, Path]:
    dubbing_file = session / "tmp" / "audio_dubbing.wav"
    timings_file = session / "metadata" / "timings.json"
    cache_dir = session / "segments" / "stretched"
    dubbing_file.parent.mkdir(parents=True, exist_ok=True)
    timings_file.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if dubbing_file.exists() and timings_file.exists():
        return dubbing_file, timings_file

    data = json.loads(translation_file.read_text(encoding="utf-8"))
    translation = data["translation"]
    tts_files = [tts_dir / f"{i:04d}.wav" for i in range(1, len(translation) + 1)]
    for path in tts_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing TTS segment: {path}")

    _, sample_rate = _audio_duration(tts_files[0])
    base = _base_speed_factor(translation, tts_files)

    vocals: np.ndarray | None = None
    vocals_rate = 0
    if original_vocals_file is not None and original_vocals_file.exists():
        vocals, vocals_rate = _load_mono_audio(original_vocals_file)

    final_audio = np.zeros(0, dtype=np.float32)
    last_end_ms = 0.0
    for segment, tts_file in zip(translation, tts_files):
        last_end_ms = final_audio.shape[0] / sample_rate * 1000.0
        real_start_ms = max(float(segment["start_time"]), last_end_ms)
        if real_start_ms > last_end_ms:
            gap, _ = _gap_audio(
                vocals, vocals_rate, translation, last_end_ms, real_start_ms, sample_rate
            )
            final_audio = np.concatenate([final_audio, gap])

        if is_original_audio(segment):
            y, source_rate = _load_audio(tts_file)
            y = _resample(y, source_rate, sample_rate)
            desired_samples = max(
                0,
                int(round((segment["end_time"] - segment["start_time"]) * sample_rate / 1000)),
            )
            y = y[:desired_samples]
            real_end_ms = real_start_ms + len(y) / sample_rate * 1000.0
        else:
            current_sec, _ = _audio_duration(tts_file)
            desired_sec = (segment["end_time"] - real_start_ms) / 1000.0
            speed = base * _local_factor(current_sec, base, desired_sec)
            target_sec = current_sec * speed
            y, source_rate = _stretch_segment(tts_file, speed, target_sec, cache_dir)
            y = _resample(y, source_rate, sample_rate)
            adjusted_sec = len(y) / sample_rate
            real_end_ms = max(real_start_ms + adjusted_sec * 1000.0, float(segment["end_time"]))

        final_audio = np.concatenate([final_audio, y])
        segment["actual_start_time"] = int(real_start_ms)
        segment["actual_end_time"] = int(real_end_ms)

    if vocals is not None:
        tail_start_ms = final_audio.shape[0] / sample_rate * 1000.0
        vocals_end_ms = len(vocals) / vocals_rate * 1000.0
        if vocals_end_ms > tail_start_ms:
            tail, filled = _gap_audio(
                vocals, vocals_rate, translation, tail_start_ms, vocals_end_ms, sample_rate
            )
            if filled:
                final_audio = np.concatenate([final_audio, tail])

    sf.write(str(dubbing_file), final_audio, sample_rate)
    timings_file.write_text(
        json.dumps({"translation": translation}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dubbing_file, timings_file
