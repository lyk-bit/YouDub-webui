from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.app.adapters import audio


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 8000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples.astype(np.float32), sample_rate)
    return path


def test_merge_tts_audio_keeps_original_audio_for_original_mode(
    monkeypatch, tmp_path
):
    session = tmp_path / "session"
    tts_dir = session / "segments" / "tts"
    translation_file = session / "metadata" / "translation.en.json"
    original = np.linspace(-0.75, 0.75, 4000, dtype=np.float32)
    original_file = _write_wav(tts_dir / "0001.wav", original)
    meaningful_file = _write_wav(tts_dir / "0002.wav", np.zeros(4000, dtype=np.float32))
    translation_file.parent.mkdir(parents=True, exist_ok=True)
    translation_file.write_text(
        json.dumps(
            {
                "translation": [
                    {
                        "dst": "（笑声）",
                        "audio_mode": "original",
                        "start_time": 0,
                        "end_time": 500,
                    },
                    {
                        "dst": "Meaningful.",
                        "audio_mode": "tts",
                        "start_time": 700,
                        "end_time": 1200,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_load(path: Path):
        samples, sample_rate = sf.read(path, dtype="float32")
        return samples, sample_rate

    monkeypatch.setattr(audio, "_load_audio", fake_load)
    monkeypatch.setattr(audio, "_audio_duration", lambda _path: (0.5, 8000))
    monkeypatch.setattr(audio, "_base_speed_factor", lambda *_args: 1.0)
    monkeypatch.setattr(
        audio,
        "_stretch_segment",
        lambda path, _ratio, _target, _cache: (sf.read(path, dtype="float32")[0], 8000),
    )

    dubbing_file, timings_file = audio.merge_tts_audio(
        translation_file,
        tts_dir,
        session,
    )

    mixed, sample_rate = sf.read(dubbing_file, dtype="float32")
    source_samples, _ = sf.read(original_file, dtype="float32")
    assert sample_rate == 8000
    assert np.allclose(mixed[:4000], source_samples, atol=1e-6)
    assert not np.allclose(mixed[:4000], np.zeros(4000, dtype=np.float32))
    timings = json.loads(timings_file.read_text(encoding="utf-8"))["translation"]
    assert timings[0]["actual_start_time"] == 0
    assert timings[0]["actual_end_time"] == 500
    assert meaningful_file.exists()


def test_base_speed_factor_ignores_original_audio(monkeypatch, tmp_path):
    original_file = tmp_path / "original.wav"
    translated_file = tmp_path / "translated.wav"
    durations = {original_file: (10.0, 8000), translated_file: (2.0, 8000)}
    monkeypatch.setattr(audio, "_audio_duration", lambda path: durations[path])

    factor = audio._base_speed_factor(
        [
            {"dst": "", "start_time": 0, "end_time": 1000},
            {"dst": "Translated.", "start_time": 1000, "end_time": 2000},
        ],
        [original_file, translated_file],
    )

    assert factor == audio.BASE_FACTOR_MIN


def test_merge_tts_audio_keeps_delayed_original_audio(monkeypatch, tmp_path):
    session = tmp_path / "session"
    tts_dir = session / "segments" / "tts"
    translation_file = session / "metadata" / "translation.en.json"
    meaningful = np.zeros(8000, dtype=np.float32)
    original = np.linspace(-0.75, 0.75, 4000, dtype=np.float32)
    _write_wav(tts_dir / "0001.wav", meaningful)
    original_file = _write_wav(tts_dir / "0002.wav", original)
    translation_file.parent.mkdir(parents=True, exist_ok=True)
    translation_file.write_text(
        json.dumps(
            {
                "translation": [
                    {"dst": "Meaningful.", "start_time": 0, "end_time": 500},
                    {"dst": "", "start_time": 500, "end_time": 1000},
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_load(path: Path):
        samples, sample_rate = sf.read(path, dtype="float32")
        return samples, sample_rate

    monkeypatch.setattr(audio, "_load_audio", fake_load)
    monkeypatch.setattr(audio, "_audio_duration", lambda path: (1.0 if path.name == "0001.wav" else 0.5, 8000))
    monkeypatch.setattr(audio, "_base_speed_factor", lambda *_args: 1.0)
    monkeypatch.setattr(
        audio,
        "_stretch_segment",
        lambda path, _ratio, _target, _cache: (sf.read(path, dtype="float32")[0], 8000),
    )

    dubbing_file, timings_file = audio.merge_tts_audio(translation_file, tts_dir, session)

    mixed, _ = sf.read(dubbing_file, dtype="float32")
    source_samples, _ = sf.read(original_file, dtype="float32")
    assert np.allclose(mixed[8000:12000], source_samples, atol=1e-6)
    timings = json.loads(timings_file.read_text(encoding="utf-8"))["translation"]
    assert timings[1]["actual_start_time"] == 1000
    assert timings[1]["actual_end_time"] == 1500


def _patch_merge_helpers(monkeypatch):
    monkeypatch.setattr(audio, "_audio_duration", lambda _path: (0.5, 8000))
    monkeypatch.setattr(audio, "_base_speed_factor", lambda *_args: 1.0)
    monkeypatch.setattr(
        audio,
        "_stretch_segment",
        lambda path, _ratio, _target, _cache: (sf.read(path, dtype="float32")[0], 8000),
    )


def _write_gap_translation(translation_file: Path) -> None:
    translation_file.parent.mkdir(parents=True, exist_ok=True)
    translation_file.write_text(
        json.dumps(
            {
                "translation": [
                    {"dst": "Hello.", "start_time": 0, "end_time": 500},
                    {"dst": "World.", "start_time": 1200, "end_time": 1700},
                ]
            }
        ),
        encoding="utf-8",
    )


def test_merge_tts_audio_backfills_gap_with_original_vocals(monkeypatch, tmp_path):
    session = tmp_path / "session"
    tts_dir = session / "segments" / "tts"
    translation_file = session / "metadata" / "translation.en.json"
    _write_wav(tts_dir / "0001.wav", np.zeros(4000, dtype=np.float32))
    _write_wav(tts_dir / "0002.wav", np.zeros(4000, dtype=np.float32))
    _write_gap_translation(translation_file)

    # Loud non-speech vocals (moans) in the 500-1200ms gap and after the last
    # segment, silence elsewhere.
    vocals = np.zeros(20000, dtype=np.float32)
    vocals[4000:9600] = 0.5
    vocals[13600:20000] = 0.4
    vocals_file = _write_wav(session / "media" / "audio_vocals.wav", vocals)

    _patch_merge_helpers(monkeypatch)

    dubbing_file, _timings_file = audio.merge_tts_audio(
        translation_file,
        tts_dir,
        session,
        original_vocals_file=vocals_file,
    )

    mixed, _ = sf.read(dubbing_file, dtype="float32")
    vocals_samples, _ = sf.read(vocals_file, dtype="float32")
    # Gap between segments, tail after the last segment, and the full length.
    assert mixed.shape[0] == 20000
    assert np.allclose(mixed[4200:9400], vocals_samples[4200:9400], atol=1e-4)
    assert np.allclose(mixed[13800:19800], vocals_samples[13800:19800], atol=1e-4)
    assert not np.allclose(mixed[4000:9600], np.zeros(5600, dtype=np.float32))


def test_merge_tts_audio_keeps_quiet_gap_silent(monkeypatch, tmp_path):
    session = tmp_path / "session"
    tts_dir = session / "segments" / "tts"
    translation_file = session / "metadata" / "translation.en.json"
    _write_wav(tts_dir / "0001.wav", np.zeros(4000, dtype=np.float32))
    _write_wav(tts_dir / "0002.wav", np.zeros(4000, dtype=np.float32))
    _write_gap_translation(translation_file)

    # Separation bleed stays below the RMS gate and must not leak into the mix.
    vocals = np.full(20000, 0.001, dtype=np.float32)
    vocals_file = _write_wav(session / "media" / "audio_vocals.wav", vocals)

    _patch_merge_helpers(monkeypatch)

    dubbing_file, _timings_file = audio.merge_tts_audio(
        translation_file,
        tts_dir,
        session,
        original_vocals_file=vocals_file,
    )

    mixed, _ = sf.read(dubbing_file, dtype="float32")
    assert mixed.shape[0] == 13600
    assert np.allclose(mixed[4000:9600], np.zeros(5600, dtype=np.float32), atol=1e-6)


def test_merge_tts_audio_does_not_backfill_inside_segment_tail(monkeypatch, tmp_path):
    session = tmp_path / "session"
    tts_dir = session / "segments" / "tts"
    translation_file = session / "metadata" / "translation.en.json"
    # TTS clips are shorter than the first segment, so the 500-1000ms stretch
    # still belongs to that segment and must stay silent instead of replaying
    # the original speech tail.
    _write_wav(tts_dir / "0001.wav", np.zeros(4000, dtype=np.float32))
    _write_wav(tts_dir / "0002.wav", np.zeros(4000, dtype=np.float32))
    translation_file.parent.mkdir(parents=True, exist_ok=True)
    translation_file.write_text(
        json.dumps(
            {
                "translation": [
                    {"dst": "Hello.", "start_time": 0, "end_time": 1000},
                    {"dst": "World.", "start_time": 1000, "end_time": 1500},
                ]
            }
        ),
        encoding="utf-8",
    )

    vocals = np.full(12000, 0.5, dtype=np.float32)
    vocals_file = _write_wav(session / "media" / "audio_vocals.wav", vocals)

    _patch_merge_helpers(monkeypatch)

    dubbing_file, _timings_file = audio.merge_tts_audio(
        translation_file,
        tts_dir,
        session,
        original_vocals_file=vocals_file,
    )

    mixed, _ = sf.read(dubbing_file, dtype="float32")
    assert mixed.shape[0] == 12000
    assert np.allclose(mixed[4000:8000], np.zeros(4000, dtype=np.float32), atol=1e-6)


def test_kotoba_style_gap_survives_pipeline_and_backfills_original_vocals(
    monkeypatch, tmp_path
):
    from backend.app.adapters import whisper_asr
    from backend.app.adapters.asr_sentence_fixer import fix_asr_sentences

    session = tmp_path / "session"
    # Kotoba-Whisper chunk output: speech 0-5s and 20-25s. The non-speech
    # vocals (moans) at 5-20s produce no chunk, so they must be backfilled
    # from the original vocal track after sentence fixing and translation.
    chunks = [
        {"text": "おはようございます。", "timestamp": (0.0, 5.0)},
        {"text": "ありがとうございます。", "timestamp": (20.0, 25.0)},
    ]
    utterances = whisper_asr._convert_chunks(chunks, 25000)
    asr_file = session / "metadata" / "asr.json"
    asr_file.parent.mkdir(parents=True, exist_ok=True)
    asr_file.write_text(
        json.dumps(
            {
                "audio_info": {"duration": 25000},
                "result": {"text": "", "utterances": utterances},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    fixed = fix_asr_sentences(asr_file, session, language="ja")
    items = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]
    translation_file = session / "metadata" / "translation.zh.json"
    translation_file.write_text(
        json.dumps(
            {
                "translation": [
                    {
                        "dst": "早上好。",
                        "start_time": item["start_time"],
                        "end_time": item["end_time"],
                    }
                    for item in items
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tts_dir = session / "segments" / "tts"
    for index in range(1, len(items) + 1):
        _write_wav(tts_dir / f"{index:04d}.wav", np.zeros(4000, dtype=np.float32))

    vocals = np.zeros(200000, dtype=np.float32)
    vocals[40000:160000] = 0.5  # loud non-speech vocals at 5-20s
    vocals_file = _write_wav(session / "media" / "audio_vocals.wav", vocals)

    _patch_merge_helpers(monkeypatch)

    dubbing_file, _timings_file = audio.merge_tts_audio(
        translation_file,
        tts_dir,
        session,
        original_vocals_file=vocals_file,
    )

    mixed, _ = sf.read(dubbing_file, dtype="float32")
    # The moan gap (minus the 300ms segment padding) is backfilled with the
    # original vocal track instead of silence.
    backfilled = mixed[43000:158000]
    assert np.allclose(backfilled, np.full_like(backfilled, 0.5), atol=1e-4)
    assert float(np.sqrt(np.mean(np.square(backfilled)))) > 0.3
