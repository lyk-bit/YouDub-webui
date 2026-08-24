from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from backend.app.adapters import whisper_asr


def test_release_model_clears_cached_model(monkeypatch):
    monkeypatch.setattr(whisper_asr, "_MODEL", object())
    monkeypatch.setattr(whisper_asr, "_KOTOBA_PIPE", object())

    whisper_asr.release_model()

    assert whisper_asr._MODEL is None
    assert whisper_asr._KOTOBA_PIPE is None


def test_kotoba_model_name_default_and_disable(monkeypatch):
    monkeypatch.delenv("KOTOBA_WHISPER_MODEL", raising=False)
    assert whisper_asr._kotoba_model_name() == whisper_asr.KOTOBA_WHISPER_DEFAULT_MODEL

    for disabled in ("off", "none", "0", " false "):
        monkeypatch.setenv("KOTOBA_WHISPER_MODEL", disabled)
        assert whisper_asr._kotoba_model_name() == ""

    monkeypatch.setenv("KOTOBA_WHISPER_MODEL", "kotoba-tech/kotoba-whisper-v1.1")
    assert whisper_asr._kotoba_model_name() == "kotoba-tech/kotoba-whisper-v1.1"


def _install_fake_transformers(monkeypatch, *, model_calls, processor_calls, fail_local):
    calls = {"load": 0}

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            model_calls.append({"name": name, **kwargs})
            if fail_local and kwargs.get("local_files_only"):
                raise OSError("not found in cached path")
            return SimpleNamespace(model=True)

    class FakeProcessorLoader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            processor_calls.append({"name": name, **kwargs})
            if fail_local and kwargs.get("local_files_only"):
                raise OSError("not found in cached path")
            return SimpleNamespace(
                tokenizer=SimpleNamespace(tok=True),
                feature_extractor=SimpleNamespace(fe=True),
            )

    fake_transformers = SimpleNamespace(
        AutoModelForSpeechSeq2Seq=FakeModelLoader,
        AutoProcessor=FakeProcessorLoader,
        pipeline=lambda *args, **kwargs: calls.setdefault("pipe", SimpleNamespace(called=True)),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(float16="float16", tensor=lambda x: x)
    )
    return fake_transformers


def test_load_kotoba_prefers_local_cache_without_network(monkeypatch):
    model_calls: list[dict] = []
    processor_calls: list[dict] = []
    _install_fake_transformers(
        monkeypatch, model_calls=model_calls, processor_calls=processor_calls, fail_local=False
    )
    monkeypatch.delenv("KOTOBA_WHISPER_MODEL", raising=False)
    monkeypatch.setattr(whisper_asr, "_KOTOBA_PIPE", None)
    monkeypatch.setattr(
        whisper_asr, "resolve_device", lambda component: SimpleNamespace(selected="cpu")
    )

    whisper_asr._load_kotoba()

    assert len(model_calls) == 1
    assert model_calls[0]["local_files_only"] is True
    assert len(processor_calls) == 1
    assert processor_calls[0]["local_files_only"] is True


def test_load_kotoba_falls_back_to_download_when_cache_missing(monkeypatch):
    model_calls: list[dict] = []
    processor_calls: list[dict] = []
    _install_fake_transformers(
        monkeypatch, model_calls=model_calls, processor_calls=processor_calls, fail_local=True
    )
    monkeypatch.delenv("KOTOBA_WHISPER_MODEL", raising=False)
    monkeypatch.setattr(whisper_asr, "_KOTOBA_PIPE", None)
    monkeypatch.setattr(
        whisper_asr, "resolve_device", lambda component: SimpleNamespace(selected="cpu")
    )

    whisper_asr._load_kotoba()

    # The model load fails locally first; the whole block falls back to the
    # network path, so the processor never attempts its local-only load.
    assert [call["local_files_only"] for call in model_calls] == [True, False]
    assert [call["local_files_only"] for call in processor_calls] == [False]


def test_load_model_removes_corrupt_cache_and_retries(monkeypatch, tmp_path):
    calls = {"count": 0}
    model = object()
    cache_file = tmp_path / "tiny.pt"
    cache_file.write_bytes(b"bad")

    def load_model(name, device, download_root=None):
        calls["count"] += 1
        assert name == "tiny"
        assert device == "cpu"
        assert download_root == str(tmp_path)
        if calls["count"] == 1:
            raise RuntimeError("SHA256 checksum does not match")
        return model

    fake_whisper = SimpleNamespace(_MODELS={"tiny": "https://example.com/tiny.pt"}, load_model=load_model)
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("WHISPER_DOWNLOAD_ROOT", str(tmp_path))
    monkeypatch.setattr(whisper_asr, "_MODEL", None)
    monkeypatch.setattr(whisper_asr, "resolve_device", lambda component: SimpleNamespace(selected="cpu"))

    assert whisper_asr._load_model() is model
    assert calls["count"] == 2
    assert not cache_file.exists()


def _fake_audio_length(monkeypatch, duration_ms: int) -> None:
    class FakeAudio:
        def __len__(self):
            return duration_ms

    monkeypatch.setattr(whisper_asr.AudioSegment, "from_file", lambda _path: FakeAudio())


def test_convert_chunks_keeps_gaps_between_chunks():
    chunks = [
        {"text": "おはようございます。", "timestamp": (0.0, 5.0)},
        {"text": "ありがとうございます。", "timestamp": (20.0, 25.0)},
    ]

    utterances = whisper_asr._convert_chunks(chunks, 25000)

    assert [(u["start_time"], u["end_time"]) for u in utterances] == [
        (0, 5000),
        (20000, 25000),
    ]


def test_convert_chunks_clamps_open_ended_midstream_chunk_to_next_start():
    # A mid-stream open-ended chunk must not extend to the full duration,
    # which would swallow the 5-10s gap and the 65-90s tail that
    # merge_audio backfills with original non-speech vocals.
    chunks = [
        {"text": "こんにちは。", "timestamp": (0.0, 5.0)},
        {"text": "さようなら。", "timestamp": (10.0, None)},
        {"text": "また明日。", "timestamp": (60.0, 65.0)},
    ]

    utterances = whisper_asr._convert_chunks(chunks, 90000)

    assert [(u["start_time"], u["end_time"]) for u in utterances] == [
        (0, 5000),
        (10000, 60000),
        (60000, 65000),
    ]


def test_convert_chunks_last_open_chunk_extends_to_audio_duration():
    chunks = [{"text": "最後のセグメントです。", "timestamp": (55.0, None)}]

    utterances = whisper_asr._convert_chunks(chunks, 90000)

    assert [(u["start_time"], u["end_time"]) for u in utterances] == [(55000, 90000)]


def test_recognize_speech_japanese_uses_kotoba_pipeline(monkeypatch, tmp_path):
    calls: list[dict] = []

    class FakeKotoba:
        def __call__(self, audio_input, return_timestamps=True):
            calls.append({"audio": audio_input, "return_timestamps": return_timestamps})
            return {
                "text": "今日はいい天気です。",
                "chunks": [
                    {"text": "今日はいい天気です。", "timestamp": (0.0, 1.25)},
                    {"text": "次の文です。", "timestamp": (2.0, None)},
                    {"text": "  ", "timestamp": (3.0, 4.0)},
                ],
            }

    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"audio")
    monkeypatch.delenv("KOTOBA_WHISPER_MODEL", raising=False)
    monkeypatch.setattr(whisper_asr, "_load_kotoba", lambda: FakeKotoba())
    monkeypatch.setattr(
        whisper_asr, "_load_model", lambda: (_ for _ in ()).throw(AssertionError("whisper must not load"))
    )
    _fake_audio_length(monkeypatch, 2500)

    monkeypatch.setattr(
        "backend.app.adapters.whisper_asr._load_mono_audio",
        lambda _file: ([0.0, 0.1, 0.2], 16000),
    )

    output = whisper_asr.recognize_speech(vocals, tmp_path, language="ja")

    assert len(calls) == 1
    assert calls[0]["return_timestamps"] is True
    assert calls[0]["audio"]["sampling_rate"] == 16000
    assert output == tmp_path / "metadata" / "asr.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audio_info"]["duration"] == 2500
    assert payload["result"]["utterances"] == [
        {"text": "今日はいい天気です。", "start_time": 0, "end_time": 1250, "words": []},
        # Open-ended trailing chunk falls back to the audio duration.
        {"text": "次の文です。", "start_time": 2000, "end_time": 2500, "words": []},
    ]


def test_recognize_speech_english_uses_whisper_even_with_kotoba_enabled(monkeypatch, tmp_path):
    calls: list[dict] = []

    class FakeModel:
        def transcribe(self, path, **kwargs):
            calls.append({"path": path, **kwargs})
            return {
                "text": "Hello world.",
                "segments": [{"text": "Hello world.", "start": 0.0, "end": 1.0, "words": []}],
            }

    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"audio")
    monkeypatch.delenv("KOTOBA_WHISPER_MODEL", raising=False)
    monkeypatch.setattr(whisper_asr, "_load_model", lambda: FakeModel())
    monkeypatch.setattr(
        whisper_asr,
        "_load_kotoba",
        lambda: (_ for _ in ()).throw(AssertionError("kotoba must not load")),
    )
    _fake_audio_length(monkeypatch, 1000)

    output = whisper_asr.recognize_speech(vocals, tmp_path, language="en")

    assert calls == [
        {"path": str(vocals), "language": "en", "word_timestamps": True, "verbose": False}
    ]
    assert output == tmp_path / "metadata" / "asr.json"


def test_recognize_speech_passes_japanese_language_to_whisper_when_kotoba_disabled(
    monkeypatch, tmp_path
):
    calls: list[dict] = []

    class FakeModel:
        def transcribe(self, path, **kwargs):
            calls.append({"path": path, **kwargs})
            return {
                "text": "今日はいい天気です。",
                "segments": [
                    {
                        "text": "今日はいい天気です。",
                        "start": 0.0,
                        "end": 1.25,
                        "words": [],
                    }
                ],
            }

    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"audio")
    monkeypatch.setenv("KOTOBA_WHISPER_MODEL", "off")
    monkeypatch.setattr(whisper_asr, "_load_model", lambda: FakeModel())
    monkeypatch.setattr(
        whisper_asr,
        "_load_kotoba",
        lambda: (_ for _ in ()).throw(AssertionError("kotoba must not load")),
    )
    _fake_audio_length(monkeypatch, 1250)

    output = whisper_asr.recognize_speech(vocals, tmp_path, language="ja")

    assert calls == [
        {
            "path": str(vocals),
            "language": "ja",
            "word_timestamps": True,
            "verbose": False,
        }
    ]
    assert output == tmp_path / "metadata" / "asr.json"
