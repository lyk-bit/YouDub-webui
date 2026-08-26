from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from backend.app.adapters import speaker_diarization


class _FakeAnnotation:
    def __init__(self, tracks: list[tuple[float, float, str]]):
        self._tracks = tracks

    def itertracks(self, yield_label=True):
        for start, end, label in self._tracks:
            yield SimpleNamespace(start=start, end=end), "track", label


class _FakePipeline:
    def __init__(self, annotation):
        self._annotation = annotation
        self.devices: list[str] = []

    def to(self, device):
        self.devices.append(device)
        return self

    def __call__(self, path):
        return self._annotation


class _FailingPipeline:
    def to(self, device):
        return self

    def __call__(self, path):
        raise RuntimeError("boom")


def _utt(text: str, start: int, end: int, additions: dict | None = None) -> dict:
    utt = {"text": text, "start_time": start, "end_time": end}
    if additions is not None:
        utt["additions"] = additions
    return utt


def _write_asr(tmp_path, utterances: list) -> tuple:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    asr_file = session / "metadata" / "asr.json"
    payload = {
        "audio_info": {"duration": 10000},
        "result": {"text": "", "utterances": utterances},
    }
    asr_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return asr_file, session


def _install_fake_pyannote(monkeypatch, pipeline=None) -> None:
    """在 sys.modules 注入伪 pyannote.audio，绕过真实依赖。"""
    pyannote_module = types.ModuleType("pyannote")
    audio_module = types.ModuleType("pyannote.audio")
    audio_module.Pipeline = pipeline if pipeline is not None else object
    pyannote_module.audio = audio_module
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio_module)


@pytest.fixture(autouse=True)
def _reset_cached_pipeline(monkeypatch):
    monkeypatch.setattr(speaker_diarization, "_PIPELINE", None)
    monkeypatch.delenv("PYANNOTE_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)


def test_diarization_enabled_env_switch(monkeypatch):
    monkeypatch.setenv("SPEAKER_DIARIZATION", "false")
    assert speaker_diarization.diarization_enabled() is False
    monkeypatch.setenv("SPEAKER_DIARIZATION", "true")
    assert speaker_diarization.diarization_enabled() is True
    monkeypatch.delenv("SPEAKER_DIARIZATION", raising=False)
    assert speaker_diarization.diarization_enabled() is True


def test_assign_speakers_picks_longest_overlap():
    utts = [_utt("hello", 1000, 3000)]
    turns = [
        {"start": 0, "end": 1500, "label": "1"},   # overlap 500ms
        {"start": 1500, "end": 4000, "label": "2"},  # overlap 1500ms
    ]

    out = speaker_diarization.assign_speakers(utts, turns)

    assert out[0]["additions"]["speaker"] == "2"


def test_assign_speakers_defaults_without_overlap_and_keeps_additions():
    utts = [_utt("hello", 1000, 2000, {"speaker": "stale", "note": "keep"})]
    turns = [{"start": 4000, "end": 5000, "label": "2"}]

    out = speaker_diarization.assign_speakers(utts, turns)

    assert out[0]["additions"]["speaker"] == "1"
    assert out[0]["additions"]["note"] == "keep"


def test_relabel_by_duration_ranks_by_total_speech():
    turns = [
        {"start": 0, "end": 1000, "label": "SPEAKER_00"},     # 1000ms
        {"start": 2000, "end": 6000, "label": "SPEAKER_01"},  # 4000ms
        {"start": 7000, "end": 7100, "label": "SPEAKER_02"},  # 100ms
    ]

    out = speaker_diarization._relabel_by_duration(turns)

    assert [t["label"] for t in out] == ["2", "1", "3"]


def test_apply_diarization_writes_speakers(tmp_path, monkeypatch):
    utts = [
        _utt("first", 0, 2000),
        _utt("second", 2100, 4000),
    ]
    asr_file, _ = _write_asr(tmp_path, utts)
    fake = _FakePipeline(_FakeAnnotation([(0.0, 2.0, "SPEAKER_00"), (2.1, 4.0, "SPEAKER_01")]))
    monkeypatch.setattr(speaker_diarization, "_load_pipeline", lambda: (fake, None))

    message = speaker_diarization.apply_diarization(asr_file, tmp_path / "vocals.wav")

    assert "2 speakers" in message
    assert fake.devices == ["cpu"]
    data = json.loads(asr_file.read_text(encoding="utf-8"))
    assert [u["additions"]["speaker"] for u in data["result"]["utterances"]] == ["1", "2"]


def test_apply_diarization_skips_when_disabled(tmp_path, monkeypatch):
    asr_file, _ = _write_asr(tmp_path, [_utt("hello", 0, 1000)])
    before = asr_file.read_text(encoding="utf-8")
    monkeypatch.setenv("SPEAKER_DIARIZATION", "false")
    called = []
    monkeypatch.setattr(
        speaker_diarization, "_load_pipeline", lambda: (called.append(1), (None, None))[1]
    )

    message = speaker_diarization.apply_diarization(asr_file, tmp_path / "vocals.wav")

    assert message == "speaker diarization disabled"
    assert called == []
    assert asr_file.read_text(encoding="utf-8") == before


def test_apply_diarization_degrades_when_pipeline_unavailable(tmp_path, monkeypatch):
    asr_file, _ = _write_asr(tmp_path, [_utt("hello", 0, 1000)])
    before = asr_file.read_text(encoding="utf-8")
    monkeypatch.setattr(
        speaker_diarization, "_load_pipeline", lambda: (None, "pyannote.audio is not installed")
    )

    message = speaker_diarization.apply_diarization(asr_file, tmp_path / "vocals.wav")

    assert "skipped" in message and "not installed" in message
    assert asr_file.read_text(encoding="utf-8") == before


def test_apply_diarization_degrades_when_inference_fails(tmp_path, monkeypatch):
    asr_file, _ = _write_asr(tmp_path, [_utt("hello", 0, 1000)])
    before = asr_file.read_text(encoding="utf-8")
    monkeypatch.setattr(
        speaker_diarization, "_load_pipeline", lambda: (_FailingPipeline(), None)
    )

    message = speaker_diarization.apply_diarization(asr_file, tmp_path / "vocals.wav")

    assert "skipped" in message and "boom" in message
    assert asr_file.read_text(encoding="utf-8") == before


def test_apply_diarization_reports_no_turns(tmp_path, monkeypatch):
    asr_file, _ = _write_asr(tmp_path, [_utt("hello", 0, 1000)])
    before = asr_file.read_text(encoding="utf-8")
    monkeypatch.setattr(
        speaker_diarization, "_load_pipeline", lambda: (_FakePipeline(_FakeAnnotation([])), None)
    )

    message = speaker_diarization.apply_diarization(asr_file, tmp_path / "vocals.wav")

    assert "no speech turns" in message
    assert asr_file.read_text(encoding="utf-8") == before


def test_load_pipeline_skips_without_token_or_mirror(monkeypatch):
    _install_fake_pyannote(monkeypatch)
    monkeypatch.setattr(speaker_diarization, "_PIPELINE", None)

    pipeline, reason = speaker_diarization._load_pipeline()

    assert pipeline is None
    assert "PYANNOTE_HF_TOKEN" in reason


def test_load_pipeline_uses_token_when_present(monkeypatch):
    calls: dict = {}

    class _FakeHFPipeline:
        @staticmethod
        def from_pretrained(name, use_auth_token=None):
            calls["name"] = name
            calls["token"] = use_auth_token
            return "pipeline-object"

    _install_fake_pyannote(monkeypatch, pipeline=_FakeHFPipeline)
    monkeypatch.setenv("PYANNOTE_HF_TOKEN", "token-123")
    monkeypatch.setattr(speaker_diarization, "_PIPELINE", None)

    pipeline, reason = speaker_diarization._load_pipeline()

    assert pipeline == "pipeline-object"
    assert reason is None
    assert calls["token"] == "token-123"
    assert calls["name"] == "pyannote/speaker-diarization-3.1"
    assert speaker_diarization._PIPELINE == "pipeline-object"


def test_selected_device_falls_back_from_mps(monkeypatch):
    monkeypatch.setenv("DIARIZATION_DEVICE", "mps")

    assert speaker_diarization._selected_device() == "cpu"


def test_release_model_clears_cache(monkeypatch):
    fake = _FakePipeline(_FakeAnnotation([]))
    monkeypatch.setattr(speaker_diarization, "_PIPELINE", fake)

    speaker_diarization.release_model()

    assert speaker_diarization._PIPELINE is None
