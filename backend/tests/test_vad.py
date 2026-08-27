from __future__ import annotations

import json

from backend.app.adapters import asr_sentence_fixer, vad


def _utt(text: str, start: int, end: int) -> dict:
    return {"text": text, "start_time": start, "end_time": end}


def _write_asr(tmp_path, utterances: list, duration: int = 10000, text: str = "") -> tuple:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    asr_file = session / "metadata" / "asr.json"
    payload = {
        "audio_info": {"duration": duration},
        "result": {"text": text, "utterances": utterances},
    }
    asr_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return asr_file, session


def test_clamp_shrinks_utterance_to_speech_overlap():
    utts = [_utt("a", 0, 3000)]
    speech = [(500, 2000)]

    result, dropped = vad.clamp_utterances_to_speech(utts, speech)

    assert dropped == 0
    assert (result[0]["start_time"], result[0]["end_time"]) == (500, 2000)


def test_clamp_keeps_sentence_inside_speech_unchanged():
    utts = [_utt("a", 1000, 2000)]
    speech = [(500, 2500)]

    result, dropped = vad.clamp_utterances_to_speech(utts, speech)

    assert dropped == 0
    assert (result[0]["start_time"], result[0]["end_time"]) == (1000, 2000)


def test_clamp_spans_multiple_speech_intervals():
    # 一句话覆盖多段被短暂静音隔开的语音：收缩到首个重叠起点与末个重叠终点。
    utts = [_utt("a", 0, 5000)]
    speech = [(400, 1000), (2000, 3000), (4000, 4500)]

    result, dropped = vad.clamp_utterances_to_speech(utts, speech)

    assert dropped == 0
    assert (result[0]["start_time"], result[0]["end_time"]) == (400, 4500)


def test_clamp_drops_utterance_without_speech_overlap():
    utts = [_utt("a", 0, 1000), _utt("b", 1500, 2500)]
    speech = [(1600, 2200)]

    result, dropped = vad.clamp_utterances_to_speech(utts, speech)

    assert dropped == 1
    assert [u["text"] for u in result] == ["b"]
    assert (result[0]["start_time"], result[0]["end_time"]) == (1600, 2200)


def test_clamp_drops_degenerate_shrunken_utterance():
    # 与语音区间仅重叠 50ms（低于 MIN_UTTERANCE_MS）的句子按退化处理。
    utts = [_utt("a", 0, 1000), _utt("b", 1500, 3000)]
    speech = [(950, 1000), (1600, 2800)]

    result, dropped = vad.clamp_utterances_to_speech(utts, speech)

    assert dropped == 1
    assert [u["text"] for u in result] == ["b"]
    assert (result[0]["start_time"], result[0]["end_time"]) == (1600, 2800)


def test_clamp_falls_back_when_all_utterances_dropped():
    # 全部句子都无重叠视为 VAD 失效：原样返回以保护字幕。
    utts = [_utt("a", 0, 1000), _utt("b", 2000, 3000)]
    speech = [(5000, 6000)]

    result, dropped = vad.clamp_utterances_to_speech(utts, speech)

    assert dropped == 0
    assert result == utts


def test_clamp_without_speech_returns_input_unchanged():
    utts = [_utt("a", 0, 1000)]

    result, dropped = vad.clamp_utterances_to_speech(utts, [])

    assert result == utts
    assert dropped == 0


def test_clamp_never_extends_beyond_original_bounds():
    # 语音区间超出句子原边界时也不能外扩，只做收缩。
    utts = [_utt("a", 1000, 2000)]
    speech = [(0, 3000)]

    result, _ = vad.clamp_utterances_to_speech(utts, speech)

    assert (result[0]["start_time"], result[0]["end_time"]) == (1000, 2000)


def test_clamp_preserves_order_and_monotonicity():
    utts = [_utt("a", 0, 2084), _utt("b", 2084, 2313), _utt("c", 2400, 3000)]
    speech = [(300, 2000), (2100, 2900)]

    result, dropped = vad.clamp_utterances_to_speech(utts, speech)

    assert dropped == 0
    starts = [u["start_time"] for u in result]
    ends = [u["end_time"] for u in result]
    assert starts == sorted(starts)
    assert all(e > s for s, e in zip(starts, ends))
    assert all(ends[i] <= starts[i + 1] for i in range(len(result) - 1))


def test_fixer_applies_vad_and_creates_gaps(tmp_path, monkeypatch):
    # 模拟 Kotoba 连续时间戳：end_time 与下一句 start_time 相接，
    # 收缩后两句之间出现真实间隙（merge_audio 回填原声的触发条件）。
    utts = [_utt("今日もかわいいパンツだね", 0, 3000), _utt("筋金入りのシスコンだ", 3000, 6000)]
    asr_file, session = _write_asr(tmp_path, utts, duration=8000)
    vocals_file = session / "media" / "audio_vocals.wav"
    monkeypatch.setattr(vad, "speech_intervals", lambda _p: [(200, 2800), (3800, 5700)])

    fixed = asr_sentence_fixer.fix_asr_sentences(
        asr_file, session, language="ja", vocals_file=vocals_file
    )
    data = json.loads(fixed.read_text(encoding="utf-8"))

    assert data["vad"]["applied"] is True
    assert data["vad"]["speech_intervals"] == 2
    assert data["vad"]["dropped"] == 0
    out = data["result"]["utterances"]
    # VAD 收缩 [-200/+300 padding 后] 仍保留超过 400ms 的句间间隙。
    assert out[0]["start_time"] >= 100
    assert out[0]["end_time"] <= 3100
    assert out[1]["start_time"] >= 3700
    assert out[1]["end_time"] <= 6000
    assert out[1]["start_time"] - out[0]["end_time"] >= 400


def test_fixer_skips_vad_when_unavailable(tmp_path, monkeypatch):
    utts = [_utt("Hello world.", 100, 1200)]
    asr_file, session = _write_asr(tmp_path, utts)
    vocals_file = session / "media" / "audio_vocals.wav"
    monkeypatch.setattr(vad, "speech_intervals", lambda _p: None)

    fixed = asr_sentence_fixer.fix_asr_sentences(
        asr_file, session, vocals_file=vocals_file
    )
    data = json.loads(fixed.read_text(encoding="utf-8"))

    assert data["vad"] == {"applied": False, "reason": "unavailable"}
    assert data["result"]["utterances"][0]["text"] == "Hello world."


def test_fixer_without_vocals_file_keeps_previous_behavior(tmp_path):
    utts = [_utt("Hello world.", 100, 1200), _utt("How are you?", 1500, 2800)]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, start_pad=50, end_pad=100)
    data = json.loads(fixed.read_text(encoding="utf-8"))

    assert data["vad"] is None
    assert [u["text"] for u in data["result"]["utterances"]] == ["Hello world.", "How are you?"]


def test_fixer_drops_hallucinated_sentence_outside_speech(tmp_path, monkeypatch):
    # 无语音重叠的句子是 ASR 在非语音音频上的幻觉（或纯呻吟误转写）。
    utts = [_utt("本編だ", 0, 2000), _utt("ありがとう", 2500, 4000)]
    asr_file, session = _write_asr(tmp_path, utts, duration=5000)
    vocals_file = session / "media" / "audio_vocals.wav"
    monkeypatch.setattr(vad, "speech_intervals", lambda _p: [(2600, 3900)])

    fixed = asr_sentence_fixer.fix_asr_sentences(
        asr_file, session, language="ja", vocals_file=vocals_file
    )
    data = json.loads(fixed.read_text(encoding="utf-8"))

    assert data["vad"]["dropped"] == 1
    assert [u["text"] for u in data["result"]["utterances"]] == ["ありがとう"]


def test_speech_intervals_returns_none_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("VAD_ENABLED", "off")

    assert vad.speech_intervals(tmp_path / "any.wav") is None
