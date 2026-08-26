from __future__ import annotations

import json

import pytest

from backend.app.adapters import asr_sentence_fixer


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


def test_fix_asr_sentences_passes_through_utterances(tmp_path):
    utts = [_utt("Hello world.", 100, 1200), _utt("How are you?", 1500, 2800)]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, start_pad=50, end_pad=100)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["Hello world.", "How are you?"]


def test_fix_asr_sentences_drops_empty_text(tmp_path):
    utts = [_utt("Hello.", 0, 500), _utt("   ", 600, 800), _utt("World.", 900, 1500)]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["Hello.", "World."]


def test_fix_asr_sentences_applies_padding_within_gap(tmp_path):
    utts = [_utt("a", 1000, 2000), _utt("b", 3000, 4000)]
    asr_file, session = _write_asr(tmp_path, utts, duration=5000)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, start_pad=100, end_pad=300)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert out[0]["start_time"] == 900
    assert out[0]["end_time"] == 2300
    assert out[1]["start_time"] == 2900
    assert out[1]["end_time"] == 4300


def test_fix_asr_sentences_clamps_to_duration(tmp_path):
    utts = [_utt("only", 100, 4900)]
    asr_file, session = _write_asr(tmp_path, utts, duration=5000)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, start_pad=200, end_pad=500)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert out[0]["start_time"] == 0  # 100 - 200 -> clamp 0
    assert out[0]["end_time"] == 5000


def test_fix_asr_sentences_raises_when_empty(tmp_path):
    asr_file, session = _write_asr(tmp_path, [_utt("  ", 0, 100)])

    with pytest.raises(RuntimeError):
        asr_sentence_fixer.fix_asr_sentences(asr_file, session)


def test_fix_asr_sentences_reuses_cache(tmp_path):
    utts = [_utt("hi", 0, 500)]
    asr_file, session = _write_asr(tmp_path, utts)

    first = asr_sentence_fixer.fix_asr_sentences(asr_file, session)
    first.write_text('{"already": true}', encoding="utf-8")
    second = asr_sentence_fixer.fix_asr_sentences(asr_file, session)

    assert json.loads(second.read_text(encoding="utf-8")) == {"already": True}


def test_fix_asr_sentences_preserves_speaker(tmp_path):
    utts = [
        {"text": "Hello.", "start_time": 100, "end_time": 1200,
         "additions": {"speaker": "2"}},
        {"text": "World.", "start_time": 1500, "end_time": 2800},
    ]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert out[0]["additions"]["speaker"] == "2"
    assert "additions" not in out[1]


def test_fix_asr_sentences_japanese_merge_keeps_first_speaker(tmp_path):
    utts = [
        {"text": "今日はいい天気", "start_time": 100, "end_time": 1200,
         "additions": {"speaker": "1"}},
        {"text": "です。", "start_time": 1250, "end_time": 1800,
         "additions": {"speaker": "2"}},
    ]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, language="ja")
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["今日はいい天気です。"]
    assert out[0]["additions"]["speaker"] == "1"


def test_fix_asr_sentences_japanese_merge_inherits_later_speaker(tmp_path):
    utts = [
        {"text": "今日はいい天気", "start_time": 100, "end_time": 1200},
        {"text": "です。", "start_time": 1250, "end_time": 1800,
         "additions": {"speaker": "3"}},
    ]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, language="ja")
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert out[0]["additions"]["speaker"] == "3"


def test_fix_asr_sentences_japanese_split_inherits_speaker(tmp_path):
    utts = [
        {"text": "そうですか。わかりました。", "start_time": 100, "end_time": 2200,
         "additions": {"speaker": "2"}},
    ]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, language="ja")
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["そうですか。", "わかりました。"]
    assert [u["additions"]["speaker"] for u in out] == ["2", "2"]


def test_fix_asr_sentences_merges_incomplete_japanese_chunks(tmp_path):
    utts = [_utt("今日はいい天気", 100, 1200), _utt("です。", 1250, 1800)]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(
        asr_file, session, start_pad=0, end_pad=0, language="ja"
    )
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["今日はいい天気です。"]
    assert out[0]["start_time"] == 100
    assert out[0]["end_time"] == 1800


def test_fix_asr_sentences_keeps_complete_japanese_chunks_separate(tmp_path):
    utts = [_utt("行きましょう。", 100, 1200), _utt("楽しみですね。", 1300, 2400)]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(
        asr_file, session, start_pad=0, end_pad=0, language="ja"
    )
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["行きましょう。", "楽しみですね。"]


def test_fix_asr_sentences_does_not_merge_across_large_gap(tmp_path):
    utts = [_utt("あの", 100, 500), _utt("新しいプロジェクト", 3000, 5000)]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(
        asr_file, session, start_pad=0, end_pad=0, language="ja"
    )
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["あの", "新しいプロジェクト"]


def test_fix_asr_sentences_splits_multiple_sentences_inside_japanese_chunk(tmp_path):
    utts = [_utt("そうですか。わかりました。", 100, 2200)]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(
        asr_file, session, start_pad=0, end_pad=0, language="ja"
    )
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["そうですか。", "わかりました。"]
    assert out[0]["start_time"] == 100
    assert out[1]["end_time"] == 2200
    # 时间戳按字符占比切分：连续、单调递增且覆盖原区间
    assert out[0]["end_time"] == out[1]["start_time"]
    assert out[0]["start_time"] < out[0]["end_time"] < out[1]["end_time"]


def test_fix_asr_sentences_keeps_english_utterances_unchanged(tmp_path):
    utts = [_utt("Hello world", 100, 1200), _utt("how are you", 1300, 2400)]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(
        asr_file, session, start_pad=0, end_pad=0, language="en"
    )
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["Hello world", "how are you"]
