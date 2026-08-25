from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .. import runtime_security
from ..config import MODEL_CACHE_DIR
from ..devices import device_type, resolve_device

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
_DISABLED_VALUES = {"0", "false", "no", "off", "none"}
_DEFAULT_SPEAKER = "1"

_PIPELINE = None


def diarization_enabled() -> bool:
    value = os.getenv("SPEAKER_DIARIZATION", "true").strip().lower()
    return value not in _DISABLED_VALUES


def release_model() -> None:
    global _PIPELINE
    _PIPELINE = None


def _model_name() -> str:
    return os.getenv("DIARIZATION_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL


def _hf_token() -> str | None:
    for key in ("PYANNOTE_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return None


def _prepare_hf_home() -> None:
    # 与 Kotoba-Whisper 一致，把 HuggingFace 缓存统一落在 MODEL_CACHE_DIR 下。
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "huggingface"))


def _selected_device() -> str:
    selected = resolve_device("diarization").selected
    if device_type(selected) == "mps":
        # pyannote.audio 的算子在 MPS 上不可靠，与 Whisper 同样回退 CPU。
        return "cpu"
    return selected


def _load_pipeline():
    """懒加载 pyannote pipeline；返回 (pipeline, None) 或 (None, 跳过原因)。"""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE, None

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return None, "pyannote.audio is not installed"

    token = _hf_token()
    if not token and not os.getenv("HF_ENDPOINT"):
        # 默认模型是 HuggingFace gated repo，无令牌必然 401；
        # 提前跳过，避免无凭据时的网络等待。
        return None, "set PYANNOTE_HF_TOKEN after accepting the pyannote model terms on huggingface.co, or point HF_ENDPOINT at a mirror"

    _prepare_hf_home()
    try:
        _PIPELINE = Pipeline.from_pretrained(_model_name(), use_auth_token=token or None)
    except Exception as exc:
        log.warning("Speaker diarization model failed to load: %s", exc)
        return None, f"model {_model_name()} failed to load"
    return _PIPELINE, None


def _overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _extract_turns(annotation) -> list[dict]:
    """把 pyannote annotation 展开为毫秒时间轴上的说话人片段列表。"""
    turns = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        start_ms = int(round(float(segment.start) * 1000))
        end_ms = int(round(float(segment.end) * 1000))
        if end_ms > start_ms and label:
            turns.append({"start": start_ms, "end": end_ms, "label": str(label)})
    return turns


def _relabel_by_duration(turns: list[dict]) -> list[dict]:
    """按总发言时长把 pyannote 标签重排为 "1"、"2"...，最活跃者为 "1"。"""
    totals: dict[str, int] = {}
    for turn in turns:
        totals[turn["label"]] = totals.get(turn["label"], 0) + (turn["end"] - turn["start"])
    order = sorted(totals, key=lambda label: (-totals[label], label))
    mapping = {label: str(index + 1) for index, label in enumerate(order)}
    return [{**turn, "label": mapping[turn["label"]]} for turn in turns]


def assign_speakers(utterances: list[dict], turns: list[dict]) -> list[dict]:
    """为每条 utterance 选择重叠时长最长的说话人；无重叠回退 "1"。"""
    result = []
    for utt in utterances:
        best_label, best_ms = _DEFAULT_SPEAKER, 0
        for turn in turns:
            ms = _overlap(utt["start_time"], utt["end_time"], turn["start"], turn["end"])
            if ms > best_ms:
                best_ms, best_label = ms, turn["label"]
        additions = dict(utt.get("additions") or {})
        additions["speaker"] = best_label
        result.append({**utt, "additions": additions})
    return result


def apply_diarization(asr_file: Path, vocals_file: Path) -> str:
    """对 ASR 结果标注说话人，写回 asr.json；返回用于阶段消息的描述。

    任何失败都降级为跳过（保留原有 speaker 字段），不阻断管线。
    """
    if not diarization_enabled():
        return "speaker diarization disabled"

    pipeline, reason = _load_pipeline()
    if pipeline is None:
        return f"speaker diarization skipped ({reason})"

    try:
        pipeline.to(_selected_device())
        annotation = pipeline(str(vocals_file))
    except Exception as exc:
        log.warning("Speaker diarization failed: %s", exc, exc_info=True)
        return f"speaker diarization skipped ({exc})"

    turns = _relabel_by_duration(_extract_turns(annotation))
    if not turns:
        return "speaker diarization found no speech turns"
    speakers = sorted({turn["label"] for turn in turns}, key=int)

    data = json.loads(Path(asr_file).read_text(encoding="utf-8"))
    utterances = data.get("result", {}).get("utterances") or []
    data["result"]["utterances"] = assign_speakers(utterances, turns)
    runtime_security.atomic_write_private_text(
        Path(asr_file), json.dumps(data, ensure_ascii=False, indent=2)
    )
    return f"speaker diarization: {len(speakers)} speakers detected"
