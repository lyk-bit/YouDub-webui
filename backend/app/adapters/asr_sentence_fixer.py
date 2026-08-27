from __future__ import annotations

import json
import re
from pathlib import Path

# 日语重分句参数：相邻 chunk 间隔超过该值视为不同话语，不合并；
# 合并段总时长超过该值强制截断，防止 ASR 无句读时无限合并。
_JA_MAX_MERGE_GAP_MS = 1200
_JA_MAX_MERGE_SPAN_MS = 30000

# 日语句末完结判定：命中任意一条即视为完整句，不与下一片段合并。
# 覆盖句末标点、敬体（です/ます系）、断定（だ系）、过去（た系）、
# 常见终止形词尾（う段假名/る/い）以及终助词。
_JA_COMPLETE_RE = re.compile(
    r"[。．！？!?…」』）)]\s*$"
    r"|(?:ました|ません|でした|でしょう|ましょう|でしょ|じゃん)$"
    r"|(?:です|ます)(?:か|ね|よ|な|わ)?$"
    r"|だ(?:った|ろう|よ|ね|な|ぞ|わ|けど)?$"
    r"|た(?:の?だ|ん|よ|ね|な)?$"
    r"|(?:ない|たい|られる|させる|せる|れる)$"
    r"|[うくぐすつぬぶむるい]$"
    # 终助词不收 さ/も：口语里它们多为接续、强调用法（あのさ、それでさ、それでも），
    # 按完结处理会让后续话语失去并句机会。
    r"|[かよねなわぞ]$"
)


def _ja_complete(text: str) -> bool:
    return bool(_JA_COMPLETE_RE.search(text))


def _speaker_of(utt: dict) -> str | None:
    additions = utt.get("additions")
    if isinstance(additions, dict):
        value = additions.get("speaker")
        if value is not None and str(value).strip():
            return str(value)
    return None


def _same_speaker(prev: dict, utt: dict) -> bool:
    # 任一段缺说话人标注时放行合并（沿用已有标签）；双方都有且不同则视为对话交接，不合。
    prev_speaker, next_speaker = _speaker_of(prev), _speaker_of(utt)
    return prev_speaker is None or next_speaker is None or prev_speaker == next_speaker


# 句末标点；其后位置为潜在句子边界。半角 ! ? 与全角同权。
_JA_TERMINALS = set("。．！？!?…")
# 引号与括号配对期间内部标点不算边界，闭括号始终跟随前句。
_JA_OPEN_BRACKETS = set("「『（《〈【〔(")
_JA_CLOSE_BRACKETS = set("」』）》〉】〕)")


def _ja_sentence_pieces(text: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _JA_OPEN_BRACKETS:
            depth += 1
            i += 1
        elif ch in _JA_CLOSE_BRACKETS:
            depth = max(depth - 1, 0)
            i += 1
        elif ch in _JA_TERMINALS and depth == 0:
            j = i + 1
            while j < len(text) and text[j] in _JA_TERMINALS:
                j += 1
            # 下一个字符是闭括号时不切，把闭括号留给前句；连续标点整段跳过。
            if j < len(text) and text[j] not in _JA_CLOSE_BRACKETS:
                pieces.append(text[start:j])
                start = j
            i = j
        else:
            i += 1
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def _merge_japanese_chunks(utts: list, max_gap_ms: int, max_span_ms: int) -> list:
    merged: list[dict] = []
    for utt in utts:
        if merged:
            prev = merged[-1]
            gap = utt["start_time"] - prev["end_time"]
            span = utt["end_time"] - prev["start_time"]
            # 前一段以未完结形态收尾（如名词、助词、て形结尾）且时间上连续则并入；
            # 说话人不同视为对话交接，不并，避免把插话并进别人的话轮。
            if (
                not _ja_complete(prev["text"])
                and _same_speaker(prev, utt)
                and gap <= max_gap_ms
                and span <= max_span_ms
            ):
                prev["text"] += utt["text"]
                prev["end_time"] = utt["end_time"]
                # 说话人元数据取首段；首段没有时沿用后段的
                if "additions" not in prev and "additions" in utt:
                    prev["additions"] = utt["additions"]
                continue
        merged.append(dict(utt))
    return merged


def _split_japanese_sentences(seg: dict) -> list:
    pieces = _ja_sentence_pieces(seg["text"])
    if len(pieces) <= 1:
        return [seg]

    total_chars = sum(len(p) for p in pieces)
    span = seg["end_time"] - seg["start_time"]
    out: list[dict] = []
    cursor = seg["start_time"]
    consumed = 0
    for idx, piece in enumerate(pieces):
        consumed += len(piece)
        if idx == len(pieces) - 1:
            end = seg["end_time"]
        else:
            # 按字符占比在原时间段内线性分配时间戳
            end = seg["start_time"] + int(span * consumed / total_chars)
            end = min(max(end, cursor + 1), seg["end_time"])
        out.append({**seg, "text": piece, "start_time": cursor, "end_time": end})
        cursor = end

    if any(o["end_time"] <= o["start_time"] for o in out):
        return [seg]
    return out


def _resegment_japanese(utts: list) -> list:
    merged = _merge_japanese_chunks(utts, _JA_MAX_MERGE_GAP_MS, _JA_MAX_MERGE_SPAN_MS)
    result: list[dict] = []
    for seg in merged:
        result.extend(_split_japanese_sentences(seg))
    return result


def _start_pad(idx: int, utts: list, start_pad: int, end_pad: int, min_gap: int) -> int:
    orig_start = utts[idx]["start_time"]
    if idx == 0:
        return max(0, orig_start - start_pad)

    prev_end = utts[idx - 1]["end_time"]
    gap = orig_start - prev_end
    total = start_pad + end_pad

    if gap >= total + min_gap:
        return orig_start - start_pad
    if gap > min_gap:
        share = int((gap - min_gap) * start_pad / total)
        return orig_start - share
    return prev_end + gap // 2


def _end_pad(idx: int, utts: list, duration: int, start_pad: int, end_pad: int, min_gap: int) -> int:
    orig_end = utts[idx]["end_time"]
    if idx == len(utts) - 1:
        return min(duration, orig_end + end_pad) if duration else orig_end + end_pad

    next_start = utts[idx + 1]["start_time"]
    gap = next_start - orig_end
    total = start_pad + end_pad

    if gap >= total + min_gap:
        return orig_end + end_pad
    if gap > min_gap:
        share = int((gap - min_gap) * end_pad / total)
        return orig_end + share
    return orig_end + gap // 2


def _apply_padding(utts: list, duration: int, start_pad: int, end_pad: int) -> list:
    if not utts:
        return utts

    min_gap = 50
    result = []
    for idx in range(len(utts)):
        new_start = _start_pad(idx, utts, start_pad, end_pad, min_gap)
        new_end = _end_pad(idx, utts, duration, start_pad, end_pad, min_gap)
        clamped_end = min(duration, new_end) if duration else new_end
        result.append({
            **utts[idx],
            "start_time": max(0, new_start),
            "end_time": clamped_end,
        })
    return result


def _normalize(utterances: list) -> list:
    normalized = []
    for u in utterances:
        text = u.get("text", "").strip()
        if not text:
            continue
        item = {"text": text, "start_time": u["start_time"], "end_time": u["end_time"]}
        # 保留说话人等附加元数据，供翻译与 TTS 按人区分
        additions = u.get("additions")
        if isinstance(additions, dict):
            item["additions"] = dict(additions)
        normalized.append(item)
    return normalized


def fix_asr_sentences(asr_file: Path, session: Path,
                     start_pad: int = 100, end_pad: int = 300,
                     language: str = "en") -> Path:
    output_file = session / "metadata" / "asr_fixed.json"
    if output_file.exists():
        return output_file

    data = json.loads(Path(asr_file).read_text(encoding="utf-8"))
    utterances = data["result"]["utterances"]
    duration = data.get("audio_info", {}).get("duration", 0)

    new_utts = _normalize(utterances)
    if not new_utts:
        raise RuntimeError("ASR result has no utterances.")

    if language == "ja":
        new_utts = _resegment_japanese(new_utts)

    padded = _apply_padding(new_utts, duration, start_pad, end_pad)
    payload = {
        "audio_info": data.get("audio_info", {}),
        "result": {"text": data["result"].get("text", ""), "utterances": padded},
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file
