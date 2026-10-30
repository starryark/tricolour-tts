"""Shared constants and helpers for the Tricolour cast TTS build."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

DATASET = Path(os.environ.get("TRI_DATASET", r"F:\Tricolour\voice_dataset_game"))
WORK = Path(os.environ.get("TRI_WORK", r"F:\Tricolour\tts_work"))

# The 14 characters with >= 12 usable minutes (research/voice_model_approach_2026-10.md s2).
ROSTER = ["moxiaoju", "wenzhi", "luoyan", "chiguo", "chiyao", "yexin",
          "zhute", "huanglili", "moshu", "dinglaoshi", "linjie", "wenzhou", "xuyayi", "ciertou"]

SR = 24000
LEAD_PAD = 0.12
TAIL_PAD = 0.25
FADE = 0.005
TARGET_LUFS = -23.0
PEAK_MAX = 10 ** (-1.0 / 20)          # -1 dBFS
GROUP_GAIN_LIMIT = 12.0               # |per-(series, speaker) gain| cap, dB

MIN_DUR, MAX_DUR = 1.0, 15.0          # main pool, measured on the re-cut clip
SHORT_SHARE = 0.05                    # cap for lexical clips under 1 s

_CJK = r"[\u4e00-\u9fff]"
INTERJ = set("嗯啊哦呃唔诶欸哈呀嘿哼呜咦哇喔噢嗷呐啦嘛呢吧哎唉嘻呵吼")


def cjk_count(text: str) -> int:
    return len(re.findall(_CJK, text))


def lexical_count(text: str) -> int:
    """CJK characters that are not bare interjections."""
    return sum(1 for ch in re.findall(_CJK, text) if ch not in INTERJ)


def _year(m: re.Match) -> str:
    digits = "零一二三四五六七八九"
    return "".join(digits[int(d)] for d in m.group(1)) + "年"


def digits_to_hanzi(text: str) -> str:
    """Spoken-form numerals. Years are read digit by digit (2005年 -> 二零零五年)."""
    import cn2an
    text = re.sub(r"(\d{4})年", _year, text)
    if re.search(r"\d", text):
        text = cn2an.transform(text, "an2cn")
    return text


def normalise_text(text: str) -> str:
    """Training text, in the form inference text should take.

    Ellipses become the tokenizer's single-token forms, dash runs become ——, spaces next
    to CJK are removed (Kurisu: display spaces after punctuation caused runaways).
    """
    t = text.strip()
    t = re.sub(r"\.{2,}|。{2,}|．{2,}", "……", t)
    t = re.sub(r"…+", lambda m: "……" if len(m.group()) <= 2 else "…………", t)
    t = re.sub(r"-{2,}|—+|―+", "——", t)
    t = t.replace("~", "～")
    t = re.sub(rf"(?<={_CJK}|[，。！？、…—～：；「」“”])\s+|\s+(?={_CJK}|[，。！？、…—～：；「」“”])", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    if re.search(r"\d", t):
        t = digits_to_hanzi(t)
    return t


def text_drop_reason(raw: str, norm: str) -> str | None:
    """Lines that are not Mandarin speech to imitate."""
    latin = len(re.findall(r"[A-Za-z]", norm))
    cjk = cjk_count(norm)
    if cjk == 0 or (latin > cjk and cjk < 6):
        return "foreign_line"
    if re.search(rf"{_CJK}\s+{_CJK}", raw):
        return "spaced_line"            # song lyrics are printed with phrase spaces
    return None


def speech_bounds(wav: np.ndarray, hop: int = 240) -> tuple[int, int] | None:
    """First/last sample of audible speech: frames within 40 dB of the clip's loud frames
    (kurisu_v2/pod/p03_build.py)."""
    n = len(wav) // hop
    if n < 3:
        return None
    frames = wav[: n * hop].reshape(n, hop)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    ref = np.percentile(db, 95)
    active = np.where(db > max(ref - 40.0, -75.0))[0]
    if active.size == 0:
        return None
    start = max(0, active[0] * hop - hop)
    end = min(len(wav), (active[-1] + 2) * hop)
    return start, end


def stable_unit(key: str) -> float:
    """Deterministic [0, 1) from a string, for seeded selections that survive re-runs."""
    import hashlib
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16) / float(1 << 48)
