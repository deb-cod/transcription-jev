from __future__ import annotations

from dataclasses import dataclass
import re

from app.config.loader import ChunkingConfig


SPEAKER_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9 ._'-]{0,39}:\s*")


@dataclass(frozen=True)
class TranscriptChunk:
    index: int
    text: str
    start_character: int
    end_character: int


def chunk_transcript(text: str, config: ChunkingConfig) -> list[TranscriptChunk]:
    if not text:
        return []
    if not config.enabled or len(text) <= config.max_characters:
        return [TranscriptChunk(0, text, 0, len(text))]

    units = _speaker_units(text) if config.preserve_speaker_turns else _paragraph_units(text)
    units = [piece for unit in units for piece in _split_oversized(unit, config.max_characters)]
    chunk_texts: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and _joined_length(current + [unit]) > config.max_characters:
            chunk_texts.append("\n".join(current))
            current = _overlap_tail(current, config.overlap_characters)
            while current and _joined_length(current + [unit]) > config.max_characters:
                current.pop(0)
        current.append(unit)
    if current:
        candidate = "\n".join(current)
        if not chunk_texts or candidate != chunk_texts[-1]:
            chunk_texts.append(candidate)

    chunks: list[TranscriptChunk] = []
    search_from = 0
    for index, chunk_text in enumerate(chunk_texts):
        first_line = chunk_text.splitlines()[0]
        start = text.find(first_line, max(0, search_from - config.overlap_characters))
        if start < 0:
            start = search_from
        end = min(len(text), start + len(chunk_text))
        chunks.append(TranscriptChunk(index, chunk_text, start, end))
        search_from = end
    return chunks


def estimate_chunk_count(text: str, config: ChunkingConfig) -> int:
    return len(chunk_transcript(text, config)) if text else 0


def _speaker_units(text: str) -> list[str]:
    units: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if SPEAKER_LINE.match(line) and current:
            units.append("\n".join(current))
            current = [line]
        elif line:
            current.append(line)
        elif current:
            units.append("\n".join(current))
            current = []
    if current:
        units.append("\n".join(current))
    return units or [text]


def _paragraph_units(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if part.strip()]


def _split_oversized(unit: str, maximum: int) -> list[str]:
    if len(unit) <= maximum:
        return [unit]
    speaker = ""
    match = SPEAKER_LINE.match(unit)
    content = unit
    if match:
        speaker = match.group(0)
        content = unit[match.end():]
    available = maximum - len(speaker)
    if available < 100:
        speaker = ""
        available = maximum
    words = content.split()
    parts: list[str] = []
    current = speaker
    for word in words:
        addition = (" " if current else "") + word
        if current and len(current) + len(addition) > maximum:
            parts.append(current)
            current = speaker + word
        else:
            current += addition
    if current:
        parts.append(current)
    return parts


def _overlap_tail(units: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return []
    tail: list[str] = []
    size = 0
    for unit in reversed(units):
        added = len(unit) + (1 if tail else 0)
        if tail and size + added > overlap:
            break
        tail.insert(0, unit)
        size += added
        if size >= overlap:
            break
    return tail


def _joined_length(units: list[str]) -> int:
    return sum(map(len, units)) + max(0, len(units) - 1)

