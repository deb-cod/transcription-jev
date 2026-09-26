from __future__ import annotations

import re


_REPEATED_ARTIFACT = re.compile(r"^(?:\[?inaudible\]?|\[?silence\]?|\.{3,})$", re.IGNORECASE)


def preprocess_transcript(transcript: str) -> str:
    """Normalize safely without summarizing or removing semantic content."""
    text = transcript.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    output: list[str] = []
    previous = None
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        # Only collapse adjacent exact duplicates and repeated non-semantic artifacts.
        if line == previous:
            continue
        if _REPEATED_ARTIFACT.fullmatch(line) and previous and _REPEATED_ARTIFACT.fullmatch(previous):
            continue
        output.append(line)
        previous = line
    return "\n".join(output).strip()

