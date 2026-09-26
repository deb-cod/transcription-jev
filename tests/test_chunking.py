from app.config.loader import ChunkingConfig
from app.transcript.chunker import chunk_transcript
from app.transcript.preprocessor import preprocess_transcript


def test_speaker_aware_chunking_keeps_end_evidence() -> None:
    filler = "Agent: Please hold.\nCaller: I am reviewing the reference.\n" * 30
    ending = "Caller: This is Dr. Patel's office calling about prior authorization."
    text = preprocess_transcript(filler + ending)
    chunks = chunk_transcript(text, ChunkingConfig(True, 500, 80, True))
    assert len(chunks) > 1
    assert ending in chunks[-1].text
    assert all(len(chunk.text) <= 500 for chunk in chunks)


def test_does_not_split_short_speaker_turns() -> None:
    text = "Agent: Hello.\nCaller: I need an appointment.\nAgent: Certainly."
    chunks = chunk_transcript(text, ChunkingConfig(True, 50, 0, True))
    assert "Caller: I need an appointment." in [line for chunk in chunks for line in chunk.text.splitlines()]

