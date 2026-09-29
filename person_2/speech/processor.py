"""
speech/processor.py
Minimal working stub for speech-to-text processing.
Currently a pass-through (returns the input unchanged) so the rest of the
app can run end-to-end. Replace transcribe_audio() with a real STT model
(e.g. Whisper, Google Speech-to-Text) when ready.
"""

class SpeechProcessor:
    def __init__(self):
        self.ready = True

    def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Placeholder: real implementation should run STT here (e.g. Whisper)
        and return the transcribed text.
        """
        raise NotImplementedError(
            "transcribe_audio() is a stub. Wire up a real speech-to-text model here."
        )

    def analyze_speech_characteristics(self, transcript: str) -> dict:
        """
        Placeholder for observable communication characteristics:
        relevance, structure, clarity, filler words, response length, vocabulary.
        Returns a basic word-count based stub for now.
        """
        words = transcript.split()
        filler_words = {"um", "uh", "like", "you know", "so", "actually"}
        filler_count = sum(1 for w in words if w.lower().strip(",.") in filler_words)

        return {
            "word_count": len(words),
            "filler_word_count": filler_count,
            "note": "Basic stub metrics only — expand with real NLP analysis later."
        }