import os
import re
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

COMMON_FILLER_WORDS = [
    "um", "uh", "like", "actually", "basically", 
    "you know", "sort of", "kind of", "so yeah", "literally"
]

class SpeechProcessor:
    def __init__(self, model_name: str = "whisper-large-v3-turbo"):
        self.model_name = model_name

    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribes an audio file (.wav, .mp3, .m4a) using Groq Whisper."""
        with open(audio_file_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model=self.model_name,
                response_format="text",
                language="en"
            )
        return transcription.strip()

    def analyze_communication(self, transcript: str) -> dict:
        """Evaluates communication characteristics: filler words, vocabulary, and length."""
        words = re.findall(r"\b\w+\b", transcript.lower())
        total_words = len(words)

        # Count filler words
        filler_counts = {}
        for filler in COMMON_FILLER_WORDS:
            # Match word sequences or individual tokens
            matches = len(re.findall(r"\b" + re.escape(filler) + r"\b", transcript.lower()))
            if matches > 0:
                filler_counts[filler] = matches

        total_fillers = sum(filler_counts.values())
        filler_rate = (total_fillers / total_words * 100) if total_words > 0 else 0.0

        # Assess clarity and structure indicators
        has_intro_or_transition = any(
            token in transcript.lower() 
            for token in ["firstly", "in contrast", "for example", "specifically", "therefore"]
        )

        return {
            "transcript": transcript,
            "word_count": total_words,
            "filler_word_count": total_fillers,
            "filler_rate_percent": round(filler_rate, 2),
            "filler_breakdown": filler_counts,
            "has_structural_connectors": has_intro_or_transition
        }