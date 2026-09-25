import os
import subprocess
from speech.processor import SpeechProcessor

def create_sample_wav(output_path: str, text: str):
    """Uses Windows SAPI text-to-speech via PowerShell to generate a real .wav file for testing."""
    ps_command = f"""
    Add-Type -AssemblyName System.Speech;
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
    $synth.SetOutputToWaveFile('{output_path}');
    $synth.Speak('{text}');
    $synth.Dispose();
    """
    subprocess.run(["powershell", "-Command", ps_command], check=True)

def main():
    audio_path = "sample_test.wav"
    test_spoken_text = "Um basically, high bias causes underfitting, and like high variance causes overfitting."
    
    print("1. Generating test audio file...")
    create_sample_wav(audio_path, test_spoken_text)

    print("2. Transcribing with Groq Whisper...")
    processor = SpeechProcessor()
    transcript = processor.transcribe_audio(audio_path)
    print(f"Transcript: \"{transcript}\"\n")

    print("3. Analyzing communication metrics...")
    metrics = processor.analyze_communication(transcript)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Cleanup temporary test audio
    if os.path.exists(audio_path):
        os.remove(audio_path)

if __name__ == "__main__":
    main()