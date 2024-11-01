import whisper
import os
import sounddevice as sd
import numpy as np

model = whisper.load_model("turbo")


def record_audio(duration=5, sample_rate=16000):
    audio = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32"
    )
    sd.wait()
    return np.squeeze(audio)


def prepare_audio_and_detect_language(audio):

    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language}")
    return detected_language


files = ["japanese", "russian", "german"]

for audio_file in files:

    # load audio and pad/trim it to fit 30 seconds
    file_path = os.path.join(os.getcwd(), audio_file + ".m4a")
    audio = whisper.load_audio(file_path)
    prepare_audio_and_detect_language(audio)

print("Say something...")
audio = record_audio()
prepare_audio_and_detect_language(audio)
