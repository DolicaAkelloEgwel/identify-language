import whisper
import os
import numpy as np


model = whisper.load_model("turbo")

files = ["japanese", "russian", "german"]

for audio_file in files:

    # load audio and pad/trim it to fit 30 seconds
    file_path = os.path.join(os.getcwd(), audio_file + ".m4a")
    print(file_path)
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)
