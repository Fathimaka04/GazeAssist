"""
verify_tts.py

Week 11: standalone test of the AI4Bharat Indic Parler-TTS model.
Tests text-to-speech generation and playback in English, Hindi, and
Malayalam - completely separate from the camera/gaze pipeline, so any
setup issues here are isolated from the working gaze system.

Does NOT touch GazeBlinkEngine, IntentFusionEngine, or the camera at
all. Just: load the model once, generate speech for one phrase per
language, save it as a .wav file, play it out loud, and print how
long each step took (first-load time is expected to be much slower
than subsequent generations, since the model has to load into GPU
memory once).

Run with:
    python verify_tts.py
(terminal only - not the VS Code Run button)
"""

import time
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import sounddevice as sd


MODEL_NAME = "ai4bharat/indic-parler-tts"

# One phrase per language to test, plus a voice "description" - the
# model uses this description to decide how the voice should sound
# (this is NOT translated or spoken itself, just a style instruction).
TEST_CASES = [
    {
        "language": "English",
        "text": "Hello, this is a test.",
        "description": "A clear, calm female voice speaking at a moderate pace.",
    },
    {
        "language": "Hindi",
        "text": "नमस्ते, यह एक परीक्षण है।",
        "description": "A clear, calm female voice speaking at a moderate pace.",
    },
    {
        "language": "Malayalam",
        "text": "ഹലോ, ഇത് ഒരു പരീക്ഷണമാണ്.",
        "description": "A clear, calm female voice speaking at a moderate pace.",
    },
]


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model (first run will also download it - this can take a while)...")
    t_load_start = time.time()

    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    t_load_end = time.time()
    print(f"Model loaded in {t_load_end - t_load_start:.1f}s\n")

    for case in TEST_CASES:
        print(f"--- {case['language']} ---")
        print(f"Text: {case['text']}")

        t_gen_start = time.time()

        input_ids = description_tokenizer(case["description"], return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(case["text"], return_tensors="pt").input_ids.to(device)

        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()

        t_gen_end = time.time()
        print(f"Generated in {t_gen_end - t_gen_start:.2f}s")

        filename = f"tts_test_{case['language'].lower()}.wav"
        sf.write(filename, audio_arr, model.config.sampling_rate)
        print(f"Saved to {filename}")

        print("Playing...")
        sd.play(audio_arr, model.config.sampling_rate)
        sd.wait()  # blocks until playback finishes before moving to the next language
        print()

    print("Done. Check tts_test_english.wav, tts_test_hindi.wav, tts_test_malayalam.wav")


if __name__ == "__main__":
    main()
    