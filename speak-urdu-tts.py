"""Generate a WAV from Urdu text using local QnVoice acoustic and vocoder assets."""

import argparse
import json
import re
import textwrap
import unicodedata
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort


SAMPLE_RATE = 22_050
DIGITS = ("صفر", "ایک", "دو", "تین", "چار", "پانچ", "چھ", "سات", "آٹھ", "نو")
CHARACTERS = str.maketrans({"ي": "ی", "ك": "ک", "“": '"', "”": '"', "’": "'", "—": "،", "–": "،", "ـ": ""})


def spoken_text(text):
    normalized = unicodedata.normalize("NFKC", text).translate(CHARACTERS)
    pieces = []
    for character in normalized:
        if character == "\u200b":
            pieces.append(" ")
        elif unicodedata.category(character) != "Cf":
            digit = unicodedata.decimal(character, None)
            pieces.append(character if digit is None else f" {DIGITS[digit]} ")
    return " ".join("".join(pieces).split())


def read_model(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def synthesize(text, assets, threads):
    text = spoken_text(text)
    if not any(character.isalpha() for character in text) or len(text) > 400:
        raise ValueError("Enter up to 400 characters of Urdu text.")
    vocab = json.loads((assets / "urdu-vocabulary.json").read_text())
    unsupported = sorted(set(text).difference(vocab))
    if unsupported:
        raise ValueError("Unsupported characters " + ", ".join(repr(character) for character in unsupported))
    acoustic = read_model(assets / "urdu-acoustic.onnx", threads)
    vocoder = read_model(assets / "urdu-vocoder.onnx", threads)
    waveform = []
    for sentence in re.split(r"(?<=[۔.!؟?])\s+", text):
        for part in textwrap.wrap(sentence, width=100, break_long_words=False, break_on_hyphens=False):
            ids = [0]
            for character in part:
                ids.extend((vocab[character], 0))
            mel = acoustic.run(["mel"], {
                "x": np.asarray([ids], dtype=np.int64),
                "x_lengths": np.asarray([len(ids)], dtype=np.int64),
                "scales": np.asarray([0.667, 1.0], dtype=np.float32),
            })[0]
            audio = vocoder.run(None, {vocoder.get_inputs()[0].name: mel})[0].reshape(-1)
            if not audio.size or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.001:
                raise ValueError("The model could not generate audible speech for this phrase.")
            if waveform:
                waveform.append(np.zeros(SAMPLE_RATE // 5, dtype=np.float32))
            waveform.append(audio)
    return np.concatenate(waveform)


def write_wav(path, audio):
    samples = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        output.writeframes(samples.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
        audio = synthesize(text, args.model_dir, args.threads)
        write_wav(args.output, audio)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{args.output}  {audio.size / SAMPLE_RATE:.2f} s")


if __name__ == "__main__":
    main()
