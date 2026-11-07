# QnVoice Urdu Text to Speech

This release contains the three-step Matcha acoustic model used for new Urdu speech in QnVoice development build 14. The acoustic graph runs with ONNX Runtime and a separate HiFi-GAN vocoder. Both models run offline after setup and produce 22.05 kHz mono speech from Urdu script.

- [iOS text to speech research paper](QnVoice-Urdu-Text-to-Speech-for-iOS.pdf)
- [All versions of the preprint on Zenodo](https://doi.org/10.5281/zenodo.24029948)
- [Original Matcha model on Hugging Face](https://huggingface.co/AhsanTalal/urdu-matcha-tts/tree/d2201d251bf13bc461e7ef88ba889aa780246ee4)
- [Benchmark prompts and measurements](benchmarks)
- [Paired evaluation audio](samples/benchmark)

## Run the model

Use Python 3.11 and install the dependencies once. The following command synthesizes one of the held-out test phrases.

```sh
python3 -m pip install -r requirements.txt
shasum -a 256 -c SHA256SUMS
python3 speak-urdu-tts.py --model-dir model --text-file samples/glass-of-water.txt --output example.wav
```

The acoustic model takes token IDs, input length, and synthesis scales and returns a mel spectrogram and its frame count. The iPhone 15 Pro test measured a median of 1.30 seconds for warm synthesis across 72 runs, compared with 1.50 seconds for the original graph. The timings cover acoustic inference and vocoding after model loading. Native-speaker listening tests remain pending.

## Provenance

QnVoice retained the Matcha teacher’s text encoder and vocabulary, shortened the flow solver to three steps, and trained a convolutional mel correction on synthetic teacher outputs. See [NOTICE](NOTICE) for model and dataset attribution and [SHA256SUMS](SHA256SUMS) for the exact files. The upstream Matcha graph is Apache 2.0 and the HiFi-GAN vocoder is MIT. The upstream dataset cards list CC BY 4.0. The paper describes the remaining review of gated access terms and speaker rights for commercial release.
