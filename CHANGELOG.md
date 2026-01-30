# Changelog

## 3.1.1 (unrelased)

- Fix transformers language
- Add initial prompt to transformers

## 3.1.0

- Refactor to dynamically load models
- Only prefer Parakeet for English (other languages don't detect reliably)
- Add `--vad-filter`, `--vad-threshold`, `--vad-min-speech-ms`, `--vad-min-silence-ms` (thanks @lmoe)
- Add `zeroconf` to Docker image

## 3.0.2

- Set `--data-dir /data` in Docker run script

## 3.0.1

- Fix model auto selection logic

## 3.0.0

- Add support for `sherpa-onnx` and Nvidia's parakeet model
- Add support for [GigaAM](https://github.com/salute-developers/GigaAM) for Russian via [`onnx-asr`](https://github.com/istupakov/onnx-asr)
- Add `--stt-library` to select speech-to-text library (deprecate `--use-transformers`)
- Default `--model` to "auto" (prefer parakeet)
- Add Docker build here
- Default `--language` to "auto"
- Add `--cpu-threads` for faster-whisper (@Zerwin)

## 2.5.0

- Add support for HuggingFace transformers Whisper models (--use-transformers)

## 2.4.0

- Add "auto" for model and beam size (0) to select values based on CPU

## 2.3.0

- Bump faster-whisper package to 1.1.0
- Supports model `turbo` for faster processing

## 2.2.0

- Bump faster-whisper package to 1.0.3

## 2.1.0

- Added `--initial-prompt` (see https://github.com/openai/whisper/discussions/963)

## 2.0.0

- Use faster-whisper PyPI package
- `--model` can now be a HuggingFace model like `Systran/faster-distil-whisper-small.en`

## 1.1.0

- Fix enum use for Python 3.11+
- Add tests and Github actions
- Bump tokenizers to 0.15
- Bump wyoming to 1.5.2

## 1.0.0

- Initial release

