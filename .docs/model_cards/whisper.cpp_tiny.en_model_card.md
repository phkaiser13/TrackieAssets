# Model Card: whisper.cpp (tiny.en)

## Model Details
- **Model Name:** whisper.cpp (tiny.en model)
- **Version:** 1.0.0 (placeholder)
- **Description:** An integration of the `whisper.cpp` library for efficient, low-latency Automatic Speech Recognition (ASR). This card specifically covers the `tiny.en` model, optimized for English speech-to-text on CPU.
- **Authors:** OpenAI (original Whisper model), Georgi Gerganov (whisper.cpp), Trackie Team (integration)
- **License:** Apache-2.0 (for Trackie integration code), MIT (for whisper.cpp)

## Technical Specifications
- **Weights SHA256:** `[To be filled for ggml-tiny.en.bin upon release]`
- **Formats Available:** ggml (`.bin` files)
- **Quantization Levels:** The base model is already highly optimized; further quantization may be applied.

## Training
- **Training Data Summary:** The base model was trained by OpenAI. Fine-tuning for specific accents (e.g., PT-BR) may be performed on datasets like LibriSpeech or custom accent corpora.
- **Preprocessing:**
  - Audio is resampled to 16kHz, 16-bit PCM.
  - Voice Activity Detection (VAD) is used to segment audio into speech chunks.
- **Fine-tuning Method:** Quantization-aware and distillation-friendly fine-tuning approaches are preferred.
- **Reproducibility:**
    - **Fine-tuning Script:** `[To be provided if applicable]`
    - **Dataset Source:** `[To be documented for fine-tuning runs]`

## Usage and Limitations
- **Intended Use:** Low-latency, offline ASR for applications like voice commands, transcription, and as an input modality for TrackieLLM. Optimized for ARM (NEON) and x86 (AVX) CPUs.
- **Limitations:** The `tiny.en` model is optimized for speed and low resource usage, not maximum accuracy. It may struggle with heavy accents, noisy environments, or technical jargon. It only supports English.
- **Evaluation Metrics:** Word Error Rate (WER).

## Security and Privacy
- **Security & Privacy Notes:** All processing is performed locally and offline. No audio data is sent to external services. Users should handle audio data in accordance with privacy best practices. The model is not designed to recognize specific speakers.
