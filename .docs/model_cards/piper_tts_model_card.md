# Model Card: Piper TTS

## Model Details
- **Model Name:** Piper TTS
- **Version:** 1.0.0 (placeholder)
- **Description:** A fast, local, neural text-to-speech system based on the Piper project. This integration provides C++/Rust bindings for offline TTS synthesis with a real-time streaming API.
- **Authors:** Rhasspy (Piper), Trackie Team (integration)
- **License:** MIT (for Piper), Apache-2.0 (for integration code)

## Technical Specifications
- **Weights SHA256:** `[To be filled for each voice pack upon release]`
- **Formats Available:** TorchScript (`.pt`), ONNX, native Piper voice files (`.blob`)
- **Voice Packs:**
  - `pt_BR-faber-medium`
  - `pt_BR-edresson-low`
- **Quantization Levels:** Not specified, but models are optimized for CPU execution.

## Training
- **Training Data Summary:** The provided voice packs are pre-trained on high-quality speech corpora (e.g., public domain audiobooks). The training process follows the VITS model architecture.
- **Training Tool:** Training is done using the Piper training pipeline.
- **Reproducibility:**
    - **Training details for custom voices:** `[To be documented if new voices are trained]`

## Usage and Limitations
- **Intended Use:** High-quality, offline text-to-speech synthesis for providing audio feedback in Trackie applications. The streaming API is designed for real-time interaction.
- **Limitations:** The quality of the synthesized speech depends on the chosen voice pack (e.g., "low" quality voices are faster but less natural). Does not convey complex emotions or prosody unless specifically trained to do so.
- **Evaluation Metrics:** Mean Opinion Score (MOS) for speech naturalness.

## Security and Privacy
- **Security & Privacy Notes:** All TTS synthesis is performed locally. No text or synthesized audio is sent to external services. The system is secure for processing sensitive text, but users should be aware of the potential for misuse in generating deceptive audio content.
