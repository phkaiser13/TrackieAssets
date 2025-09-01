# Model Card: Wake Word and VAD System (wake_vad)

## Model Details
- **Model Name:** Wake Word and Voice Activity Detection (wake_vad)
- **Version:** 1.0.0 (placeholder)
- **Description:** A two-stage system for audio processing. It combines a proprietary wake word engine with an open-source Voice Activity Detection (VAD) model. The system is designed to listen for a specific wake word and then detect speech, activating downstream processes like ASR only when necessary.
- **Authors:** Picovoice (Porcupine), Silero AI Team (Silero VAD), Trackie Team (integration)
- **License:** Proprietary (for Porcupine component), MIT (for Silero VAD), Apache-2.0 (for integration code)

## Technical Specifications

### Component 1: Wake Word (Porcupine)
- **Model:** Porcupine
- **Format:** Proprietary `.pv` model file.
- **Notes:** Requires a specific license key for use in production. The integration provides a C++ wrapper.

### Component 2: Voice Activity Detection (Silero VAD)
- **Model:** Silero VAD
- **Format:** ONNX, TorchScript
- **Weights SHA256:** `[To be filled upon release]`
- **Quantization:** INT8 and other standard ONNX quantizations are applicable.

## Training
This system uses pre-trained models. Fine-tuning is not applicable for the integrated components.
- **Wake Word:** The Porcupine model is pre-trained by Picovoice. Custom wake words can be trained using their tools.
- **VAD:** The Silero VAD model is pre-trained on a large corpus of audio.

## Usage and Limitations
- **Intended Use:** Low-power, always-on listening for a wake word, followed by VAD to detect speech segments. This is used as a trigger for the ASR pipeline (e.g., `whisper.cpp`) to reduce computational load.
- **Limitations:**
  - The Porcupine component is proprietary and requires a license.
  - VAD performance can be affected by background noise. Thresholds need to be tuned for the specific environment.
  - Wake word accuracy can vary with noise and speaker accent.
- **Evaluation Metrics:** False Accept Rate (FAR), False Reject Rate (FRR) for wake word; precision/recall for VAD.

## Security and Privacy
- **Security & Privacy Notes:** This system processes audio locally and does not send it to the cloud. The wake word engine is designed to only activate on specific keywords, minimizing data processing of non-triggered speech. However, as it is always listening, clear user consent and notification are required.
