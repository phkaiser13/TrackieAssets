# Model Card: Tesseract-OCR

## Model Details
- **Model Name:** Tesseract-OCR
- **Version:** 5.x (placeholder, depends on base Tesseract version)
- **Description:** An optical character recognition (OCR) engine. This integration focuses on using Tesseract's native capabilities, with improved language packs (e.g., for PT-BR) and a lightweight C++/Rust wrapper for the Trackie ecosystem.
- **Authors:** Google (original Tesseract), Trackie Team (integration, language pack training)
- **License:** Apache-2.0

## Technical Specifications
- **Weights SHA256:** `[To be filled for each .traineddata file upon release]`
- **Formats Available:** `.traineddata` (tessdata) language packs
- **Quantization Levels:** Not applicable (model is not a typical deep neural network).

## Training
- **Training Data Summary:** Language packs are trained or fine-tuned on a combination of synthetic text rendered with various fonts and scanned documents. Special focus on improving the PT-BR language pack.
- **Preprocessing:**
  - Image binarization and deskewing are performed before OCR.
- **Training Tool:** Tesseract's native training tools (`lstmtraining`).
- **Reproducibility:**
    - **Training Script:** `[To be provided, will wrap Tesseract tooling]`
    - **Hyperparameters File:** `[To be provided]`
    - **Dataset Source:** `[To be documented per language pack]`

## Usage and Limitations
- **Intended Use:** Offline, CPU-optimized OCR for extracting text from images. Integrated into TrackieLINK and TrackieLLM for document processing and data extraction tasks.
- **Limitations:** Accuracy depends heavily on image quality, font, and language. Preprocessing (deskewing, binarization) is critical for good performance. Not as robust as modern transformer-based document understanding models for complex layouts.
- **Evaluation Metrics:** Character Error Rate (CER), Word Error Rate (WER).

## Security and Privacy
- **Security & Privacy Notes:** All processing is done offline. The engine does not have network capabilities. Users are responsible for ensuring that the images being processed do not contain sensitive data. The underlying Tesseract library is subject to its own security maintenance.
