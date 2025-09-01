# Contributing to Trackie Assets

First off, thank you for considering contributing! Your help is essential for keeping this project great.

## How Can I Contribute?

### Reporting Bugs
- Please use the GitHub issue tracker and provide as much information as possible.

### Suggesting Enhancements
- Use the issue tracker to suggest new features or improvements.

### Pull Requests
We actively welcome your pull requests.
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Use the Pull Request template provided in `.github/PULL_REQUEST_TEMPLATE.md`.

## Styleguides

### Code Style
- **C/C++:** Follow the Google C++ Style Guide. Use `clang-format` to format your code.
- **Rust:** Use `rustfmt` with the default settings.
- **Python:** Follow PEP 8. Use `flake8` or `black` to check your code.
- **Git Commit Messages:** Use the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

### Model Submission Checklist
When submitting a new model or updating an existing one, please ensure you have completed the following checklist in your pull request description:

- [ ] **Model Card:** A complete model card is included in `.docs/model_cards`.
- [ ] **License:** The model weights and code are compatible with the Apache-2.0 license.
- [ ] **Dataset Provenance:** The dataset used for training is clearly documented, including its license and any privacy considerations.
- [ ] **Reproducibility:** A training script, configuration file, and environment details are provided.
- [ ] **Verification:** SHA256 checksums for all model weights are provided.
- [ ] **Inference Stub:** A minimal C++, Rust, or Python inference example is included in the `examples/` directory.
- [ ] **Responsible AI:** The model has been evaluated for potential biases and failure modes. A summary is included in the model card's "Limitations" section.

## Responsible AI
We are committed to building responsible and ethical AI. Contributors are expected to:
- **Test for Bias:** Evaluate models for potential biases across different demographics and subgroups.
- **Ensure Privacy:** Do not use personally identifiable information (PII) in training data. Anonymize or aggregate data where necessary.
- **Be Transparent:** Clearly document model limitations, potential risks, and intended use cases.

Thank you for your contributions!
