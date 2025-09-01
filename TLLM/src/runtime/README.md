# TLLM Core Runtime

This directory contains the core runtime for TrackieLLM, implemented in high-performance C++ and/or Rust.

Its responsibilities include:
- Efficiently loading quantized weights (e.g., GGUF format).
- Managing the model lifecycle.
- Exposing a C-style ABI for interoperability with other languages (Rust, C++, Python).
- Providing a streaming API for token generation.
